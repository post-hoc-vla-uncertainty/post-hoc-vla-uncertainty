"""Episode-level estimators over observation, flow-group, or patch features."""
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import torch
from .common import atomic_json, atomic_npz, array_hash, read_json
from .cache_lock import cache_lock
from .config import is_target_arm
from .kernel_variance import KernelVariance, geometry_lengthscale, rbf, stable_cholesky
from .preprocessing import fit_projection, UninformativeBank
from .target_gp import TargetGP


def flatten_groups(x):
    x=np.asarray(x)
    if x.ndim==2:return x,1
    if x.ndim!=3:raise ValueError('Expected episode features or episode/probe/features')
    return x.reshape(-1,x.shape[-1]),x.shape[1]


def probe_weights(count,features):
    times=len(features.flow_indices)
    if count==1:return np.ones(1)
    if count%times:raise ValueError('Probe count does not match declared time grid')
    w=np.asarray(features.flow_weights or [1.]*times,dtype=float)
    return np.tile(w/w.sum(),count//times)/(count//times)


class EpisodeEstimator:
    def __init__(self,features,gp,*,target=False,device='cpu'):
        self.features,self.gp,self.target=features,gp,target
        cls=TargetGP if target else KernelVariance
        if target and gp.output_dependence=='coregionalized':
            from .coregionalized import CoregionalizedGP
            cls=CoregionalizedGP
        self.model=cls(features,gp,device=device)

    def fit(self,x,y=None,*,tasks=None):
        x,p=flatten_groups(x)
        if self.target:
            y,q=flatten_groups(y)
            if p!=q:raise ValueError('Input and target flow probes must be aligned')
            self.model.fit(x,y,tasks=np.repeat(tasks,p),groups=np.repeat(np.arange(len(tasks)),p))
        elif self.features.compression.startswith('pls'):
            y,q=flatten_groups(y)
            if p!=q:raise ValueError('PLS input and target observations must align')
            self.model.fit(x,y)
        else:self.model.fit(x)
        self.diagnostics={**self.model.diagnostics,'support_probes_per_episode':p,'support_episodes':len(x)//p}
        return self

    def score(self,x):
        n=len(x);x,p=flatten_groups(x)
        return self.model.score(x).reshape(n,p)@probe_weights(p,self.features)

    def covariance_components(self):return self.model.covariance_components()

    @torch.no_grad()
    def prepare_ivr(self,reference,candidates):
        """Project and factor all queried inputs once; covariance blocks stay lazy."""
        from .kernel_variance import rbf
        n=len(reference);shape=candidates.shape
        raw=np.concatenate([reference,candidates.reshape(-1,shape[-1])])
        z=self.model.projection.transform(raw)
        components=[]
        if self.target and self.gp.output_dependence=='coregionalized':
            gp=self.model;strategy=gp.model.variational_strategy
            covariance=strategy.base_variational_strategy(gp.tensor(z)).lazy_covariance_matrix
            coefficients=strategy.lmc_coefficients.detach()
            def joint(a,b):
                ia=torch.as_tensor(np.asarray(a).reshape(-1).astype(np.int64),device=coefficients.device)
                ib=torch.as_tensor(np.asarray(b).reshape(-1).astype(np.int64),device=coefficients.device)
                kernel=covariance[...,ia[:,None],ib[None,:]].to_dense()
                kernel=kernel-strategy.base_variational_strategy.jitter_val*(ia[:,None]==ib[None,:])
                return torch.einsum('qnm,qi,qj->nimj',kernel,coefficients,coefficients).reshape(len(ia)*coefficients.shape[1],len(ib)*coefficients.shape[1])
            components=[(joint,gp.likelihood.task_noises.detach().cpu().numpy(),1/gp.diagnostics['output_count'])]
        else:
            models=self.model.models if self.target else [self.model]
            for model in models:
                points=model.tensor(z);sparse=self.gp.inference=='sparse'
                length=model.length if self.target else model.lengthscale
                amplitude=model.amplitude if self.target else 1.
                if sparse:
                    lu,ls=model.terms[:2] if self.target else (model.lu,model.ls)
                    minus=torch.linalg.solve_triangular(lu,rbf(model.inducing,points,length,amplitude).to(lu.dtype),upper=False)
                    plus=torch.linalg.solve_triangular(ls,minus.to(ls.dtype),upper=False)
                else:
                    factor=model.terms[0] if self.target else model.factor
                    support=model.x if self.target else model.z
                    minus=torch.linalg.solve_triangular(factor,rbf(support,points,length,amplitude).to(factor.dtype),upper=False);plus=None
                def covariance(a,b,points=points,minus=minus,plus=plus,length=length,amplitude=amplitude):
                    ia=torch.as_tensor(np.asarray(a).reshape(-1).astype(np.int64),device=points.device)
                    ib=torch.as_tensor(np.asarray(b).reshape(-1).astype(np.int64),device=points.device)
                    result=rbf(points[ia],points[ib],length,amplitude)-minus[:,ia].T@minus[:,ib]
                    if plus is not None:result+=plus[:,ia].T@plus[:,ib]
                    return result
                noise=float(model.noise) if self.target else self.gp.ridge
                components.append((covariance,np.asarray([noise]),1/len(models)))
        return np.arange(n)[:,None],np.arange(n,n+np.prod(shape[:2])).reshape(*shape[:2],1),components

    def save(self,path):
        self.model.save(path)
        atomic_json(Path(path)/'episode_estimator.json',self.diagnostics)

    @classmethod
    def load(cls,path,features,gp,*,target=False,device='cpu'):
        obj=cls(features,gp,target=target,device=device)
        obj.model=type(obj.model).load(path,features,gp,device=device)
        obj.diagnostics=read_json(Path(path)/'episode_estimator.json')
        if hasattr(obj.model, 'prediction_numerics'):
            obj.diagnostics['prediction_numerics'] = obj.model.prediction_numerics
        return obj


class PatchEstimator:
    """All permitted patches condition a streamed sparse posterior per camera.

    The training-only geometry sample is explicitly recorded. It is not the
    conditioning bank: every distinct projected support patch updates the latter.
    """
    def __init__(self,features,gp,*,device='cpu'):
        self.features=features;self.gp=replace(gp,inference='sparse',inducing_points=512,
            sparse_justification='Native patch support exceeds episode-scale exact inference; fixed 512 per camera')
        self.device=device;self.models=[]

    def fit(self,x,**kwargs):
        if x.ndim!=4:raise ValueError('Expected episodes/cameras/patches/features')
        n,cameras,patches,width=x.shape
        # BEGIN RUN PROGRESS
        from run_progress import Progress
        progress=Progress('patch GP',cameras,unit='cameras',every=1)
        # END RUN PROGRESS
        for camera in range(cameras):
            rng=np.random.default_rng((self.features.seed if self.features.geometry_seed is None else self.features.geometry_seed)+camera)
            sample=rng.choice(n*patches,min(n*patches,self.features.geometry_samples),replace=False)
            geometry=np.asarray(x[sample//patches,camera,sample%patches,:])
            model=KernelVariance(self.features,self.gp,device=self.device)
            model.projection=fit_projection(geometry,method=self.features.compression,components=self.features.components,
                                           seed=self.features.seed,variance_floor=self.features.variance_floor)
            locations=np.unique(model.projection.transform(geometry),axis=0)
            if self.features.compression.endswith('_whiten'):
                base=fit_projection(geometry,method=self.features.compression.removesuffix('_whiten'),
                    components=self.features.components,seed=self.features.seed,variance_floor=self.features.variance_floor)
                _,indices=np.unique(base.transform(geometry),axis=0,return_index=True)
                locations=model.projection.transform(geometry)[indices]
            model.lengthscale,diagnostics=geometry_lengthscale(model.tensor(locations))
            picked=np.sort(rng.choice(len(locations),min(512,len(locations)),replace=False))
            model.inducing=model.tensor(locations[picked]);model.sparse=True
            model.lu,jitter=stable_cholesky(rbf(model.inducing,model.inducing,model.lengthscale),self.gp.jitter)
            model.dtype=model.lu.dtype;model.inducing=model.inducing.to(model.dtype)
            sufficient=torch.eye(len(picked),device=model.device,dtype=model.dtype)
            # BEGIN RUN PROGRESS
            camera_progress=Progress(f'patch GP camera {camera}',n,unit='support observations',every=128)
            # END RUN PROGRESS
            seen=set()
            for start in range(0,n,8):
                rows=model.projection.transform(np.asarray(x[start:start+8,camera]).reshape(-1,width))
                fresh=[]
                for row in np.unique(rows,axis=0):
                    key=row.tobytes()
                    if key not in seen:seen.add(key);fresh.append(row)
                if not fresh:continue
                for block in model.tensor(np.stack(fresh)).split(self.gp.algebra_batch_size):
                    a=torch.linalg.solve_triangular(model.lu,rbf(model.inducing,block,model.lengthscale),upper=False)
                    sufficient+=a@a.T/self.gp.ridge
                # BEGIN RUN PROGRESS
                camera_progress.update(min(n,start+8),f'unique conditioned patches={len(seen)}')
                # END RUN PROGRESS
            model.ls,_=stable_cholesky(sufficient,self.gp.jitter)
            # BEGIN RUN PROGRESS
            camera_progress.finish()
            # END RUN PROGRESS
            model.z=model.tensor(np.empty((0,locations.shape[1])))
            model.diagnostics={**diagnostics,'precision':str(model.dtype),'inference':'sparse','lengthscale':model.lengthscale,
                'ridge':self.gp.ridge,'jitter_used':jitter,'support_rows':n*patches,'unique_rows':len(seen),
                'support_episodes':n,'inducing_points':len(picked),'requested_inducing_points':512,
                'geometry_fit_rows':len(sample),'components':locations.shape[1],'all_unique_patches_conditioned':True}
            model.diagnostics.update(model.projection.diagnostics)
            # BEGIN RUN PROGRESS
            progress.update(camera+1,f'conditioned {len(seen)} patches; inducing={len(picked)}')
            # END RUN PROGRESS
            self.models.append(model)
        # BEGIN RUN PROGRESS
        progress.finish()
        # END RUN PROGRESS
        self.diagnostics={'method':'patch_kv','cameras':[m.diagnostics for m in self.models],
                          'aggregation':self.features.patch_aggregation,'camera_aggregation':self.features.camera_aggregation}
        return self

    def patch_scores(self,x):
        if getattr(self,'cache_directory',None) is not None:
            path=self.cache_directory/'patch_score_cache'/(array_hash(x)+'.npz')
            with cache_lock(path.with_suffix('.lock'),label='shared patch variances'):
                if path.exists():
                    with np.load(path,allow_pickle=False) as saved:return saved['variances']
                values=self._patch_scores(x)
                atomic_npz(path,variances=values)
                return values
        return self._patch_scores(x)

    def _patch_scores(self,x):
        n,c,p,d=x.shape
        out=np.empty((n,c,p),dtype=np.float32)
        for camera,model in enumerate(self.models):
            for start in range(0,n,8):
                rows=np.asarray(x[start:start+8,camera]).reshape(-1,d)
                out[start:start+8,camera]=model.score(rows).reshape(-1,p)
        return out

    def aggregate(self,values):
        if self.features.patch_aggregation=='mean':cameras=values.mean(-1)
        elif self.features.patch_aggregation=='max':cameras=values.max(-1)
        else:
            count=max(1,math.ceil(values.shape[-1]*self.features.patch_top_fraction))
            cameras=np.partition(values,-count,axis=-1)[...,-count:].mean(-1)
        return cameras.max(-1) if self.features.camera_aggregation=='max' else cameras.mean(-1)

    def score(self,x):return self.aggregate(self.patch_scores(x))

    def save(self,path):
        for i,m in enumerate(self.models):m.save(Path(path)/f'camera_{i}')
        atomic_json(Path(path)/'episode_estimator.json',self.diagnostics)
        self.cache_directory=Path(path)

    @classmethod
    def load(cls,path,features,gp,*,device='cpu',**kwargs):
        obj=cls(features,gp,device=device);obj.diagnostics=read_json(Path(path)/'episode_estimator.json')
        obj.diagnostics.update(aggregation=features.patch_aggregation,camera_aggregation=features.camera_aggregation)
        obj.models=[KernelVariance.load(Path(path)/f'camera_{i}',features,obj.gp,device=device) for i in range(len(obj.diagnostics['cameras']))]
        obj.cache_directory=Path(path)
        return obj


def make_estimator(config,*,device='cpu',path=None):
    if config.features.bank=='visual_patches':
        if is_target_arm(config.arm):raise ValueError('No implicit patch KV-T targets')
        cls=PatchEstimator;kwargs={}
    else:cls=EpisodeEstimator;kwargs={'target':is_target_arm(config.arm)}
    if path is not None:return cls.load(path,config.features,config.gp,device=device,**kwargs)
    return cls(config.features,config.gp,device=device,**kwargs)
