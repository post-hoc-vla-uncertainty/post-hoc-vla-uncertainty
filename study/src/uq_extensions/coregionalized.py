"""Focused sparse LMC comparison, with full joint-output covariance for IVR."""
from pathlib import Path
import time
import numpy as np
import torch
import gpytorch
from sklearn.metrics import r2_score
from .common import atomic_json,atomic_npz,read_json,isolated_rng,identity,array_hash,plain
from .preprocessing import Projection,matrix
from .target_gp import TargetGP,split_episodes
from .kernel_variance import geometry_lengthscale,nonnegative_variance


class LMC(gpytorch.models.ApproximateGP):
    def __init__(self,inducing,outputs,rank,jitter=1e-6):
        batch=torch.Size([rank]);points=inducing.unsqueeze(0).repeat(rank,1,1)
        q=gpytorch.variational.CholeskyVariationalDistribution(len(inducing),batch_shape=batch)
        base=gpytorch.variational.VariationalStrategy(self,points,q,learn_inducing_locations=False,jitter_val=jitter)
        strategy=gpytorch.variational.LMCVariationalStrategy(base,num_tasks=outputs,num_latents=rank,latent_dim=-1,jitter_val=jitter)
        super().__init__(strategy)
        self.mean_module=gpytorch.means.ZeroMean(batch_shape=batch)
        self.covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=inducing.shape[1],batch_shape=batch),batch_shape=batch)

    def forward(self,x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x),self.covar_module(x))


class CoregionalizedGP(TargetGP):
    def tensor(self,x):return torch.as_tensor(x,dtype=getattr(torch,self.gp.precision),device=self.device)

    def _train_joint(self,z,y,*,phase='full',validation=None):
        x=self.tensor(z);y=self.tensor(y)
        seed=self.features.seed if self.gp.training_seed is None else self.gp.training_seed
        ids=np.sort(np.random.default_rng(seed).choice(len(x),min(self.gp.inducing_points,len(x)),replace=False))
        with isolated_rng(seed):
            model=LMC(x[ids],y.shape[1],min(self.gp.coreg_rank,y.shape[1]),self.gp.jitter).to(device=self.device,dtype=x.dtype)
            likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y.shape[1],rank=0,has_global_noise=False,
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)).to(device=self.device,dtype=x.dtype)
            length,_=geometry_lengthscale(x)
            model.covar_module.base_kernel.lengthscale=length
            model.covar_module.outputscale=1.
            likelihood.task_noises=1e-3
            optimizer=torch.optim.Adam([*model.parameters(),*likelihood.parameters()],lr=self.gp.learning_rate)
            mll=gpytorch.mlls.VariationalELBO(likelihood,model,num_data=len(x))
            model.train();likelihood.train();history=[]
            from .batched_gp import atomic_torch
            checkpoint=Path(self.training_directory)/phase/'joint.pt' if self.training_directory else None
            key=identity(dict(z=array_hash(np.asarray(z)),y=array_hash(y.cpu().numpy()),gp=plain(self.gp),seed=seed))
            milestones=[];offset=0;elapsed=0.
            if checkpoint and checkpoint.exists():
                saved=torch.load(checkpoint,map_location=self.device,weights_only=True)
                if saved['identity']!=key:raise RuntimeError('LMC optimizer checkpoint identity changed')
                model.load_state_dict(saved['model']);likelihood.load_state_dict(saved['likelihood'])
                optimizer.load_state_dict(saved['optimizer']);history=saved['history'];milestones=saved['milestones']
                offset=len(history);elapsed=saved['seconds']
            started=time.perf_counter()
            def save():
                if checkpoint:
                    atomic_torch(checkpoint,dict(identity=key,model=model.state_dict(),likelihood=likelihood.state_dict(),
                        optimizer=optimizer.state_dict(),history=history,milestones=milestones,seconds=elapsed+time.perf_counter()-started))
                    atomic_json(checkpoint.with_suffix('.json'),dict(step=len(history),milestones=milestones,
                        seconds=elapsed+time.perf_counter()-started))
            # BEGIN RUN PROGRESS
            from run_progress import Progress
            progress=Progress('coregionalized GP',self.gp.fit_steps,initial=offset,every=min(100,self.gp.checkpoint_every))
            # END RUN PROGRESS
            for step in range(offset+1,self.gp.fit_steps+1):
                optimizer.zero_grad();total=0.
                for start in range(0,len(x),self.gp.algebra_batch_size):
                    end=min(len(x),start+self.gp.algebra_batch_size)
                    loss=-mll(model(x[start:end]),y[start:end])*(end-start)/len(x)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite LMC ELBO')
                    loss.backward();total+=float(loss.detach())
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite LMC gradient')
                optimizer.step();history.append(total)
                if step in self.gp.diagnostic_steps or step==self.gp.fit_steps:
                    record=dict(step=step,objective=total,seconds=elapsed+time.perf_counter()-started)
                    if validation is not None:
                        model.eval();likelihood.eval()
                        prediction=self._means(model,validation[0])
                        record['validation_r2']=r2_score(validation[1],prediction,multioutput='raw_values').tolist()
                        model.train();likelihood.train()
                    milestones.append(record)
                if step%self.gp.checkpoint_every==0 or step==self.gp.fit_steps:save()
                # BEGIN RUN PROGRESS
                progress.update(detail=f'negative ELBO={total:.6g}')
                # END RUN PROGRESS
        model.eval();likelihood.eval()
        # BEGIN RUN PROGRESS
        progress.finish(f'final negative ELBO={history[-1]:.6g}')
        # END RUN PROGRESS
        self.training_records[phase]=[dict(outputs=[0,y.shape[1]],steps=len(history),milestones=milestones,
            seconds=elapsed+time.perf_counter()-started,resumed_from_step=offset)]
        return model,likelihood,history

    @torch.no_grad()
    def _means(self,model,z):
        return np.concatenate([model(q).mean.cpu().numpy() for q in self.tensor(z).split(256)])

    def fit(self,x,y,*,tasks,groups=None,mapping_check=True):
        started=time.perf_counter();x,y=matrix(x),matrix(y)
        _,inverse,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
        for g in np.flatnonzero(counts>1):
            if np.max(np.ptp(y[inverse==g],axis=0))>1e-5:raise ValueError('Repeated inputs have different downstream targets: missing conditioning information')
        mapping={'performed':False}
        if mapping_check:
            fit,test=split_episodes(tasks,self.features.seed if self.gp.split_seed is None else self.gp.split_seed,groups)
            projection,output,targets=self._geometry(x[fit],y[fit])
            z=projection.transform(x[fit]);zt=projection.transform(x[test]);expected=output.transform(y[test])
            model,likelihood,_=self._train_joint(z,targets,phase='mapping_validation',validation=(zt,expected))
            prediction=self._means(model,zt)
            linear=np.linalg.lstsq(np.column_stack([np.ones(len(z)),z]),targets,rcond=None)[0]
            r2=r2_score(expected,prediction,multioutput='raw_values')
            mapping=dict(performed=True,gp_r2=r2.tolist(),linear_r2=r2_score(expected,np.column_stack([np.ones(len(zt)),zt])@linear,multioutput='raw_values').tolist(),
                         usable=bool(np.median(r2)>0),fit_indices=fit.tolist(),test_indices=test.tolist(),grouped_by_episode=groups is not None,transforms_fit_only_on_training_split=True)
            del model,likelihood
        self.projection,self.output_projection,targets=self._geometry(x,y)
        self.model,self.likelihood,history=self._train_joint(self.projection.transform(x),targets)
        self.diagnostics=dict(method='kv_t_coregionalized',output_count=targets.shape[1],support_rows=len(x),
            coreg_rank=min(self.gp.coreg_rank,targets.shape[1]),inference='sparse',output_projection=self.output_projection.method,
            mapping_check=mapping,fit_seconds=time.perf_counter()-started,loss_history=history,
            latent_variance_excludes_noise=True,inducing_locations_frozen=True,optimizer_steps=self.gp.fit_steps,
            training_records=self.training_records,geometry=self.geometry_diagnostics)
        self.diagnostics.update(noise_variances=self.likelihood.task_noises.detach().cpu().tolist(),
            lmc_coefficients=self.model.variational_strategy.lmc_coefficients.detach().cpu().tolist(),
            lengthscales=self.model.covar_module.base_kernel.lengthscale.detach().cpu().tolist(),input_components=self.projection.weights.shape[1])
        return self

    @torch.no_grad()
    def score(self,x,batch_size=256):
        z=self.tensor(self.projection.transform(x))
        base=self.model.variational_strategy.base_variational_strategy
        coefficients=self.model.variational_strategy.lmc_coefficients
        # Numerical diagonal jitter is not latent function uncertainty.
        return np.concatenate([nonnegative_variance((base(q).variance-base.jitter_val).T@coefficients.square()).mean(-1).cpu().numpy() for q in z.split(batch_size)])

    @torch.no_grad()
    def joint_covariance(self,x,y):
        x=self.tensor(self.projection.transform(x));y=self.tensor(self.projection.transform(y))
        latent=self.model.variational_strategy.base_variational_strategy(torch.cat([x,y]))
        kernel=latent.lazy_covariance_matrix[...,:len(x),len(x):].to_dense()
        coefficients=self.model.variational_strategy.lmc_coefficients
        return torch.einsum('qnm,qi,qj->nimj',kernel,coefficients,coefficients).reshape(len(x)*coefficients.shape[1],len(y)*coefficients.shape[1])

    def covariance_components(self):
        m=self.diagnostics['output_count']
        return [(self.joint_covariance,self.likelihood.task_noises.detach().cpu().numpy(),1/m)]

    def save(self,path):
        path=Path(path)
        arrays={**self.projection.arrays(),**self.output_projection.arrays('output_')}
        for kind,obj in (('model',self.model),('likelihood',self.likelihood)):
            arrays.update({kind+'::'+k:v.detach().cpu().numpy() for k,v in obj.state_dict().items()})
        atomic_npz(path/'estimator.npz',**arrays);atomic_json(path/'estimator_fit.json',self.diagnostics)

    @classmethod
    def load(cls,path,feature_config,gp_config,*,device='cpu'):
        obj=cls(feature_config,gp_config,device=device);path=Path(path);obj.diagnostics=read_json(path/'estimator_fit.json')
        with np.load(path/'estimator.npz',allow_pickle=False) as a:
            obj.projection=Projection.restore(a,gp_config.input_projection)
            obj.output_projection=Projection.restore(a,obj.diagnostics['output_projection'],'output_')
            inducing=obj.tensor(a['model::variational_strategy.base_variational_strategy.inducing_points'][0])
            m=obj.diagnostics['output_count'];rank=obj.diagnostics['coreg_rank']
            with isolated_rng(feature_config.seed):
                obj.model=LMC(inducing,m,rank,gp_config.jitter).to(device=device,dtype=inducing.dtype)
                obj.likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=m,rank=0,has_global_noise=False,
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)).to(device=device,dtype=inducing.dtype)
            for kind,module in (('model',obj.model),('likelihood',obj.likelihood)):
                module.load_state_dict({k.split('::',1)[1]:torch.as_tensor(a[k],device=device) for k in a.files if k.startswith(kind+'::')})
                module.eval()
        return obj
