"""Task-balanced, label-free greedy integrated variance reduction."""
import numpy as np
import torch

from .kernel_variance import nonnegative_variance
from .protocol import task_weights


def _scalar_ivr(estimator, reference_features, reference_tasks, candidate_features, candidate_ids, *, count=5, seed=42):
    ids = list(candidate_ids)
    if len(ids) != len(set(ids)) or not count <= len(ids):
        raise ValueError("IVR requires enough distinct candidate IDs")
    order = np.argsort(ids)
    ids = [ids[i] for i in order]
    candidates = np.asarray(candidate_features)[order]
    rc = estimator.cross_covariance(reference_features, candidates).clone()
    cc = estimator.cross_covariance(candidates, candidates).clone()
    cc = (cc + cc.T) / 2
    w = torch.as_tensor(task_weights(reference_tasks), device=rc.device, dtype=rc.dtype)
    initial = float(np.dot(task_weights(reference_tasks), estimator.score(reference_features)))
    rng = np.random.default_rng(seed)
    available = np.ones(len(ids), dtype=bool)
    chosen, records = [], []
    integrated = initial
    for _ in range(count):
        diag = nonnegative_variance(cc.diagonal())
        denominator = diag + estimator.gp.ridge
        gains = (w[:, None] * rc.square()).sum(0) / denominator
        values = gains.cpu().numpy()
        maximum = values[available].max()
        ties = np.flatnonzero(available & np.isclose(values, maximum, rtol=1e-8, atol=1e-12))
        index = int(rng.choice(ties))
        gain = float(values[index])
        chosen.append(ids[index]); available[index] = False
        records.append(dict(id=ids[index], gain=gain, tied_ids=[ids[i] for i in ties],
                            integrated_variance_before=integrated, integrated_variance_after=integrated - gain))
        integrated -= gain
        cross = cc[:, index].clone()
        rc -= rc[:, index:index + 1].clone() @ cross[None, :] / denominator[index]
        cc -= cross[:, None] @ cross[None, :] / denominator[index]
        cc = (cc + cc.T) / 2
        nonnegative_variance(cc.diagonal())
    return chosen, {"rule": "KV-IVR", "seed": seed, "initial_integrated_variance": initial,
                    "final_integrated_variance": integrated, "steps": records,
                    "conditioning_updated": False, "reference_task_count": len(set(reference_tasks))}


@torch.no_grad()
def select_ivr(estimator, reference_features, reference_tasks, candidate_features, candidate_ids, *, count=5, seed=42):
    """Condition on entire demonstration probe groups, using each output's noise.

    Reference/candidate covariance blocks are evaluated in bounded chunks. No
    labels are consumed and the fitted estimator is never mutated. This also
    supports the joint point/output covariance of the coregionalized model.
    """
    from .estimators import probe_weights
    if not hasattr(estimator, 'covariance_components'):
        return _scalar_ivr(estimator,reference_features,reference_tasks,candidate_features,candidate_ids,count=count,seed=seed)
    # Keep the inexpensive vectorized scalar implementation for ordinary KV.
    from .kernel_variance import KernelVariance
    if isinstance(estimator,KernelVariance) and np.ndim(candidate_features)==2:
        return _scalar_ivr(estimator,reference_features,reference_tasks,candidate_features,candidate_ids,count=count,seed=seed)
    if hasattr(estimator,'model') and isinstance(estimator.model,KernelVariance) and np.ndim(candidate_features)==2:
        return _scalar_ivr(estimator.model,reference_features,reference_tasks,candidate_features,candidate_ids,count=count,seed=seed)
    ids=list(candidate_ids)
    if len(ids)!=len(set(ids)) or not 0<count<=len(ids):raise ValueError('IVR requires enough distinct candidate IDs')
    candidates=np.asarray(candidate_features);reference=np.asarray(reference_features)
    if candidates.ndim==2:candidates=candidates[:,None,:]
    if reference.ndim==2:reference=reference[:,None,:]
    if candidates.ndim!=3 or reference.ndim!=3:raise ValueError('IVR expects episode/probe/features')
    order=np.argsort(ids);ids=[ids[i] for i in order];candidates=candidates[order]
    weights=(task_weights(reference_tasks)[:,None]*probe_weights(reference.shape[1],estimator.features)).reshape(-1)
    reference=reference.reshape(-1,reference.shape[-1])
    components=estimator.covariance_components();rng=np.random.default_rng(seed)
    initial=float(np.dot(task_weights(reference_tasks),estimator.score(reference_features)))
    if hasattr(estimator,'prepare_ivr'):
        reference,candidates,components=estimator.prepare_ivr(reference,candidates)
    # BEGIN RUN PROGRESS
    from run_progress import Progress
    progress=Progress('IVR greedy candidate comparisons',sum(len(ids)-i for i in range(count)),unit='candidates',every=max(1,len(ids)//10))
    # END RUN PROGRESS
    chosen=[];records=[];selected=[];integrated=initial
    for _ in range(count):
        gains=np.zeros(len(ids));states=[]
        sx=np.concatenate([candidates[i] for i in selected]) if selected else None
        for covariance,noise,scale in components:
            noise=np.atleast_1d(noise);outputs=len(noise);factor=None
            if sx is not None:
                ss=covariance(sx,sx);ss=(ss+ss.T)/2
                ss+=torch.diag(torch.as_tensor(np.tile(noise,len(sx)),device=ss.device,dtype=ss.dtype))
                factor=torch.linalg.cholesky(ss)
            states.append((covariance,noise,scale,factor))
        for i,q in enumerate(candidates):
            if i in selected:continue
            for covariance,noise,scale,factor in states:
                qq=covariance(q,q);qs=covariance(q,sx) if sx is not None else None
                if factor is not None:qq-=qs@torch.cholesky_solve(qs.T,factor)
                qq=(qq+qq.T)/2
                qq+=torch.diag(torch.as_tensor(np.tile(noise,len(q)),device=qq.device,dtype=qq.dtype))
                qfactor=torch.linalg.cholesky(qq)
                outputs=len(noise)
                for start in range(0,len(reference),256):
                    r=reference[start:start+256];rq=covariance(r,q)
                    if factor is not None:rq-=covariance(r,sx)@torch.cholesky_solve(qs.T,factor)
                    reduction=(rq*torch.cholesky_solve(rq.T,qfactor).T).sum(1)
                    w=torch.as_tensor(np.repeat(weights[start:start+len(r)],outputs),device=rq.device,dtype=rq.dtype)
                    gains[i]+=float(w@reduction)*scale
            # BEGIN RUN PROGRESS
            progress.update(detail=f'batch slot={len(selected)+1}/{count}; candidate={ids[i]}; gain={gains[i]:.6g}')
            # END RUN PROGRESS
        available=np.ones(len(ids),dtype=bool);available[selected]=False
        if not np.isfinite(gains).all() or gains[available].min() < -1e-8:raise RuntimeError('Invalid IVR gain')
        maximum=gains[available].max();ties=np.flatnonzero(available&np.isclose(gains,maximum,rtol=1e-8,atol=1e-12))
        index=int(rng.choice(ties));gain=float(gains[index]);selected.append(index);chosen.append(ids[index])
        records.append(dict(id=ids[index],gain=gain,tied_ids=[ids[i] for i in ties],
            integrated_variance_before=integrated,integrated_variance_after=integrated-gain))
        integrated-=gain
    # BEGIN RUN PROGRESS
    progress.finish(f'selected={chosen}; integrated variance {initial:.6g} -> {integrated:.6g}')
    # END RUN PROGRESS
    return chosen,dict(rule='KV-T-IVR' if getattr(estimator,'target',False) else 'KV-IVR',seed=seed,
        initial_integrated_variance=initial,final_integrated_variance=integrated,steps=records,
        conditioning_updated=False,reference_task_count=len(set(reference_tasks)),probes_per_acquired_demo=candidates.shape[1],
        output_covariance_components=len(components),noise_rule='per-output fitted noise, or KV ridge')
