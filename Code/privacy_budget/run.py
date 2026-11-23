"""Paired polynomial ablation and privacy allocation experiments; Python >=3.10."""
import os
# One BLAS thread per worker prevents nested CPU/RAM oversubscription.
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import multiprocessing as mp
import queue
import time
import traceback
from .memory_guard import snapshot
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import assortativity as core


def summarize(raw):
    keys = ['suite','dataset','epsilon','degree_share','mechanism','variant']
    records=[]
    for key, f in raw.groupby(keys, sort=False):
        row=dict(zip(keys,key)); n=len(f)
        row['trials']=n
        row['r_true']=f.r_true.iloc[0]
        for metric,col in [('mse','squared_error'),('mae','absolute_error'),('mare','relative_error'),
                           ('bias_r','error_r'),('relative_bias_p','relative_error_p'),
                           ('relative_bias_q','relative_error_q'),('zero_rate','zero_fallback'),
                           ('projection_rate','projection')]:
            x=f[col].to_numpy(float)
            row[metric]=float(x.mean()) if np.isfinite(x).all() else np.nan
            row[metric+'_se']=float(x.std(ddof=1)/np.sqrt(n)) if n>1 and np.isfinite(x).all() else np.nan
        records.append(row)
    return pd.DataFrame(records)



def worker_loop(g, args, di, name, jobs, results):
    while True:
        task=jobs.get()
        if task is None:
            return
        suite,ei,eps,ri,rho,t,path_string=task
        path=Path(path_string)
        args.degree_share=rho
        rec={'name':name}
        try:
            # Allocation settings share random seeds (common random numbers).
            rows=core.paired_trial(g,eps,ei,t,di,args)
            out=[]
            for row in rows:
                for variant in (['plugin','corrected'] if suite=='ablation' else ['corrected']):
                    ph,qh=(row['plug_p'],row['plug_q']) if variant=='plugin' else (row['p_hat'],row['q_hat'])
                    if not np.isfinite(ph) or not np.isfinite(qh):
                        raise ArithmeticError('Nonfinite reconstructed statistic; inspect numerical scale')
                    rh=float(core.final_estimate(ph,qh)) if variant=='plugin' else row['r_hat']
                    err=rh-g.r
                    out.append(dict(row,suite=suite,dataset=rec['name'],degree_share=rho,variant=variant,
                        r_hat=rh,p_hat_variant=ph,q_hat_variant=qh,error_r=err,
                        squared_error=err**2,absolute_error=abs(err),
                        relative_error=abs(err)/abs(g.r) if g.r!=0 else np.nan,
                        relative_error_p=(ph-g.p)/g.q,relative_error_q=(qh-g.q)/g.q,
                        zero_fallback=int(qh<=0),projection=int(qh>0 and abs(ph/qh)>1)))
            f=pd.DataFrame(out)
            tmp=path.with_suffix('.tmp');f.to_csv(tmp,index=False);tmp.replace(path)
            results.put(('ok', path_string))
        except BaseException:
            results.put(('error', traceback.format_exc()))
            return


def memory_problem(args):
    percent,available,rss=snapshot()
    gib=1024**3
    reasons=[]
    if percent>=args.max_ram_percent: reasons.append(f'system RAM {percent:.1f}% >= {args.max_ram_percent}%')
    if available/gib<args.min_available_gb: reasons.append(f'available RAM {available/gib:.2f} GiB < {args.min_available_gb} GiB')
    if args.max_job_gb>0 and rss/gib>=args.max_job_gb: reasons.append(f'process-tree RSS {rss/gib:.2f} GiB >= {args.max_job_gb} GiB')
    return '; '.join(reasons)


def run_tasks(g,args,di,name,tasks):
    if not tasks: return
    reason=memory_problem(args)
    if reason: raise RuntimeError('MEMORY STOP before workers: '+reason)
    ctx=mp.get_context('spawn')  # Same behavior on Windows and Linux.
    jobs=ctx.Queue();results=ctx.Queue()
    workers=[ctx.Process(target=worker_loop,args=(g,args,di,name,jobs,results)) for _ in range(min(args.workers,len(tasks)))]
    started=[]
    try:
        for proc in workers:
            proc.start();started.append(proc)
            reason=memory_problem(args)
            if reason: raise RuntimeError('MEMORY STOP during startup: '+reason)
        remaining=iter(tasks)
        for _ in started: jobs.put(next(remaining))
        done=0;last=time.monotonic()
        while done<len(tasks):
            reason=memory_problem(args)
            if reason: raise RuntimeError('MEMORY STOP: '+reason)
            if any(proc.exitcode is not None for proc in started):
                raise RuntimeError('Worker exited unexpectedly (possibly OS memory termination). Completed checkpoints are preserved.')
            try: status,message=results.get(timeout=args.memory_check_seconds)
            except queue.Empty:
                if time.monotonic()-last>=30:
                    print(f'  {done}/{len(tasks)} new trials finished; workers still running',flush=True);last=time.monotonic()
                continue
            if status!='ok': raise RuntimeError(message)
            done+=1
            if done%10==0 or done==len(tasks): print(f'  {done}/{len(tasks)} new trials saved',flush=True)
            task=next(remaining,None)
            if task is not None: jobs.put(task)
        for _ in started: jobs.put(None)
        for proc in started: proc.join(timeout=10)
    finally:
        # Only stop child processes belonging to this invocation.
        for proc in started:
            if proc.is_alive(): proc.terminate()
        for proc in started: proc.join(timeout=5)
        for proc in started:
            if proc.is_alive(): proc.kill();proc.join(timeout=5)
        jobs.cancel_join_thread();results.cancel_join_thread()
        jobs.close();results.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['ablation','allocation','all'],default='all')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eps',help='Default: ablation .1,.3,.5,1,2,3; allocation .5,1')
    p.add_argument('--ratios',default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    p.add_argument('--degree-share',type=float,default=.5)
    p.add_argument('--trials',type=int,default=100)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--delta',type=float,default=1e-8)
    p.add_argument('--precision',type=int,default=65536)
    p.add_argument('--wrap-probability',type=float,default=1e-18)
    p.add_argument('--max-local-pairs',type=int,default=800000000)
    p.add_argument('--skip-explicit-permutations',action='store_true',help='Accuracy simulation only; no deployed-time claims')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-ram-percent',type=float,default=85)
    p.add_argument('--min-available-gb',type=float,default=2)
    p.add_argument('--max-job-gb',type=float,default=0,help='Process-tree RSS limit in GiB; 0 disables this limit')
    p.add_argument('--memory-check-seconds',type=float,default=.5)
    args=p.parse_args()
    if args.workers<1 or not 0<args.max_ram_percent<=100 or args.min_available_gb<0 or args.max_job_gb<0 or not .05<=args.memory_check_seconds<=10:
        p.error('Invalid worker or memory settings')
    def numbers(s):
        v=[float(x) for x in s.split(',')]
        if not v or len(set(v))!=len(v) or not all(np.isfinite(v)): p.error('Invalid or duplicate numeric list')
        return v
    ratios=numbers(args.ratios)
    if not all(0<x<1 for x in ratios) or not 0<args.degree_share<1: p.error('Ratios must be in (0,1)')
    if args.trials<1 or not 0<args.delta<.5 or not 0<args.wrap_probability<1 or args.precision<1 or args.max_local_pairs<1 or args.seed<0: p.error('Invalid settings')
    suites=['ablation','allocation'] if args.suite=='all' else [args.suite]
    eps_by_suite={s:numbers(args.eps or ('0.1,0.3,0.5,1,2,3' if s=='ablation' else '0.5,1')) for s in suites}
    if not all(e>0 for v in eps_by_suite.values() for e in v): p.error('epsilon must be positive')
    table=pd.read_csv(args.manifest,dtype=str).fillna('')
    if not {'name','nodes','edge_file'}<=set(table) or table.empty or table.name.duplicated().any(): p.error('Manifest requires unique name,nodes,edge_file columns')
    args.output.mkdir(parents=True,exist_ok=True)
    cfg={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    for runtime_key in ['workers','max_ram_percent','min_available_gb','max_job_gb','memory_check_seconds']:
        cfg.pop(runtime_key)
    cfg.update(version='supplement_parallel_v4' ,manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
               driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    settings=args.output/'settings.json'
    if settings.exists():
        previous=json.loads(settings.read_text())
        compatible=previous==cfg
        if previous.get('version')=='supplement_v3':
            # Same stochastic core and experiment settings: serial-v3 checkpoints are reusable.
            ignore={'version','driver_sha256'}
            compatible={k:v for k,v in previous.items() if k not in ignore}=={k:v for k,v in cfg.items() if k not in ignore}
            if compatible: print('Reusing compatible serial-v3 checkpoints.',flush=True)
        if not compatible:
            raise ValueError('Experiment settings or code changed: use a NEW output directory. Worker/memory options may change freely.')
    settings.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    args.beta=.05;args.no_local=False;args.no_decentral=False
    pieces=[];graphs=[];ablation_share=args.degree_share
    for di,rec in enumerate(table.to_dict('records')):
        g=core.load_record(rec,args.manifest.resolve().parent);n=len(g.degrees)
        if n<19 or not np.isfinite(g.r): raise ValueError(f'{rec["name"]}: require n>=19 and Q>0')
        if n*(n-1)//2>args.max_local_pairs: raise ValueError('Local pair limit exceeded; raise --max-local-pairs explicitly')
        digest=hashlib.sha256(g.a.indptr.tobytes()+g.a.indices.tobytes()).hexdigest()
        info=dict(dataset=rec['name'],n=n,edges=g.upper.nnz,max_degree=float(g.degrees.max()),
                  r_true=g.r,q_true=g.q,graph_sha256=digest)
        identity=args.output/f'graph_{di}.json'
        if identity.exists() and json.loads(identity.read_text())!=info: raise ValueError('Graph changed; use a new output directory')
        identity.write_text(json.dumps(info,indent=2));graphs.append(info)
        tasks=[];checkpoints=[]
        for suite in suites:
            for ei,eps in enumerate(eps_by_suite[suite]):
                for ri,rho in enumerate([ablation_share] if suite=='ablation' else ratios):
                    for t in range(args.trials):
                        path=args.output/f'{suite}_d{di}_e{ei}_a{ri}_t{t}.csv'
                        checkpoints.append((path,suite,eps,rho,t))
                        if not path.exists(): tasks.append((suite,ei,eps,ri,rho,t,str(path.resolve())))
        print(f'{rec["name"]}: {len(checkpoints)-len(tasks)} cached, {len(tasks)} pending; workers={args.workers}',flush=True)
        try:
            run_tasks(g,args,di,rec['name'],tasks)
        except (KeyboardInterrupt,RuntimeError,MemoryError):
            print('STOPPED. Completed per-trial CSV files are preserved. Reduce --workers and resume the same command/output.',flush=True)
            raise
        for path,suite,eps,rho,t in checkpoints:
            f=pd.read_csv(path)
            expected=6 if suite=='ablation' else 3
            if (len(f)!=expected or set(f.mechanism)!={'local','ikos_shuffle','decentral'}
                or not np.allclose(f.epsilon,eps) or not np.allclose(f.degree_share,rho)
                or not (f.trial==t).all() or not (f.dataset==rec['name']).all()):
                raise ValueError(f'Invalid checkpoint: {path}')
            pieces.append(f)
        reason=memory_problem(args)
        if reason: raise RuntimeError('MEMORY STOP before summary: '+reason)
    raw=pd.concat(pieces,ignore_index=True);raw.to_csv(args.output/'trials.csv',index=False)
    summarize(raw).to_csv(args.output/'summary.csv',index=False)
    pd.DataFrame(graphs).to_csv(args.output/'graphs.csv',index=False)
    paired=[]
    a=raw[raw.suite=='ablation']
    for key,f in a.groupby(['dataset','epsilon','mechanism']):
        for metric in ['squared_error','relative_error']:
            w=f.pivot(index='trial',columns='variant',values=metric)
            delta=w.corrected-w.plugin
            paired.append(dict(dataset=key[0],epsilon=key[1],mechanism=key[2],metric=metric,
                              corrected_minus_plugin=float(delta.mean()),
                              paired_se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan))
    if paired: pd.DataFrame(paired).to_csv(args.output/'paired_comparison.csv',index=False)
    print('Saved to',args.output.resolve(),flush=True)

if __name__=='__main__':
    mp.freeze_support()
    main()
