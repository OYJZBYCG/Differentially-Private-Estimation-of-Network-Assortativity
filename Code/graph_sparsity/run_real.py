#!/usr/bin/env python3
"""Edge thinning on real graphs; use the unchanged extras_v2 algorithm core."""
import os
# Avoid multiplying BLAS threads inside independent worker processes.
for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import assortativity as core

ROOT = Path(__file__).resolve().parent
MECHANISMS = {'local', 'ikos_shuffle', 'decentral'}


def graph_hash(graph):
    a = graph.a.copy(); a.sort_indices()
    return hashlib.sha256(np.asarray(a.shape, dtype='<i8').tobytes() +
        a.indptr.astype('<i8').tobytes() + a.indices.astype('<i8').tobytes()).hexdigest()


def stable_id(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:4], 'little')


def thin_graph(base, probability, seed, dataset):
    """One uniform per original edge; reuse across q to obtain nested subgraphs."""
    if probability == 1.0:
        return base
    rng = np.random.default_rng(np.random.SeedSequence([seed, stable_id(dataset), 17491]))
    keep = rng.random(len(base.source)) < probability
    u, v = base.source[keep], base.target[keep]
    a = sp.coo_matrix((np.ones(len(u), dtype=np.int8), (u, v)), shape=base.a.shape).tocsr()
    return core.GraphData.from_adjacency(a.maximum(a.T))


def atomic_csv(frame, path):
    temp = path.with_suffix('.tmp'); frame.to_csv(temp, index=False); temp.replace(path)


def summarize(raw):
    records = []
    keys = ['dataset', 'retain_probability', 'graph_seed', 'epsilon', 'mechanism']
    for identity, f in raw.groupby(keys, dropna=False):
        row = dict(zip(keys, identity))
        e = f.error_r.to_numpy(float); finite = np.isfinite(e)
        re = f.relative_error_r.to_numpy(float); relative_finite = np.isfinite(re)
        ef = e[finite]
        row.update(n=int(f.n.iloc[0]), mean_degree=float(f.mean_degree.iloc[0]),
            r_true=float(f.r_true.iloc[0]), trials=len(f), finite_trials=int(finite.sum()),
            nonfinite_rate=float((~finite).mean()), relative_finite_trials=int(relative_finite.sum()),
            mse_r=float(np.mean(ef**2)) if len(ef) else np.nan,
            mae_r=float(np.mean(abs(ef))) if len(ef) else np.nan,
            mare_r=float(re[relative_finite].mean()) if relative_finite.any() else np.nan,
            bias_r=float(ef.mean()) if len(ef) else np.nan)
        for col in ['clipped', 'zero_fallback', 'wrapped', 'simulation_total_seconds',
            'degree_release_seconds', 'acquisition_seconds', 'postprocess_seconds',
            'degree_report_count', 'public_feedback_scalar_count', 'second_stage_scalar_count',
            'second_stage_bit_count', 'share_count', 'share_payload_bits']:
            row[col] = float(f[col].mean())
        records.append(row)
    by_graph = pd.DataFrame(records)
    group = by_graph.groupby(['dataset', 'retain_probability', 'epsilon', 'mechanism'], dropna=False)
    metrics = ['mean_degree', 'mse_r', 'mae_r', 'mare_r', 'bias_r', 'nonfinite_rate',
        'clipped', 'zero_fallback', 'wrapped', 'simulation_total_seconds', 'degree_release_seconds',
        'acquisition_seconds', 'postprocess_seconds', 'degree_report_count',
        'public_feedback_scalar_count', 'second_stage_scalar_count', 'second_stage_bit_count',
        'share_count', 'share_payload_bits']
    summary = group[metrics].mean().join(group[metrics].sem().add_suffix('_graph_se'))
    summary = summary.join(group.size().rename('graph_instances'))
    summary = summary.join(group.mare_r.count().rename('mare_graph_instances')).reset_index()
    return by_graph, summary


_WORKER_GRAPH = None
_WORKER_ARGS = None

def initialize_worker(graph, args):
    global _WORKER_GRAPH, _WORKER_ARGS
    _WORKER_GRAPH, _WORKER_ARGS = graph, args

def run_worker(task):
    e, ei, trial, random_id = task
    return core.paired_trial(_WORKER_GRAPH, e, ei, trial, random_id, _WORKER_ARGS)

def completed_frames(graph, eps, random_id, args, folder, info):
    """Bounded prefetch; retain original trial order and deterministic RNG keys."""
    tasks = [(ei, e, trial) for ei,e in enumerate(eps) for trial in range(args.trials)]
    missing = [(ei,e,t) for ei,e,t in tasks if not (folder/f'eps_{ei}_trial_{t}.csv').exists()]
    pending = {}; next_task = 0; fresh = 0
    count = min(args.workers, len(missing))
    pool = (ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context('spawn'),
            initializer=initialize_worker, initargs=(graph,args)) if count else None)
    try:
        def fill():
            nonlocal next_task
            while pool and next_task < len(missing) and len(pending) < 2*count:
                ei,e,t = missing[next_task]; next_task += 1
                pending[(ei,t)] = pool.submit(run_worker,(e,ei,t,random_id))
        fill()
        for ei,e,trial in tasks:
            path=folder/f'eps_{ei}_trial_{trial}.csv'
            if path.exists(): frame=pd.read_csv(path)
            else:
                rows=pending.pop((ei,trial)).result()
                for row in rows:
                    row.update(variant='corrected',zero_fallback=int(row['q_hat']<=0))
                    for k in ['job','suite','dataset','n','retain_probability','graph_seed','mean_degree']:
                        row[k]=info[k]
                frame=pd.DataFrame(rows);frame['execution_workers']=count;atomic_csv(frame,path);fresh+=1;fill()
                if fresh==1 or fresh%10==0:
                    print(f'  saved {fresh}/{len(missing)} new paired trials (workers={count}); epsilon={e:g}, trial={trial+1}',flush=True)
            if (len(frame)!=3 or set(frame.mechanism)!=MECHANISMS
                or not np.allclose(frame.epsilon,e) or not (frame.trial==trial).all()
                or not np.allclose(frame.r_true,graph.r,rtol=1e-10,atol=1e-14)):
                raise ValueError(f'Invalid checkpoint: {path}')
            yield frame
        print(f'  graph complete: {len(tasks)-len(missing)} reused, {fresh} new paired trials',flush=True)
    finally:
        if pool:
            # On interruption, keep disk checkpoints and terminate unfinished workers.
            import sys
            if sys.exc_info()[0] is not None:
                for process in list((getattr(pool, '_processes', None) or {}).values()):
                    process.terminate()
                pool.shutdown(wait=True, cancel_futures=True)
            else:
                pool.shutdown(wait=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=min(4, max(1, (os.cpu_count() or 2)//2)))
    p.add_argument('--manifest', type=Path, default=ROOT/'datasets.csv')
    p.add_argument('--datasets', default='Deezer,GitHub', help='Case-insensitive manifest names; ALL selects every row')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--keep', default='0.2,0.4,0.6,0.8,1')
    p.add_argument('--eps', default='0.5,1')
    p.add_argument('--graph-seeds', default='2026,2027,2028')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--delta', type=float, default=1e-8)
    p.add_argument('--degree-share', type=float, default=.5)
    p.add_argument('--precision', type=int, default=65536)
    p.add_argument('--wrap-probability', type=float, default=1e-18)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--beta', type=float, default=.05)
    p.add_argument('--max-local-pairs', type=int, default=800_000_000)
    args = p.parse_args()
    if args.workers < 1: p.error('--workers must be at least 1')
    qs = sorted(set(float(x) for x in args.keep.split(',')))
    eps = list(dict.fromkeys(float(x) for x in args.eps.split(',')))
    seeds = list(dict.fromkeys(int(x) for x in args.graph_seeds.split(',')))
    if not qs or not all(np.isfinite(q) and 0 < q <= 1 for q in qs): p.error('Require 0 < keep <= 1')
    if not eps or not all(np.isfinite(e) and e > 0 for e in eps): p.error('Require epsilon > 0')
    if not seeds or min(seeds) < 0 or args.seed < 0 or args.trials < 1: p.error('Invalid seeds or trial count')
    if not (0 < args.delta < 1 and 0 < args.degree_share < 1 and 0 < args.beta < 1
            and 0 < args.wrap_probability < 1 and args.precision >= 1 and args.max_local_pairs > 0):
        p.error('Invalid privacy or computation parameters')
    args.no_local = False; args.no_decentral = False
    args.skip_explicit_permutations = False
    manifest = args.manifest.resolve()
    table = pd.read_csv(manifest, dtype=str).fillna('')
    if not {'name', 'nodes', 'edge_file'} <= set(table): p.error('Manifest requires name,nodes,edge_file')
    if table.name.str.casefold().duplicated().any(): p.error('Dataset names must be unique')
    if args.datasets.upper() != 'ALL':
        wanted = {s.strip().casefold() for s in args.datasets.split(',')}
        missing = wanted - set(table.name.str.casefold())
        if missing: p.error('Names absent from manifest: '+','.join(sorted(missing)))
        table = table[table.name.str.casefold().isin(wanted)]
    if table.empty: p.error('No datasets selected')
    if 'ba_m' in table and table.ba_m.astype(bool).any(): p.error('Use real edge files; BA rows are not accepted')
    records = table.to_dict('records')
    # Fail early on all paths before expensive experiments.
    for row in records:
        source = manifest.parent / row['edge_file']
        if not source.is_file(): raise FileNotFoundError(f'{row["name"]}: edit edge_file in {manifest}; missing {source}')
        n = int(row['nodes'])
        if n < 19 or n*(n-1)//2 > args.max_local_pairs:
            raise ValueError(f'{row["name"]}: n={n}, Local requires {n*(n-1)//2:,} pairs; limit={args.max_local_pairs:,}. No algorithm is skipped.')
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items() if k != 'workers'}
    config.update(driver_version='real_sparsity_final_estimator', core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets_selected=records,
        thinning='nested Bernoulli edge masks within each seed; fixed vertex set; q=1 run once',
        keep_values=qs, epsilon_values=eps)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = args.output/'settings.json'
    if settings.exists() and json.loads(settings.read_text(encoding='utf8')) != config:
        raise ValueError('Existing output uses different settings or code. Choose a new --output directory.')
    settings.write_text(json.dumps(config,indent=2), encoding='utf8')
    execution = dict(workers=args.workers, scheduler='ProcessPoolExecutor spawn',
        parallel_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing_note='Legacy checkpoints retained; parallel-run timings reflect CPU contention, not serial speed.')
    (args.output/'parallel_execution.json').write_text(json.dumps(execution,indent=2),encoding='utf8')
    print(f'Parallel workers: {args.workers}; existing checkpoints will be reused.',flush=True)
    pieces=[]; graph_rows=[]; base_rows=[]; job=0
    for record in records:
        name=record['name']; base=core.load_record(record,manifest.parent); base_digest=graph_hash(base)
        base_rows.append(dict(dataset=name,n=len(base.degrees),edges=base.upper.nnz,r_true=base.r,q_true=base.q,graph_sha256=base_digest))
        for q in qs:
            # q=1 is the same graph regardless of graph seed; do not inflate sample size.
            for graph_seed in ([-1] if q == 1 else seeds):
                start=time.perf_counter(); graph=thin_graph(base,q,graph_seed,name)
                if not np.isfinite(graph.r):
                    raise ValueError(f'{name}, q={q}, seed={graph_seed}: Q=0, r undefined. No resampling is performed.')
                n=len(graph.degrees);digest=graph_hash(graph)
                info=dict(job=job,suite='real_sparsity',dataset=name,n=n,retain_probability=q,
                    graph_seed=graph_seed,edges=int(graph.upper.nnz),base_edges=int(base.upper.nnz),
                    realized_edge_fraction=graph.upper.nnz/base.upper.nnz,
                    mean_degree=float(graph.degrees.mean()),max_degree=float(graph.degrees.max()),
                    isolated_vertices=int((graph.degrees==0).sum()),r_true=graph.r,q_true=graph.q,
                    graph_sha256=digest,base_graph_sha256=base_digest,graph_build_seconds=time.perf_counter()-start)
                folder=args.output/f'job_{job:03d}';folder.mkdir(exist_ok=True)
                identity=folder/'graph.json'
                if identity.exists():
                    previous=json.loads(identity.read_text(encoding='utf8'))
                    if previous['graph_sha256'] != digest or previous['base_graph_sha256'] != base_digest:
                        raise ValueError('Input graph changed: use a new output directory')
                identity.write_text(json.dumps(info,indent=2),encoding='utf8');graph_rows.append(info)
                # Save the exact fixed-size sparse graph so the result is reproducible.
                sp.save_npz(folder/'graph.npz',graph.a)
                random_id=stable_id(name,format(q,'.17g'),graph_seed)
                print(f'[{job+1}] {name}: q={q:g}, seed={graph_seed}, n={n:,}, edges={graph.upper.nnz:,}, r={graph.r:.8g}',flush=True)
                for frame in completed_frames(graph,eps,random_id,args,folder,info):
                    frame['execution_workers'] = (frame['execution_workers'] if 'execution_workers' in frame else 1)
                    pieces.append(frame)
                # Rebuild aggregate files after each completed graph; checkpoints are finer grained.
                raw=pd.concat(pieces,ignore_index=True);by_graph,summary=summarize(raw)
                atomic_csv(raw,args.output/'trials.csv');atomic_csv(by_graph,args.output/'summary_by_graph.csv')
                atomic_csv(summary,args.output/'summary.csv');atomic_csv(pd.DataFrame(graph_rows),args.output/'graphs.csv')
                atomic_csv(pd.DataFrame(base_rows),args.output/'base_graphs.csv');job+=1
    print('Saved:',args.output.resolve(),flush=True)

if __name__=='__main__':
    mp.freeze_support()
    main()
