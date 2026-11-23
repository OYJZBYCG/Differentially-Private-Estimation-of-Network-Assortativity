#!/usr/bin/env python3
"""Paired stabilization ablation and BA scaling/sparsity experiments."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import assortativity as core


def ba_graph(n, m, seed):
    """Preferential attachment, initial star on m+1 nodes, no loops/multiedges."""
    if not 1 <= m < n:
        raise ValueError('BA requires 1 <= m < n')
    rng = np.random.default_rng(seed)
    edges = [(0, i) for i in range(1, m + 1)]
    urn = [v for e in edges for v in e]
    for node in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            targets.add(urn[int(rng.integers(len(urn)))])
        for target in sorted(targets):
            edges.append((node, target))
            urn.extend((node, target))
    u, v = np.asarray(edges, dtype=np.int64).T
    a = sp.coo_matrix((np.ones(len(u), dtype=np.int8), (u, v)), shape=(n,n)).tocsr()
    return core.GraphData.from_adjacency(a.maximum(a.T))

def summarize(raw, group_keys):
    records = []
    for key, frame in raw.groupby(group_keys, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        result = dict(zip(group_keys, key))
        e = frame.error_r.to_numpy(float)
        finite = np.isfinite(e)
        # Numerical summaries are explicitly conditional on finite outputs.
        ef = e[finite]
        result.update(trials=len(frame), finite_trials=int(finite.sum()),
            nonfinite_rate=float((~finite).mean()),
            mse_r=float(np.mean(ef**2)) if len(ef) else np.nan,
            mae_r=float(np.mean(abs(ef))) if len(ef) else np.nan,
            bias_r=float(np.mean(ef)) if len(ef) else np.nan,
            abs_error_p95=float(np.quantile(abs(ef), .95)) if len(ef) else np.nan,
            r_true=float(frame.r_true.iloc[0]))
        for col in ('q_nonpositive','floor_trigger','outside_range','undefined',
                    'degree_release_seconds','acquisition_seconds','postprocess_seconds',
                    'simulation_total_seconds','degree_report_count',
                    'public_feedback_scalar_count','second_stage_scalar_count',
                    'second_stage_bit_count','share_count','share_payload_bits'):
            if col in frame: result[col] = float(frame[col].mean())
        records.append(result)
    return pd.DataFrame(records)

def main(suite="scale"):
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(suite=suite)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--eps', default='0.5,1')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--graph-seeds', default='2026,2027,2028')
    p.add_argument('--sizes', default='1000,5000,10000,20000,40000,60000,80000,100000')
    p.add_argument('--scale-m', type=int, default=10)
    p.add_argument('--sparse-n', type=int, default=10000)
    p.add_argument('--attachments', default='2,5,10,20,40,60,80,100')
    p.add_argument('--delta', type=float, default=1e-8)
    p.add_argument('--degree-share', type=float, default=.5)
    p.add_argument('--precision', type=int, default=65536)
    p.add_argument('--wrap-probability', type=float, default=1e-18)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--beta', type=float, default=.05)
    p.add_argument('--max-local-pairs', type=int, default=5_000_000_000 if suite == 'scale' else 800_000_000)
    p.add_argument('--skip-explicit-permutations', action='store_true')
    args = p.parse_args()
    eps = [float(v) for v in args.eps.split(',')]
    seeds = [int(v) for v in args.graph_seeds.split(',')]
    if (args.trials < 1 or not seeds or not eps or not all(np.isfinite(e) and e>0 for e in eps)
        or not 0<args.degree_share<1 or not 0<args.delta<1 or not 0<args.beta<1
        or args.precision<1
        or not 0<args.wrap_probability<1): p.error('Invalid parameters')
    args.no_local=False; args.no_decentral=False
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['driver_version']='ba_final_estimator'
    config['core_sha256']=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
    settings = args.output/'settings.json'
    if settings.exists() and json.loads(settings.read_text()) != config:
        raise ValueError('Different settings: use a new output directory')
    settings.write_text(json.dumps(config,indent=2))
    if suite == "scale":
        settings = [(n, args.scale_m) for n in map(int, args.sizes.split(','))]
    else:
        settings = [(args.sparse_n, m) for m in map(int, args.attachments.split(','))]
    jobs = [(n, m, seed) for n, m in settings for seed in seeds]
    pieces=[]; graph_info=[]
    name = "BA"
    for idx, (n, m, seed) in enumerate(jobs):
        t=time.perf_counter()
        graph=ba_graph(n,m,seed)
        build_seconds=time.perf_counter()-t
        n=len(graph.degrees)
        pairs=n*(n-1)//2
        if pairs > args.max_local_pairs:
            raise ValueError(f'n={n} requires {pairs:,} Local candidate pairs per trial; '
                             f'limit={args.max_local_pairs:,}. Explicitly increase --max-local-pairs '
                             'or use smaller --sizes. No scheme is silently skipped.')
        if n<19 or not np.isfinite(graph.r): raise ValueError('Require n>=19 and defined r')
        digest=hashlib.sha256(graph.a.indptr.tobytes()+graph.a.indices.tobytes()).hexdigest()
        info=dict(job=idx,suite=suite,dataset=name,n=n,ba_m=m,graph_seed=seed,
                  edges=graph.upper.nnz,mean_degree=float(graph.degrees.mean()),
                  max_degree=float(graph.degrees.max()),r_true=graph.r,q_true=graph.q,
                  graph_sha256=digest,graph_build_seconds=build_seconds)
        graph_info.append(info)
        folder=args.output/f'job_{idx:03d}';folder.mkdir(exist_ok=True)
        identity=folder/'graph.json'
        if identity.exists() and json.loads(identity.read_text())['graph_sha256']!=digest:
            raise ValueError('Graph changed; use a new output directory')
        identity.write_text(json.dumps(info,indent=2))
        print(f'{idx+1}/{len(jobs)} {suite}: {name}, n={n}, m={m}, seed={seed}',flush=True)
        for ei,e in enumerate(eps):
            for trial in range(args.trials):
                path=folder/f'eps_{ei}_trial_{trial}.csv'
                if path.exists(): frame=pd.read_csv(path)
                else:
                    rows=core.paired_trial(graph,e,ei,trial,idx,args)
                    for row in rows: row['variant']='corrected'
                    frame=pd.DataFrame(rows)
                    for k in ('job','suite','dataset','n','ba_m','graph_seed'): frame[k]=info[k]
                    temp=path.with_suffix('.tmp');frame.to_csv(temp,index=False);temp.replace(path)
                expected=3
                if len(frame)!=expected or set(frame.mechanism)!={'local','ikos_shuffle','decentral'}:
                    raise ValueError(f'Invalid checkpoint {path}')
                pieces.append(frame)
    raw=pd.concat(pieces,ignore_index=True)
    raw.to_csv(args.output/'trials.csv',index=False)
    pd.DataFrame(graph_info).to_csv(args.output/'graphs.csv',index=False)
    keys=['suite','dataset','n','ba_m','graph_seed','epsilon','mechanism','variant']
    summary=summarize(raw,keys);summary.to_csv(args.output/'summary_by_graph.csv',index=False)
    # Graph instances have equal weight; SE below measures between-graph variation.
    keys.remove('graph_seed')
    metrics=['mse_r','mae_r','bias_r','abs_error_p95','nonfinite_rate',
        'q_nonpositive','floor_trigger','outside_range','simulation_total_seconds',
        'degree_release_seconds','acquisition_seconds','postprocess_seconds',
        'degree_report_count','public_feedback_scalar_count','second_stage_scalar_count',
        'second_stage_bit_count','share_count','share_payload_bits']
    metrics=[c for c in metrics if c in summary]
    grouped=summary.groupby(keys,dropna=False)[metrics]
    mean=grouped.mean();se=grouped.sem().add_suffix('_graph_se')
    mean.join(se).reset_index().to_csv(args.output/'summary.csv',index=False)
    print('Saved:',args.output.resolve())


def load_trial_metrics(folder):
    trials = pd.read_csv(folder / "trials.csv")
    if not np.isfinite(trials[["p_hat", "q_hat", "r_true", "r_hat"]]).all().all():
        raise ValueError(f"Nonfinite input in {folder}")
    if (trials.r_true == 0).any():
        raise ValueError(f"Undefined MARE in {folder}")
    if "mean_degree" not in trials:
        graph_columns = ["job", "mean_degree"]
        trials = trials.merge(
            pd.read_csv(folder / "graphs.csv")[graph_columns],
            on="job", how="left", validate="many_to_one",
        )
    if trials.mean_degree.isna().any():
        raise ValueError(f"Missing graph metadata in {folder}")
    if not trials.variant.eq("corrected").all():
        raise ValueError(f"Run the current experiment code to generate final estimates in {folder}")
    error = trials.r_hat - trials.r_true
    trials["MSE"] = error**2
    trials["MARE"] = error.abs() / trials.r_true.abs()
    return trials


def summarize_trial_metrics(trials, keys):
    if trials.duplicated(keys + ["graph_seed", "trial"]).any():
        raise ValueError("Duplicate trial within a graph")
    by_graph = trials.groupby(keys + ["graph_seed"], dropna=False).agg(
        MSE=("MSE", "mean"), MARE=("MARE", "mean"),
        mean_degree=("mean_degree", "mean"), runs=("trial", "size"),
    )
    return by_graph.groupby(level=keys).agg(
        MSE=("MSE", "mean"), MARE=("MARE", "mean"),
        mean_degree=("mean_degree", "mean"), runs=("runs", "sum"),
    ).reset_index()
