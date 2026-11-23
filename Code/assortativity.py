"""Paper mechanisms: Local (1), Shuffle (1), and Local (2)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

def ReadEdges(edge_file: str, size: int) -> sp.csr_matrix:
    """Preserve the existing two-header-line node1/node2 format."""
    df = pd.read_csv(edge_file, skiprows=2, usecols=['node1', 'node2'])
    u = df['node1'].to_numpy(dtype=np.int64)
    v = df['node2'].to_numpy(dtype=np.int64)
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= size) or np.any(v >= size):
        raise ValueError('Node ID outside [0, size); remap noncontiguous IDs first')
    keep = u != v
    a = sp.coo_matrix((np.ones(keep.sum(), dtype=np.int8), (u[keep], v[keep])),
                      shape=(size, size)).tocsr()
    a = a.maximum(a.T).tocsr()
    a.data[:] = 1
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def validate_graph(a: sp.spmatrix) -> sp.csr_matrix:
    a = a.tocsr().astype(np.int8)
    if a.shape[0] != a.shape[1] or a.diagonal().any() or (a != a.T).nnz:
        raise ValueError('Expected an undirected, simple, symmetric adjacency matrix')
    if a.nnz and not np.all(a.data == 1):
        raise ValueError('Expected binary adjacency entries')
    return a


@dataclass
class GraphData:
    a: sp.csr_matrix
    upper: sp.csr_matrix
    degrees: np.ndarray
    source: np.ndarray
    target: np.ndarray
    s1: float
    s2: float
    s3: float
    c: float
    p: float
    q: float
    r: float

    @classmethod
    def from_adjacency(cls, adjacency: sp.spmatrix) -> 'GraphData':
        a = validate_graph(adjacency)
        n = a.shape[0]
        degrees = np.asarray(a.sum(axis=1)).ravel().astype(float)
        upper = sp.triu(a, k=1, format='csr')
        source, target = upper.nonzero()
        s1 = float(degrees.sum())
        s2 = float(np.dot(degrees, degrees))
        s3 = float(np.sum(degrees ** 3))
        c = float(np.dot(degrees[source], degrees[target]))
        p = 2 * s1 * c - s2 ** 2
        q = s1 * s3 - s2 ** 2
        # No fabricated ground truth for a regular graph or an edgeless graph.
        if q < -1e-7 * max(1.0, s1 * s3):
            raise ArithmeticError('Negative Pearson denominator; inspect graph/numerics')
        return cls(a, upper, degrees, source, target, s1, s2, s3, c,
                   p, max(0., q), float(p / q) if q > 0 else float('nan'))


def degree_moments(z: np.ndarray, b: float) -> tuple[float, float, float, float, float]:
    n = len(z)
    r1, r2, r3 = float(z.sum()), float(np.dot(z, z)), float(np.sum(z ** 3))
    s2hat = r2 - 2 * n * b * b
    lhat = r2 * r2 - 4 * (n + 2) * b * b * r2 + 4 * n * (n - 1) * b ** 4
    jhat = r1 * r3 - 6 * b * b * r1 * r1 - 6 * b * b * r2
    return r1, s2hat, r3 - 6 * b * b * r1, lhat, jhat


def denominator_radius(n: int, epsilon_degree: float, beta: float) -> float:
    """Deterministic conservative radius from the common degree-noise event."""
    b = 2.0 / epsilon_degree
    d = n - 1.
    t = n * b * math.log(n / beta)
    d1, d2, d3 = n * d, n * d * d, n * d ** 3
    e2 = 2 * d * t + t * t
    e3 = 3 * d * d * t + 3 * d * t * t + t ** 3
    return (t * (d3 + e3) + d1 * e3 + e2 * (2 * d2 + e2)
            + 6 * b * b * (d1 + t) ** 2
            + (4 * n + 2) * b * b * (d2 + e2)
            + 4 * n * (n - 1) * b ** 4)


def final_estimate(p_hat, q_hat):
    """Project the ratio when its denominator is positive, else return zero."""
    p, q = np.broadcast_arrays(np.asarray(p_hat, dtype=float), np.asarray(q_hat, dtype=float))
    ratio = np.divide(p, q, out=np.zeros_like(p), where=q > 0)
    return np.clip(ratio, -1, 1)


def emit_r(graph: GraphData, z: np.ndarray, eps_d: float, chat: float,
           correction: int, beta: float) -> dict:
    b = 2. / eps_d
    s1h, s2h, _, lh, jh = degree_moments(z, b)
    kh = s1h * chat - correction * b * b * s2h
    ph, qh = 2 * kh - lh, jh - lh
    tq = denominator_radius(len(z), eps_d, beta)
    rh = float(final_estimate(ph, qh))
    result = dict(r_hat=rh, r_true=graph.r,
                relative_error_r=(abs(rh - graph.r) / abs(graph.r)
                                  if graph.r != 0 else float('nan')),
                p_hat=ph, q_hat=qh, c_hat=chat,
                denominator_margin_diagnostic=int(qh > 3 * max(.5, tq)), tau=0.0,
                denominator_radius=tq,
                error_r=float(rh - graph.r), error_p=float(ph - graph.p),
                error_q=float(qh - graph.q))
    sum_squares = float(np.dot(z, z))
    result.update(plug_p=2*s1h*chat-sum_squares**2,
                  plug_q=s1h*float(np.sum(z**3))-sum_squares**2)
    return result


def contribution_intervals(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Public attainable interval of each owner's contribution, given noisy z."""
    positive = np.maximum(z, 0.)
    negative = np.minimum(z, 0.)
    pos = np.cumsum(positive) - positive
    neg = np.cumsum(negative) - negative
    lower = np.where(z >= 0, z * neg, z * pos)
    upper = np.where(z >= 0, z * pos, z * neg)
    return lower, upper


def local_chat(graph: GraphData, z: np.ndarray, eps_a: float,
               rng: np.random.Generator, max_pairs: int) -> float:
    n = len(z)
    if n * (n - 1) // 2 > max_pairs:
        raise ValueError('Local exact RR too large: raise --max-local-pairs explicitly')
    p = 1 / (math.exp(eps_a) + 1)
    # Stream blocks: report every possible pair, including nonedges; never
    # condition participation on whether the edge exists.
    result = 0.
    for i in range(1, n):
        for start in range(0, i, 65536):
            stop = min(i, start + 65536)
            truth = graph.a[i, start:stop].toarray().ravel().astype(bool)
            report = np.logical_xor(truth, rng.random(stop - start) < p)
            result += z[i] * float(np.dot((report.astype(float) - p) / (1 - 2 * p),
                                           z[start:stop]))
    return result


def decentralized_chat(graph: GraphData, z: np.ndarray, eps_t: float,
                       delta_d: float, eps_d: float,
                       rng: np.random.Generator) -> tuple[float, float]:
    n, b = len(z), 2 / eps_d
    lam = b * math.log(n / (2 * delta_d))
    caps = np.clip(z + lam + 1, 0, n - 1)
    largest = np.partition(caps, -2)[-2:]
    sensitivity = max(0., min(4 * n - 6., 2 * sum(largest) - 2))
    t = np.asarray(graph.a.dot(graph.degrees)).ravel()
    noisy = t + rng.laplace(0., sensitivity / eps_t, n)
    return float(np.dot(z, noisy) / 2), sensitivity


@dataclass(frozen=True)
class Plan:
    precision: int
    modulus: int
    shuffled_lanes: int
    messages_per_user: int
    statistical_security_bits: int
    integer_sensitivity: int
    normalized_sensitivity: float
    noise_alpha: float
    range_offset: float
    range_width: float
    wrap_tail: int
    wrap_probability_bound: float
    mse_c_upper_bound: float


def public_order_and_bounds(z: np.ndarray):
    order = np.argsort(-np.abs(z), kind='stable')
    zz = z[order]
    lower, upper = contribution_intervals(zz)
    return order, zz, lower, upper


def make_plan(z: np.ndarray, eps_s: float, delta_s: float,
              precision: int = 65536, wrap_probability: float = 1e-18) -> Plan:
    n = len(z)
    if n < 19 or precision < 1 or not 0 < delta_s < 1 or eps_s <= 0:
        raise ValueError('Require n>=19, precision>=1, eps_s>0, 0<delta_s<1')
    if not 0 < wrap_probability < 1:
        raise ValueError('Require 0<wrap_probability<1')
    _, zz, lower, upper = public_order_and_bounds(z)
    offset = float(lower.sum())
    width = float(np.max(upper - lower))
    if not math.isfinite(width):
        raise ArithmeticError('Non-finite public contribution interval')
    edge_change = float(abs(zz[0] * zz[1]))
    relative = min(1., edge_change / width) if width else 0.
    # +1 is conservative for stochastic rounding under coupled uniforms.
    sensitivity = max(1, math.ceil(precision * relative) + 1)
    rate = eps_s / sensitivity
    alpha = math.exp(-rate)
    if alpha == 1.:
        raise ValueError('Numerical underflow in distributed noise calibration')
    tail = math.ceil(math.log(2 / wrap_probability) / rate)
    # Shifted decoding interval [-tail, q-tail) contains [-tail, np+tail].
    minimum_modulus = n * precision + 2 * tail
    modulus = 1 << minimum_modulus.bit_length()
    if modulus >= 1 << 63:
        raise ValueError('Modulus exceeds the supported 62-bit implementation')
    security = max(1, math.ceil(math.log2((1 + math.exp(eps_s)) / delta_s)))
    lanes = max(3, math.ceil((2 * security + math.log2(modulus)) /
                             (math.log2(n) - math.log2(math.e)) + 1))
    tail_bound = min(1., 2 * math.exp(-rate * (tail + 1)) / (1 + alpha))
    variance = width ** 2 * (n / (4 * precision ** 2)
        + 2 * alpha / (precision ** 2 * (-math.expm1(-rate)) ** 2)
        + n ** 2 * tail_bound)
    return Plan(precision, modulus, lanes, lanes + 1, security,
                sensitivity, relative, alpha, offset, width, tail,
                tail_bound, variance)


def estimate_c(graph: GraphData, z: np.ndarray, plan: Plan,
               rng: np.random.Generator, simulate_shuffle: bool = True):
    """Return C_hat and diagnostics; local input is each owner's true edges.

    The analyst sums the shuffled share lanes modulo q. Permuting each lane
    explicitly is optional because sums are invariant to its permutation;
    privacy of an actual deployment requires the anonymous lanes regardless.
    """
    n = len(z)
    order, zz, lower, upper = public_order_and_bounds(z)
    if not np.isclose(float(lower.sum()), plan.range_offset) or not np.isclose(
            float(np.max(upper - lower)), plan.range_width):
        raise ValueError('Plan and public noisy degrees disagree')
    if plan.range_width == 0:
        return float(plan.range_offset), dict(wrapped=0, noise_sum=0,
              messages=0, payload_bits=0, c_target=float(plan.range_offset))
    reordered = graph.a[order, :][:, order]
    upper_edges = sp.triu(reordered, k=1, format='csr')
    owner_values = zz * np.asarray(upper_edges.T.dot(zz)).ravel()
    norm = (owner_values - lower) / plan.range_width
    if np.any(norm < -1e-10) or np.any(norm > 1 + 1e-10):
        raise ArithmeticError('True owner contribution outside public interval')
    norm = np.clip(norm, 0., 1.)
    scaled = plan.precision * norm
    integer = np.floor(scaled).astype(np.int64)
    integer += rng.random(n) < (scaled - integer)
    # Negative-binomial parameter r=1/n is the Polya(1/n, alpha) shape.
    prob_success = -math.expm1(math.log(plan.noise_alpha))  # 1 - alpha
    plus = rng.negative_binomial(1 / n, prob_success, size=n)
    minus = rng.negative_binomial(1 / n, prob_success, size=n)
    noise_sum = int(plus.sum(dtype=np.int64) - minus.sum(dtype=np.int64))
    noisy = integer + plus.astype(np.int64) - minus.astype(np.int64)
    q = plan.modulus
    mask = np.uint64(q - 1)  # power-of-two modulus
    accumulator = np.zeros(n, dtype=np.uint64)
    shuffled_total = 0
    for _ in range(plan.messages_per_user - 1):
        share = rng.integers(0, q, n, dtype=np.uint64)
        accumulator += share  # uint64 overflow is harmless modulo 2^k
        if simulate_shuffle:
            share = share[rng.permutation(n)]
        shuffled_total = (shuffled_total + int(share.sum(dtype=np.uint64))) % q
    last = (np.asarray(noisy, dtype=np.uint64) - accumulator) & mask
    if simulate_shuffle:
        last = last[rng.permutation(n)]
    shuffled_total = (shuffled_total + int(last.sum(dtype=np.uint64))) % q
    decoded = ((shuffled_total + plan.wrap_tail) % q) - plan.wrap_tail
    true_integer_sum = int(integer.sum(dtype=np.int64)) + noise_sum
    wrapped = int(decoded != true_integer_sum)
    bounded = min(float(n), max(0., decoded / plan.precision))
    chat = float(plan.range_offset + plan.range_width * bounded)
    target = float(np.dot(z[graph.source], z[graph.target]))
    if not np.isclose(float(owner_values.sum()), target, rtol=1e-12, atol=1e-5):
        raise ArithmeticError('Owner sum differs from public-weight edge sum')
    return chat, dict(wrapped=wrapped, noise_sum=noise_sum,
             messages=n * plan.messages_per_user,
             payload_bits=n * plan.messages_per_user * (q.bit_length() - 1),
             c_target=target)


def load_record(row: dict, manifest_dir: Path) -> GraphData:
    n = int(row['nodes'])
    if row.get('ba_m') and str(row['ba_m']).strip() not in ('', 'nan'):
        import networkx as nx
        ba = nx.barabasi_albert_graph(n, int(row['ba_m']),
                                     seed=int(row.get('ba_seed') or 2026))
        pairs = np.asarray(list(ba.edges()), dtype=np.int64)
        a = sp.coo_matrix((np.ones(len(pairs), dtype=np.int8),
                           (pairs[:, 0], pairs[:, 1])), shape=(n,n)).tocsr()
        return GraphData.from_adjacency(a.maximum(a.T))
    edge_path = manifest_dir / str(row['edge_file'])
    return GraphData.from_adjacency(ReadEdges(str(edge_path), n))


def rng_for(seed: int, graph_idx: int, eps_idx: int, trial: int, mechanism: int):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, graph_idx, eps_idx, trial, mechanism]))


def paired_trial(graph: GraphData, eps: float, eps_idx: int, trial: int,
                 dataset_idx: int, args) -> list[dict]:
    n = len(graph.degrees)
    degree_start = time.perf_counter()
    eps_d, eps_s = args.degree_share * eps, (1 - args.degree_share) * eps
    z = graph.degrees + rng_for(args.seed, dataset_idx, eps_idx, trial, 0).laplace(
        0., 2 / eps_d, n)
    degree_seconds = time.perf_counter() - degree_start
    conditional_target = float(np.dot(z[graph.source], z[graph.target]))
    output = []

    def append(name, chat, correction, start, extra, target=conditional_target):
        acquisition_seconds = time.perf_counter() - start
        post_start = time.perf_counter()
        result = emit_r(graph, z, eps_d, chat, correction,
                        args.beta)
        q_hat = result['q_hat']
        raw = result['p_hat'] / q_hat if q_hat > 0 else float('nan')
        output.append(dict(epsilon=eps, trial=trial, mechanism=name,
            eps_degree=eps_d, eps_second=eps_s,
            c_target_cond=target,
            error_c_cond=chat - target,
            r_unclipped=raw, clipped=int(abs(raw) > 1),
            degree_release_seconds=degree_seconds,
            acquisition_seconds=acquisition_seconds,
            postprocess_seconds=time.perf_counter()-post_start,
            simulation_total_seconds=degree_seconds + time.perf_counter()-start,
            seconds=degree_seconds + time.perf_counter()-start,
            degree_report_count=n,
            public_feedback_scalar_count=0 if name == "local" else n,
            second_stage_scalar_count=n if name == "decentral" else 0,
            second_stage_bit_count=n*(n-1)//2 if name == "local" else 0,
            share_count=extra["messages"] if name == "ikos_shuffle" else 0,
            share_payload_bits=extra["payload_bits"] if name == "ikos_shuffle" else 0,
            **extra, **result))

    start = time.perf_counter()
    plan = make_plan(z, eps_s, args.delta, args.precision,
                     args.wrap_probability)
    chat, diagnostics = estimate_c(graph, z, plan,
         rng_for(args.seed, dataset_idx, eps_idx, trial, 1),
         simulate_shuffle=not args.skip_explicit_permutations)
    if not np.isclose(diagnostics['c_target'], conditional_target,
                      rtol=1e-12, atol=1e-5):
        raise ArithmeticError('Different conditional C targets')
    append('ikos_shuffle', chat, 2, start, dict(
        messages=diagnostics['messages'], payload_bits=diagnostics['payload_bits'],
        wrapped=diagnostics['wrapped'],
        predicted_mse_c=plan.mse_c_upper_bound, precision=plan.precision,
        modulus=plan.modulus, shares_per_user=plan.messages_per_user,
        integer_sensitivity=plan.integer_sensitivity,
        relative_sensitivity=plan.normalized_sensitivity,
        public_width=plan.range_width,
        noise_alpha=plan.noise_alpha,
        security_bits=plan.statistical_security_bits,
        wrap_tail_bound=plan.wrap_probability_bound))

    if not args.no_local:
        start = time.perf_counter()
        chat = local_chat(graph, z, eps_s,
            rng_for(args.seed, dataset_idx, eps_idx, trial, 2),
            args.max_local_pairs)
        pairs = n * (n - 1) // 2
        p = 1 / (math.exp(eps_s) + 1)
        zz = z * z
        local_cond_var = p * (1 - p) / (1 - 2*p)**2 * (
            float(zz.sum())**2 - float(np.dot(zz, zz))) / 2
        append('local', chat, 2, start, dict(messages=pairs, payload_bits=pairs,
            predicted_mse_c=local_cond_var, wrapped=0))

    if not args.no_decentral:
        start = time.perf_counter()
        chat, sensitivity = decentralized_chat(graph, z, eps_s, args.delta,
            eps_d, rng_for(args.seed, dataset_idx, eps_idx, trial, 3))
        append('decentral', chat, 1, start, dict(
            messages=n, payload_bits=float('nan'),
            predicted_mse_c=sensitivity**2 * float(np.dot(z, z)) / (2 * eps_s**2),
            wrapped=0, sensitivity=sensitivity),
            target=float(np.dot(z, graph.a.dot(graph.degrees)) / 2))
    return output


