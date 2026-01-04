#!/usr/bin/env python3
"""
Create an Ala(m, G, k) graph.

Definition:
- B and C are the bipartition sets of a base bipartite graph G with |B|=|C|=n.
- Add A with |A|=m and connect ALL edges between A and B (complete bipartite).
- Keep edges of G between B and C.
- Add D with |D|=k and connect ALL edges between C and D (complete bipartite).

This script builds the graph using networkx and lets you choose a simple base G:
- "matching": perfect matching between B and C
- "cycle": 2-regular "wrap" cycle-like pattern (requires n>=2)
- "complete": K_{n,n}
- "random_t_regular": random t-regular bipartite graph (uses a fallback if generator is missing)
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import os
from typing import Dict, Hashable, List, Tuple, Literal, Optional, Sequence
import argparse
import sys

try:
    import networkx as nx
    from networkx.drawing.nx_pydot import write_dot
except ImportError:
    raise SystemExit("This script requires networkx: pip install networkx")


BaseGType = Literal["matching", "cycle", "complete", "random_t_regular"]


@dataclass(frozen=True)
class AlaParts:
    A: List[str]
    B: List[str]
    C: List[str]
    D: List[str]


def _bipartite_edge_swaps(
    G: nx.Graph,
    left: Sequence[Hashable],
    *,
    rng: random.Random,
    nswap: int,
    max_tries: int,
) -> None:
    """Perform bipartite-preserving edge swaps in-place for randomization."""
    if nswap <= 0 or max_tries <= 0:
        return
    left_set = set(left)
    edges: List[Tuple[Hashable, Hashable]] = []
    for u, v in G.edges():
        if u in left_set:
            edges.append((u, v))
        else:
            edges.append((v, u))
    m = len(edges)
    if m < 2:
        return

    tries = 0
    swaps = 0
    while swaps < nswap and tries < max_tries:
        tries += 1
        i = rng.randrange(m)
        j = rng.randrange(m)
        if i == j:
            continue
        u, v = edges[i]
        x, y = edges[j]
        if u == x or v == y:
            continue
        if G.has_edge(u, y) or G.has_edge(x, v):
            continue
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        G.add_edge(u, y)
        G.add_edge(x, v)
        edges[i] = (u, y)
        edges[j] = (x, v)
        swaps += 1


def _random_t_regular_bipartite_fallback(t: int, n: int, seed: Optional[int] = None) -> nx.Graph:
    """Fallback t-regular bipartite generator using Havel-Hakimi + swaps."""
    from networkx.algorithms import bipartite

    H = bipartite.havel_hakimi_graph([t] * n, [t] * n)
    if n < 2 or t in (0, n):
        return H

    rng = random.Random(seed)
    m = H.number_of_edges()
    nswap = min(10 * m, 10_000)
    max_tries = nswap * 10
    _bipartite_edge_swaps(H, list(range(n)), rng=rng, nswap=nswap, max_tries=max_tries)
    return H


def make_parts(m: int, n: int, k: int) -> AlaParts:
    if m < 0 or n <= 0 or k < 0:
        raise ValueError("Require n>0 and m,k>=0")
    A = [f"a{i}" for i in range(1, m + 1)]
    B = [f"b{i}" for i in range(1, n + 1)]
    C = [f"c{i}" for i in range(1, n + 1)]
    D = [f"d{i}" for i in range(1, k + 1)]
    return AlaParts(A=A, B=B, C=C, D=D)


def build_base_G(parts: AlaParts, base: BaseGType, t: int = 2, seed: Optional[int] = None) -> nx.Graph:
    """Return a bipartite graph on B∪C with edges determined by `base`."""
    B, C = parts.B, parts.C
    G = nx.Graph()
    G.add_nodes_from(B, bipartite=0, part="B")
    G.add_nodes_from(C, bipartite=1, part="C")

    n = len(B)

    if base == "matching":
        # Perfect matching: bi-ci
        for i in range(n):
            G.add_edge(B[i], C[i])

    elif base == "cycle":
        # Each bi connects to ci and c_{i+1} (wrap). Produces 2-regular on both sides for n>=2.
        if n < 2:
            raise ValueError("base='cycle' requires n>=2")
        for i in range(n):
            G.add_edge(B[i], C[i])
            G.add_edge(B[i], C[(i + 1) % n])

    elif base == "complete":
        # Complete bipartite K_{n,n}
        for b in B:
            for c in C:
                G.add_edge(b, c)

    elif base == "random_t_regular":
        # Random t-regular bipartite graph (each node in B and C has degree t)
        if t < 0:
            raise ValueError("t must be >= 0")
        if t > n:
            raise ValueError(f"t must be <= n (got t={t}, n={n})")
        # Use NetworkX generator if available; otherwise fallback to Havel-Hakimi + swaps.
        H = None
        if hasattr(nx, "random_regular_bipartite_graph"):
            try:
                H = nx.random_regular_bipartite_graph(t, n, n, seed=seed)
            except Exception:
                H = None
        if H is None:
            H = _random_t_regular_bipartite_fallback(t, n, seed=seed)
        # H labels nodes as 0..(2n-1). Remap to b*/c*
        mapping = {i: B[i] for i in range(n)}
        mapping.update({n + i: C[i] for i in range(n)})
        G = nx.relabel_nodes(H, mapping)
        nx.set_node_attributes(G, {b: {"bipartite": 0, "part": "B"} for b in B})
        nx.set_node_attributes(G, {c: {"bipartite": 1, "part": "C"} for c in C})
    else:
        raise ValueError(f"Unknown base='{base}'")

    return G


def build_ala_graph(
    m: int,
    n: int,
    k: int,
    base: BaseGType = "cycle",
    t: int = 2,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, AlaParts]:
    """
    Build Ala(m,G,k) where G is chosen by `base` on B∪C.

    Returns: (H, parts)
    """
    parts = make_parts(m, n, k)
    baseG = build_base_G(parts, base=base, t=t, seed=seed)

    H = nx.Graph()

    # Add all nodes with partition labels (useful for drawing/analysis)
    H.add_nodes_from(parts.A, bipartite=0, part="A")
    H.add_nodes_from(parts.B, bipartite=1, part="B")
    H.add_nodes_from(parts.C, bipartite=0, part="C")
    H.add_nodes_from(parts.D, bipartite=1, part="D")

    # A—B complete
    for a in parts.A:
        for b in parts.B:
            H.add_edge(a, b)

    # Add base edges B—C
    H.add_edges_from(baseG.edges())

    # C—D complete
    for c in parts.C:
        for d in parts.D:
            H.add_edge(c, d)

    return H, parts


def edge_list_text(G: nx.Graph) -> str:
    edges = sorted((min(u, v), max(u, v)) for u, v in G.edges())
    return "\n".join(f"{u} {v}" for u, v in edges)


def degrees_by_part(G: nx.Graph, parts: AlaParts) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {"A": {}, "B": {}, "C": {}, "D": {}}
    for a in parts.A:
        out["A"][a] = G.degree(a)
    for b in parts.B:
        out["B"][b] = G.degree(b)
    for c in parts.C:
        out["C"][c] = G.degree(c)
    for d in parts.D:
        out["D"][d] = G.degree(d)
    return out


def main():
    p = argparse.ArgumentParser(description="Build Ala(m,G,k) with a simple base bipartite G on B∪C.")
    p.add_argument("--m", type=int, required=True, help="|A|")
    p.add_argument("--n", type=int, required=True, help="|B|=|C|")
    p.add_argument("--k", type=int, required=True, help="|D|")
    p.add_argument("--base", type=str, default="cycle",
                   choices=["matching", "cycle", "complete", "random_t_regular"],
                   help="Base bipartite graph G between B and C")
    p.add_argument("--t", type=int, default=2, help="Degree t for base=random_t_regular")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for random_t_regular)")
    p.add_argument("--out_edgelist", type=str, default=None, help="Write edge list to file")
    p.add_argument("--out_graphml", type=str, default=None, help="Write GraphML to file (for Gephi, etc.)")

    args = p.parse_args()

    G, parts = build_ala_graph(args.m, args.n, args.k, base=args.base, t=args.t, seed=args.seed)

    print(f"Built Ala(m,G,k) with m={args.m}, n={args.n}, k={args.k}, base={args.base}")
    print(f"Vertices: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print("Parts sizes:", {k: len(getattr(parts, k)) for k in ["A", "B", "C", "D"]})

    degs = degrees_by_part(G, parts)
    # Print compact degree summaries
    for part_name in ["A", "B", "C", "D"]:
        vals = list(degs[part_name].values())
        if vals:
            print(f"{part_name}: degree min={min(vals)} max={max(vals)} sample={list(degs[part_name].items())[:3]}")
        else:
            print(f"{part_name}: (empty)")

    if args.out_edgelist:
        with open(args.out_edgelist, "w", encoding="utf-8") as f:
            f.write(edge_list_text(G) + "\n")
        print(f"Wrote edge list to {args.out_edgelist}")

    if args.out_graphml:
        nx.write_graphml(G, args.out_graphml)
        print(f"Wrote GraphML to {args.out_graphml}")
    
    write_dot(G, "output.dot")
    print(f"Wrote DOT to output.dot")

    os.system("dot -Tpng output.dot -o output.png")
    print(f"Wrote PNG to output.png")

    os.system("open output.png")
    print(f"Opened PNG in default viewer")


if __name__ == "__main__":
    main()
