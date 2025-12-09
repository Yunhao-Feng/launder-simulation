"""Laundering pattern definitions and canonical names.

This module enumerates the 8 classic money laundering patterns used in AMLSim / AMLworld.
Each constant is accompanied by a docstring summarizing the structural definition to keep
implementations aligned with the intended transaction graph shapes.
"""

from __future__ import annotations

from typing import List

# Pattern name constants
FAN_OUT = "fan-out"
"""Fan-out pattern of a vertex v.

- v is a single source account.
- v has k >= 2 distinct outgoing neighbors w_i.
- Edges v -> w_i represent splitting a lump sum into multiple smaller transfers within
  a short window.
- Intuition: split a large amount across several accounts to reduce detection.
"""

FAN_IN = "fan-in"
"""Fan-in pattern of a vertex v.

- v is a sink account with k >= 2 distinct incoming neighbors u_i.
- Edges u_i -> v consolidate funds from many smaller inputs.
- Intuition: many smaller inflows converge into a single account.
"""

GATHER_SCATTER = "gather-scatter"
"""Gather-scatter pattern at a single vertex v.

- v participates in both a fan-in (k_in >= 2 incoming neighbors) and a fan-out
  (k_out >= 2 outgoing neighbors).
- v gathers funds from many sources then redistributes them further.
"""

SCATTER_GATHER = "scatter-gather"
"""Scatter-gather pattern between vertices v and u via shared intermediates.

- Fan-out from vertex v to a set of intermediates M = {m_1, ..., m_k} (k >= 2): v -> m_i.
- Fan-in into vertex u from the same intermediates: m_i -> u.
- The intermediate set is identical on both sides: v -> M -> u.
- Intuition: v scatters funds across intermediaries that later reconverge into u.
"""

SIMPLE_CYCLE = "simple-cycle"
"""Simple cycle pattern.

- Sequence of distinct vertices v1, v2, ..., v_L (L >= 3).
- Edges v1 -> v2 -> ... -> v_L -> v1 with no other repeated vertices.
- Intuition: funds move around a loop and return to an origin.
"""

RANDOM = "random"
"""Random walk pattern.

- Funds move through a sequence of controlled accounts without returning to the starting
  node.
- Structurally similar to a simple path (no repeated start), potentially variable length.
- Intuition: traverse a chain of controlled accounts without a recognizable cycle.
"""

BIPARTITE = "bipartite"
"""Bipartite pattern.

- Two disjoint sets of vertices: inputs A = {a_i} and outputs B = {b_j}.
- Edges only go from A to B (a_i -> b_j) with no edges within sets or from B back to A.
- Intuition: a set of senders pays a set of receivers in a dense, bipartite fashion.
"""

STACK = "stack"
"""Stack pattern.

- Multi-layer bipartite structure chaining layers of accounts.
- Layer 1: A -> B, layer 2: B -> C, optionally further layers C -> D, etc.
- Each layer only has edges from the previous set to the next set.
- Intuition: pass funds through multiple shell layers to obfuscate origin.
"""

ALL_LAUNDERING_PATTERNS: List[str] = [
    FAN_OUT,
    FAN_IN,
    GATHER_SCATTER,
    SCATTER_GATHER,
    SIMPLE_CYCLE,
    RANDOM,
    BIPARTITE,
    STACK,
]
"""List of all canonical laundering pattern names."""