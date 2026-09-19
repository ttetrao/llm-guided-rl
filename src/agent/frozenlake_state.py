#!/usr/bin/env python3
"""Encode minimale per FrozenLake: stato = int 0..nS-1 (r*ncol+c, gymnasium Discrete)."""


def encode(env) -> int:
    base = env.unwrapped
    if hasattr(base, "s"):
        return int(base.s)
    try:
        return int(env.get_wrapper_attr("_last_s"))
    except Exception:
        return 0
