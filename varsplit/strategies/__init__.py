"""
Strategy registry for varsplit.

To add a new strategy:
    1. Create a new file in varsplit/strategies/
    2. Subclass BaseSplitStrategy and implement split() and kfold()
    3. Add it to REGISTRY below with a string key

That's it -- no changes needed anywhere else.

Current strategies:
    random        -- baseline random split
    positional    -- hold out all variants at a set of positions
    mutational    -- hold out specific (position, AA) substitutions
    order         -- train on low-order, test on high-order mutants
    fitness       -- train on low-fitness, test on high-fitness variants

Reserved (not yet implemented):
    combinatorial -- see strategies/combinatorial.py for rationale
"""

from .base import BaseSplitStrategy
from .random import RandomSplitStrategy
from .positional import PositionalSplitStrategy
from .mutational import MutationalSplitStrategy
from .combinatorial import CombinatorialSplitStrategy
from .order import OrderSplitStrategy
from .fitness import FitnessSplitStrategy

REGISTRY: dict[str, type[BaseSplitStrategy]] = {
    "random": RandomSplitStrategy,
    "positional": PositionalSplitStrategy,
    "mutational": MutationalSplitStrategy,
    "order": OrderSplitStrategy,
    "fitness": FitnessSplitStrategy,
    # "combinatorial": CombinatorialSplitStrategy,  # reserved, not yet implemented
}


def get_strategy(name: str) -> BaseSplitStrategy:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available strategies: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]()


__all__ = [
    "BaseSplitStrategy",
    "RandomSplitStrategy",
    "PositionalSplitStrategy",
    "MutationalSplitStrategy",
    "OrderSplitStrategy",
    "FitnessSplitStrategy",
    "REGISTRY",
    "get_strategy",
]
