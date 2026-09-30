from __future__ import annotations

from sympy.core.cache import lazy_function

from .quaternion import Quaternion

bch_expansion = lazy_function('sympy.algebras.bch', 'bch_expansion')
hall_basis = lazy_function('sympy.algebras.bch', 'hall_basis')

__all__ = ["Quaternion", "bch_expansion", "hall_basis"]
