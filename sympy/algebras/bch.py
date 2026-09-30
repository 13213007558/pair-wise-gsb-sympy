"""
Baker-Campbell-Hausdorff expansion in the Lyndon basis.

This module computes the formal series

.. math::

    Z = \\log(\\exp(X) \\exp(Y))

for noncommuting symbols ``X`` and ``Y`` up to a prescribed truncation
order, with exact rational coefficients.  The result is expressed in the
Lyndon basis of the free Lie algebra on two generators, so that
commutators equivalent under antisymmetry and the Jacobi identity are
automatically reduced to the same basis element.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

from sympy.core.add import Add
from sympy.core.numbers import Integer, Rational
from sympy.physics.quantum.commutator import Commutator

__all__ = ["bch_expansion", "hall_basis"]


def _lyndon_words(max_length):
    """All Lyndon words over the alphabet ``{0, 1}`` of length at most
    ``max_length``, in lexicographic order (Duval's algorithm)."""
    words = []
    word = [-1]
    while word:
        word[-1] += 1
        words.append(tuple(word))
        period = len(word)
        while len(word) < max_length:
            word.append(word[len(word) - period])
        while word and word[-1] == 1:
            word.pop()
    return words


@lru_cache(maxsize=None)
def _hall_data(max_length):
    """Lyndon basis data up to the given word length.

    Returns the list of Lyndon words sorted by length then
    lexicographically, the standard factorization of each word, and the
    associative expansion of each basis Lie element as a dict mapping
    words to rational coefficients.
    """
    words = sorted(_lyndon_words(max_length), key=lambda w: (len(w), w))
    word_set = set(words)
    factorization = {}
    expansion = {}
    for w in words:
        if len(w) == 1:
            expansion[w] = {w: Rational(1)}
            continue
        for i in range(1, len(w)):
            v = w[i:]
            if v in word_set:
                u = w[:i]
                break
        factorization[w] = (u, v)
        eu = expansion[u]
        ev = expansion[v]
        we = {}
        for wu, cu in eu.items():
            for wv, cv in ev.items():
                we[wu + wv] = we.get(wu + wv, 0) + cu * cv
                we[wv + wu] = we.get(wv + wu, 0) - cu * cv
        expansion[w] = we
    return words, factorization, expansion


def _bch_series(order):
    """log(exp(X) exp(Y)) as a truncated associative series.

    Returns a dict mapping nonempty words over ``{0, 1}`` of length at
    most ``order`` to exact rational coefficients, where ``0`` and ``1``
    stand for ``X`` and ``Y`` respectively.
    """
    exp_prod = {}
    for i in range(order + 1):
        for j in range(order + 1):
            if 0 < i + j <= order:
                exp_prod[(0,) * i + (1,) * j] = Rational(
                    1, factorial(i) * factorial(j))
    series = {}
    power = dict(exp_prod)
    for k in range(1, order + 1):
        coef = Rational((-1) ** (k + 1), k)
        for w, c in power.items():
            series[w] = series.get(w, 0) + coef * c
        new_power = {}
        for w1, c1 in power.items():
            for w2, c2 in exp_prod.items():
                if len(w1) + len(w2) <= order:
                    w = w1 + w2
                    new_power[w] = new_power.get(w, 0) + c1 * c2
        power = new_power
    return series


def _reduce_to_hall(series, order):
    """Rewrite a Lie series given on associative words in the Lyndon basis.

    The associative expansion of each Lyndon basis element has the
    corresponding Lyndon word as its smallest word (with coefficient
    one), so processing the basis elements in increasing order gives a
    triangular reduction.  Returns a list of ``(lyndon_word, coefficient)``
    pairs.
    """
    words, _, expansion = _hall_data(order)
    series = dict(series)
    result = []
    for w in words:
        c = series.get(w)
        if c:
            result.append((w, c))
            for ew, ec in expansion[w].items():
                series[ew] = series.get(ew, 0) - c * ec
    return result


def _bracket(word, factorization, X, Y):
    """The nested commutator associated with a Lyndon word."""
    if len(word) == 1:
        return X if word[0] == 0 else Y
    u, v = factorization[word]
    return Commutator(_bracket(u, factorization, X, Y),
                      _bracket(v, factorization, X, Y))


def _validate_order(order):
    if isinstance(order, Integer):
        order = int(order)
    if not isinstance(order, int) or order < 1:
        raise ValueError(
            "the truncation order must be a positive integer, got %r" % (order,))
    return order


def hall_basis(order, X, Y):
    r"""The Lyndon basis of the free Lie algebra on ``X`` and ``Y``.

    Explanation
    ===========

    The Lyndon basis is the Hall basis built from Lyndon words over the
    ordered alphabet ``X < Y``.  Each Lyndon word ``w`` defines a basis
    element: the letters give ``X`` and ``Y`` themselves, and a longer
    word with standard factorization ``w = u v`` (where ``v`` is the
    longest proper Lyndon suffix of ``w``) gives the nested commutator
    ``[P_u, P_v]``.  Because the basis quotients out antisymmetry and
    the Jacobi identity, equivalent commutators reduce to the same
    basis element.

    Parameters
    ==========

    order : positive integer
        Maximum total degree of the returned basis elements.
    X, Y :
        The two generators, typically noncommutative symbols.

    Returns
    =======

    list
        The basis elements of degree at most ``order``, sorted by
        degree and then lexicographically as words.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.algebras.bch import hall_basis
    >>> X, Y = symbols('X Y', commutative=False)
    >>> hall_basis(3, X, Y)
    [X, Y, [X,Y], [X,[X,Y]], -[Y,[X,Y]]]

    References
    ==========

    .. [1] Reutenauer, C. "Free Lie Algebras", Oxford University Press, 1993.
    """
    order = _validate_order(order)
    words, factorization, _ = _hall_data(order)
    return [_bracket(w, factorization, X, Y) for w in words]


def bch_expansion(X, Y, order):
    r"""Baker-Campbell-Hausdorff series truncated at the given order.

    Explanation
    ===========

    Computes

    .. math::

        Z = \log(\exp(X) \exp(Y))

    for noncommuting ``X`` and ``Y`` as a linear combination of nested
    commutators with exact rational coefficients, including all terms of
    total degree at most ``order``.  Terms of higher degree are dropped;
    every returned coefficient is exact for the stated truncation order.

    The result is expressed in the Lyndon basis of the free Lie algebra
    on ``X < Y`` (see :func:`hall_basis`), so commutators that are
    equivalent under antisymmetry and the Jacobi identity are
    automatically combined into a single basis element.  If ``X`` and
    ``Y`` commute (for instance ordinary commutative symbols), the
    series degenerates to ``X + Y``.

    Parameters
    ==========

    X, Y :
        The two noncommuting symbols (or operators).
    order : positive integer
        Truncation order: all terms of total degree up to ``order`` are
        included.  The computational cost grows exponentially with
        ``order``; values up to about 10 are practical.

    Returns
    =======

    Expr
        The truncated BCH series as a sum of rational multiples of
        nested :class:`~sympy.physics.quantum.commutator.Commutator`
        objects.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.algebras.bch import bch_expansion
    >>> X, Y = symbols('X Y', commutative=False)
    >>> bch_expansion(X, Y, 2)
    X + Y + [X,Y]/2
    >>> bch_expansion(X, Y, 3)
    X + Y + [X,Y]/2 + [X,[X,Y]]/12 - [Y,[X,Y]]/12

    With commuting symbols the series reduces to the trivial one:

    >>> x, y = symbols('x y')
    >>> bch_expansion(x, y, 5)
    x + y

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula
    """
    order = _validate_order(order)
    series = _bch_series(order)
    coefficients = _reduce_to_hall(series, order)
    _, factorization, _ = _hall_data(order)
    return Add(*[c * _bracket(w, factorization, X, Y)
                 for w, c in coefficients])
