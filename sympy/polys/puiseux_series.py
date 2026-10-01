
"""
Local branches of an algebraic curve as truncated Puiseux series.

This module implements the Newton-Puiseux algorithm.  Given a bivariate
polynomial ``F(x, y)`` with rational coefficients that is monic and
squarefree as a polynomial in ``y``, :func:`puiseux_series` computes the
Puiseux expansions of all local branches of the curve ``F(x, y) = 0`` at
``x = 0`` up to a finite truncation order.

Each branch is returned as a :class:`PuiseuxBranch` holding a truncated
series with rational exponents and (possibly algebraic) exact coefficients
together with its ramification index.  Branches are cycles of Puiseux
series: a branch of ramification ``e`` represents the ``e`` series obtained
by substituting ``x**(1/e) -> exp(2*pi*I*k/e) * x**(1/e)``.  The
ramification indices of all branches sum to the degree of ``F`` in ``y``.
"""

from math import comb

from sympy.core import Rational
from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.symbol import Dummy
from sympy.polys.domains import QQ
from sympy.polys.numberfields import field_isomorphism, to_number_field
from sympy.polys.polyerrors import (
    CoercionFailed,
    GeneratorsError,
    PolynomialError,
)
from sympy.polys.polytools import Poly
from sympy.utilities import public


class PuiseuxBranch:
    r"""A truncated Puiseux expansion of one local branch of a curve.

    Attributes
    ==========

    terms : tuple of ``(exponent, coefficient)`` pairs
        The terms of the expansion with ``exponent`` a rational number and
        ``coefficient`` an exact (possibly algebraic) number, sorted by
        increasing exponent.  All terms with exponent below the requested
        truncation order are included and are exact.
    ramification : int
        The ramification index ``e`` of the branch.  The full expansion is
        a series in ``x**(1/e)`` and the branch represents the cycle of the
        ``e`` conjugate series obtained from ``x**(1/e) -> zeta*x**(1/e)``
        with ``zeta**e = 1``.
    x : Symbol
        The variable of the expansion.
    """

    def __init__(self, terms, ramification, x):
        self.terms = tuple(terms)
        self.ramification = int(ramification)
        self.x = x

    def as_expr(self, x=None):
        """Return the truncated expansion as a SymPy expression."""
        if x is None:
            x = self.x
        return Add(*[coeff*x**exp for exp, coeff in self.terms])

    def __eq__(self, other):
        if not isinstance(other, PuiseuxBranch):
            return NotImplemented
        return (self.terms == other.terms and
                self.ramification == other.ramification)

    def __repr__(self):
        return "PuiseuxBranch(%s, ramification=%s)" % (
            self.as_expr(), self.ramification)


@public
def puiseux_series(F, x, y, n):
    r"""Compute the local branches of ``F(x, y) = 0`` at ``x = 0``.

    Parameters
    ==========

    F : Expr
        A polynomial in ``x`` and ``y`` with rational coefficients that is
        monic and squarefree as a polynomial in ``y``.
    x, y : Symbol
        The variables.  The expansions are computed in powers of ``x``.
    n : int or rational
        The truncation order.  Every returned branch contains all terms of
        the true Puiseux expansion with exponent strictly less than ``n``.

    Returns
    =======

    list of :class:`PuiseuxBranch`
        One entry per local branch (cycle of Puiseux series).  The sum of
        the ramification indices equals the degree of ``F`` in ``y``.

    Raises
    ======

    ValueError
        If ``F`` is not a polynomial in ``x`` and ``y`` with rational
        coefficients, is not monic in ``y``, is not squarefree, or if
        ``n`` is not positive.

    Examples
    ========

    >>> from sympy.polys.puiseux_series import puiseux_series
    >>> from sympy.abc import x, y

    The cusp ``y**2 = x**3`` is a single branch of ramification 2:

    >>> puiseux_series(y**2 - x**3, x, y, 3)
    [PuiseuxBranch(x**(3/2), ramification=2)]

    The node ``y**2 = x**2 + x**3`` has two branches:

    >>> puiseux_series(y**2 - x**2 - x**3, x, y, 3)
    [PuiseuxBranch(x**2/2 + x, ramification=1),
     PuiseuxBranch(-x**2/2 - x, ramification=1)]

    Notes
    =====

    The algorithm is the classical Newton-Puiseux construction: at each
    step the lower edges of the Newton polygon give the possible slopes
    ``m`` of the next term ``c*x**m``, the roots of the characteristic
    polynomial of an edge give the coefficients ``c``, and the substitution
    ``x -> x**q``, ``y -> x**p*(c + z)`` (with ``m = p/q`` in lowest terms)
    produces the equation for the remaining terms.  Coefficients are kept
    as exact algebraic numbers in a number field that is extended whenever
    a characteristic polynomial has a root outside the current field.
    """
    try:
        n = Rational(n)
    except TypeError:
        raise ValueError("the truncation order n must be a positive number")
    if n <= 0:
        raise ValueError("the truncation order n must be positive")
    p = _validate(F, x, y)
    Fdict = {(int(i), int(j)): QQ.to_sympy(c) for (i, j), c in p.terms()}
    branches = []
    for b in _constant_roots(p.eval(x, 0)):
        K, theta = _field_of(b)
        bK = K.from_sympy(b)
        FK = {mon: K.from_sympy(c) for mon, c in Fdict.items()}
        G = _substitute(FK, 0, 1, bK, 0, K)
        for terms, e in _branches(G, n, K, theta):
            if b != 0:
                terms = ((Rational(0), b),) + terms
            branches.append(PuiseuxBranch(terms, e, x))
    return branches


def _validate(F, x, y):
    """Check the input hypotheses and return ``F`` as a Poly over QQ."""
    try:
        p = Poly(F, x, y, domain=QQ)
    except (CoercionFailed, GeneratorsError, PolynomialError):
        raise ValueError(
            "F must be a polynomial in x and y with rational coefficients")
    d = p.degree(y)
    if d < 1:
        raise ValueError("F must have positive degree in y")
    lc = [(m, c) for m, c in p.terms() if m[1] == d]
    if lc != [((0, d), 1)]:
        raise ValueError("F must be monic as a polynomial in y")
    if any(m > 1 for _, m in p.sqf_list()[1]):
        raise ValueError("F must be squarefree")
    return p


def _constant_roots(F0):
    """All roots of the univariate polynomial ``F0`` over QQ."""
    roots = []
    for f, _ in F0.factor_list()[1]:
        roots.extend(f.all_roots())
    return roots


def _field_of(a):
    """Return ``(K, theta)`` with ``K = QQ.algebraic_field(theta)``."""
    if a.is_Rational:
        return QQ, None
    theta = to_number_field([a])
    return QQ.algebraic_field(theta), theta


def _extend_field(K, theta, a):
    """Extend the field ``K = QQ(theta)`` by the algebraic number ``a``.

    Returns ``(K2, theta2, a2)`` where ``a2`` is ``a`` as an element of
    ``K2``.  If ``a`` already lies in ``K`` then ``K2 is K``.
    """
    if a.is_Rational:
        return K, theta, K.from_sympy(a)
    if theta is None:
        theta2 = to_number_field([a])
    elif _in_field(K, a):
        return K, theta, K.from_sympy(a)
    elif field_isomorphism(theta, a) is not None:
        # QQ(theta, a) = QQ(a), so keep a as the generator.
        theta2 = AlgebraicNumber(a)
    else:
        theta2 = to_number_field([theta, a])
    K2 = QQ.algebraic_field(theta2)
    return K2, theta2, K2.from_sympy(a)


def _in_field(K, a):
    """Check whether the algebraic number ``a`` lies in the field ``K``."""
    try:
        K.from_sympy(a)
    except CoercionFailed:
        return False
    return True


def _substitute(F, p, q, c, E0, K):
    """Compute ``G(x, z) = x**(-q*E0) * F(x**q, x**p*(c + z))``.

    ``F`` is a dict mapping ``(i, j)`` to a coefficient in ``K`` and so is
    the result.  ``E0 = i + (p/q)*j`` on the edge being followed, so all
    exponents in the result are nonnegative integers.
    """
    shift = q*E0
    G = {}
    for (i, j), a in F.items():
        base = q*i + p*j - shift
        cpow = K.one
        cpows = [K.one]
        for _ in range(j):
            cpow *= c
            cpows.append(cpow)
        for k in range(j + 1):
            coeff = a*K(comb(j, k))*cpows[j - k]
            key = (base, k)
            G[key] = G.get(key, K.zero) + coeff
    return {mon: cf for mon, cf in G.items() if cf != K.zero}


def _newton_edges(points):
    """Lower-hull edges of positive slope of a Newton polygon.

    ``points`` is a list of ``(i, j)`` with nonnegative integer ``i`` and
    ``j``.  Returns a list of ``(p, q, E0, on)`` where the edge has slope
    ``m = p/q`` in lowest terms, ``E0 = i + m*j`` for the points ``on``
    the edge, sorted by increasing slope.
    """
    slopes = set()
    for i1, j1 in points:
        for i2, j2 in points:
            if j1 > j2 and i2 >= i1:
                slopes.add(Rational(i2 - i1, j1 - j2))
    edges = []
    for m in sorted(slopes):
        if m <= 0:
            continue
        E0 = min(i + m*j for i, j in points)
        on = [(i, j) for i, j in points if i + m*j == E0]
        if len(on) >= 2:
            edges.append((m.p, m.q, E0, on))
    return edges


def _char_roots(on, q, F, K):
    """Roots of the characteristic polynomial of an edge.

    The characteristic polynomial of the edge is
    ``phi(T) = sum c_ij*T**j`` over the edge points.  Since the exponents
    ``j`` on an edge of slope ``p/q`` are all congruent modulo ``q``, this
    is ``T**jmin * chi(T**q)`` and the nonzero roots of ``phi`` are the
    ``q``-th roots of the roots of ``chi``.  Returns the roots of ``chi``
    as SymPy expressions.
    """
    jmin = min(j for _, j in on)
    chi = {}
    for i, j in on:
        k, r = divmod(j - jmin, q)
        assert r == 0
        chi[k] = chi.get(k, K.zero) + F[(i, j)]
    poly = Poly({(k,): v for k, v in chi.items()}, Dummy('T'), domain=K)
    roots = []
    for f, _ in poly.factor_list()[1]:
        roots.extend(f.all_roots())
    return roots


def _branches(F, prec, K, theta):
    """Positive-order branches of ``F`` with terms of exponent below ``prec``.

    ``F`` is a dict mapping ``(i, j)`` to coefficients in the domain
    ``K``, monic in ``y`` of degree at least 1.  Returns a list of
    ``(terms, e)`` where ``terms`` is a tuple of ``(Rational, Expr)``
    pairs with positive increasing exponents below ``prec`` and ``e`` is
    the ramification index.
    """
    result = []
    while True:
        F = {mon: c for mon, c in F.items() if c != K.zero}
        d = max(j for _, j in F)
        jmin = min(j for _, j in F)
        if jmin > 0:
            # z divides F, so z = 0 is an exact root (simple, since F is
            # squarefree in z): a branch with no further terms.
            result.append(((), 1))
            F = {(i, j - 1): c for (i, j), c in F.items()}
            continue
        if d == 0 or (0, 0) in F:
            return result
        if d == 1:
            # F = y + b(x) with b(0) = 0, so y = -b(x) exactly.
            terms = tuple((Rational(i), _to_expr(K, -c))
                          for (i, j), c in sorted(F.items())
                          if j == 0 and Rational(i) < prec)
            result.append((terms, 1))
            return result
        for p, q, E0, on in _newton_edges(list(F)):
            m = Rational(p, q)
            new_prec = q*prec - p
            for u in _char_roots(on, q, F, K):
                a = u if q == 1 else u**Rational(1, q)
                K2, theta2, c = _extend_field(K, theta, a)
                if K2 is not K:
                    F = {mon: K2.from_sympy(K.to_sympy(cf))
                         for mon, cf in F.items()}
                    K, theta = K2, theta2
                G = _substitute(F, p, q, c, E0, K)
                if new_prec <= 0:
                    # The next term is already beyond the truncation
                    # order; only the cycle structure is still needed.
                    for e in _structures(G, K, theta):
                        result.append(((), q*e))
                else:
                    for terms, e in _branches(G, new_prec, K, theta):
                        full = ((m, _to_expr(K, c)),) + tuple(
                            (m + r/q, cf) for r, cf in terms)
                        result.append((full, q*e))
        return result


def _to_expr(K, a):
    """Convert a domain element to a plain SymPy expression."""
    return _expand_algebraics(K.to_sympy(a))


def _expand_algebraics(e):
    """Rewrite AlgebraicNumber objects as plain nested radicals/RootOfs."""
    if isinstance(e, AlgebraicNumber):
        return _expand_algebraics(e.as_expr())
    if e.is_Atom:
        return e
    return e.func(*[_expand_algebraics(arg) for arg in e.args])


def _structures(F, K, theta):
    """Ramification indices of the positive-order branches of ``F``.

    Like :func:`_branches` but only the cycle structure (number of
    branches and their ramification indices) is determined, not the terms.
    This is used when the truncation order is already exhausted: the
    recursion still has to be followed until all remaining multiplicities
    are 1, because a simple root of ``F(0, z)`` lifts to a unique integral
    power series by the implicit function theorem.
    """
    result = []
    while True:
        F = {mon: c for mon, c in F.items() if c != K.zero}
        d = max(j for _, j in F)
        jmin = min(j for _, j in F)
        if jmin > 0:
            result.append(1)
            F = {(i, j - 1): c for (i, j), c in F.items()}
            continue
        if d == 0 or (0, 0) in F:
            return result
        mu = min(j for i, j in F if i == 0)
        if mu == 1:
            # 0 is a simple root of F(0, z), so it lifts to a unique
            # integral power series: a single branch of ramification 1.
            result.append(1)
            return result
        for p, q, E0, on in _newton_edges(list(F)):
            for u in _char_roots(on, q, F, K):
                a = u if q == 1 else u**Rational(1, q)
                K2, theta2, c = _extend_field(K, theta, a)
                if K2 is not K:
                    F = {mon: K2.from_sympy(K.to_sympy(cf))
                         for mon, cf in F.items()}
                    K, theta = K2, theta2
                G = _substitute(F, p, q, c, E0, K)
                for e in _structures(G, K, theta):
                    result.append(q*e)
        return result
