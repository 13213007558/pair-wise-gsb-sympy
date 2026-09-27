"""
Local Puiseux branches of bivariate polynomial curves.

The function :func:`puiseux_branches` computes all local Puiseux branches
of a curve F(x, y) = 0 at a given rational point, where F is a bivariate
polynomial with rational coefficients.  The classical Newton--Puiseux
algorithm is used: each edge of the Newton polygon of the shifted equation
gives the possible leading terms of y as a series in (x - x0) with rational
exponents, and every branch is then refined recursively until the requested
expansion order in x is reached.
"""

from __future__ import annotations

from collections import namedtuple
from math import comb

from sympy.core import Rational, expand
from sympy.core.symbol import Dummy
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, factor_list
from sympy.polys.polytools import rem as _poly_rem
from sympy.polys.polyroots import roots
from sympy.polys.rootoftools import CRootOf
from sympy.series.order import Order


class PuiseuxBranch(namedtuple('PuiseuxBranch', ('terms', 'rem', 'mult'))):
    """One local Puiseux branch of a curve F(x, y) = 0.

    The branch is

        y = y0 + sum_k c_k (x - x0)**e_k + O((x - x0)**r)

    where terms is a tuple of pairs (c_k, e_k) with exact coefficients c_k
    (possibly algebraic numbers, e.g. sqrt(2), I or CRootOf instances) and
    strictly increasing positive rational exponents e_k.  All computed
    terms have e_k strictly less than the requested expansion order.

    The remainder order rem is the exponent r of the first term that was
    not computed: the true branch differs from the truncated series only
    by terms of order >= r in (x - x0), and r is always >= the requested
    expansion order.  If rem is None the series is exact and there is no
    remainder at all.

    mult is the number of Puiseux branches of F (counted with their
    multiplicity in F) that share this truncation.  For a squarefree F and
    a fully separated branch it is 1; it is larger when several branches
    (or a repeated factor of F) still agree up to the remainder order.
    """

    __slots__ = ()

    def as_expr(self, x, x0=0, y0=0):
        """Return the truncated series as an expression in x.

        >>> from sympy.polys.puiseux_branches import puiseux_branches
        >>> from sympy.abc import x, y
        >>> branches, _ = puiseux_branches(y**2 - x**3, x, y, 0, 0, 3)
        >>> branches[0].as_expr(x)
        x**(3/2)
        """
        result = y0
        for c, e in self.terms:
            result = result + c*(x - x0)**e
        if self.rem is not None:
            result = result + Order((x - x0)**self.rem, (x, x0))
        return result


class VerticalComponent(namedtuple('VerticalComponent', ('x0', 'mult'))):
    """A vertical component x = x0 of F(x, y) = 0 with multiplicity."""
    __slots__ = ()


class PuiseuxResult(namedtuple('PuiseuxResult', ('branches', 'vertical'))):
    """Result of :func:`puiseux_branches`.

    branches is a tuple of :class:`PuiseuxBranch` and vertical is a tuple
    of :class:`VerticalComponent` describing the factors (x - x0) of F,
    i.e. vertical line components of the curve, which are reported
    separately instead of being expanded.
    """
    __slots__ = ()


def puiseux_branches(f, x, y, x0, y0, order):
    """Compute all local Puiseux branches of F(x, y) = 0 at (x0, y0).

    Parameters
    ==========

    f : Expr
        Bivariate polynomial with rational coefficients.
    x, y : Symbol
        The variables; y is expanded as a series in (x - x0).
    x0, y0 : rational
        Rational coordinates of the expansion point.
    order : rational
        Requested expansion order in x: every branch is expanded up to
        (but not including) terms of order order in (x - x0).

    Returns
    =======

    PuiseuxResult
        A named tuple (branches, vertical).  branches contains one
        PuiseuxBranch for each distinct truncation of a local branch
        y = y0 + sum_k c_k (x - x0)**e_k; see that class for the exact
        meaning of the remainder order and the multiplicity.  vertical
        contains a VerticalComponent for each power of (x - x0) dividing
        F, i.e. for a vertical line component of the curve, which is
        reported separately rather than expanded.

    Notes
    =====

    The remainder order r of a branch certifies cancellation on
    substitution: if S(x) is the truncated series of a branch with
    multiplicity m and remainder order r, then F(x, S(x)) has no terms of
    order less than m*r in (x - x0) (and vanishes identically when the
    branch is exact, i.e. when rem is None).

    Coefficients are kept exact: conjugate branches such as y = sqrt(2)*x
    and y = -sqrt(2)*x are returned as separate branches with algebraic
    coefficients and are never merged through floating point
    approximation.

    Examples
    ========

    >>> from sympy.polys.puiseux_branches import puiseux_branches
    >>> from sympy.abc import x, y

    The cusp y**2 = x**3 has two branches with fractional exponents:

    >>> for b in puiseux_branches(y**2 - x**3, x, y, 0, 0, 3).branches:
    ...     print(b)
    PuiseuxBranch(terms=((1, 3/2),), rem=None, mult=1)
    PuiseuxBranch(terms=((-1, 3/2),), rem=None, mult=1)

    Two branches tangent to y = x that split only at higher order:

    >>> print(puiseux_branches((y - x)**2 - x**6, x, y, 0, 0, 2).branches[0])
    PuiseuxBranch(terms=((1, 1),), rem=3, mult=2)

    A curve with a vertical component reports it separately:

    >>> print(puiseux_branches(x*(y - x**2), x, y, 0, 0, 3))
    PuiseuxResult(branches=(PuiseuxBranch(terms=((1, 2),), rem=None, mult=1),), vertical=(VerticalComponent(x0=0, mult=1),))
    """
    try:
        x0 = Rational(x0)
        y0 = Rational(y0)
        order = Rational(order)
    except TypeError:
        raise ValueError("the expansion point and order must be rational")
    if order <= 0:
        raise ValueError("the expansion order must be positive")
    try:
        p = Poly(f, x, y, domain='QQ')
    except CoercionFailed:
        raise ValueError(
            "F must be a bivariate polynomial with rational coefficients")
    if p.is_zero:
        raise ValueError("F must be nonzero")

    X, Y = Dummy('X'), Dummy('Y')
    g = expand(p.as_expr().subs({x: X + x0, y: Y + y0}))
    H = {}
    for (i, j), c in Poly(g, X, Y, domain='QQ').terms():
        H[(Rational(i), j)] = Rational(c)

    vertical = []
    a = min(i for i, _ in H)
    if a:
        vertical.append(VerticalComponent(x0, a))
        H = {(i - a, j): c for (i, j), c in H.items()}

    branches = [PuiseuxBranch(tuple(t), r, m)
                for t, r, m in _newton(H, [], Rational(0), order, [])]
    return PuiseuxResult(tuple(branches), tuple(vertical))


def _newton(H, alphas, acc, order, terms):
    """Recurse over the Newton polygon of the shifted equation H.

    H maps (x-exponent, y-exponent) to coefficients, alphas lists the
    algebraic numbers the coefficients may involve, acc is the x-order
    accumulated so far and terms the coefficients computed so far.
    Returns a list of (terms, rem, mult) triples.
    """
    out = []
    m = min(j for _, j in H)
    if m:
        out.append((terms, None, m))
        H = {(i, j - m): c for (i, j), c in H.items()}
    if (0, 0) in H:
        return out
    T = Dummy('T')
    for edge in _edges(H):
        mu = edge[0]
        if acc + mu >= order:
            out.append((terms, acc + mu, edge[2] - edge[4]))
            continue
        P = _char_poly(H, edge, T)
        for c, cm in _char_roots(P, T, alphas):
            cal = alphas + [a for a in c.atoms(CRootOf) if a not in alphas]
            sub = _subst(H, mu, c, cal)
            out.extend(_newton(sub, cal, acc + mu, order,
                               terms + [(c, acc + mu)]))
    return out


def _edges(H):
    """Lower-left Newton polygon edges of H with positive slope.

    Each edge is (mu, i_a, j_a, i_b, j_b) where mu = (i_b - i_a)/(j_a - j_b)
    is the exponent of the leading term Y ~ c*X**mu of the branches it
    represents.
    """
    i_min = min(i for i, _ in H)
    j0 = min(j for i, j in H if i == i_min)
    edges = []
    i_cur, j_cur = i_min, j0
    while True:
        best = None
        for i, j in H:
            if j >= j_cur:
                continue
            mu = (i - i_cur)/(j_cur - j)
            if best is None or mu < best[0] or (mu == best[0] and j < best[2]):
                best = (mu, i, j)
        if best is None:
            break
        mu, i_nxt, j_nxt = best
        edges.append((mu, i_cur, j_cur, i_nxt, j_nxt))
        i_cur, j_cur = i_nxt, j_nxt
    return edges


def _char_poly(H, edge, T):
    """Characteristic polynomial of an edge: P(c) = 0 for Y ~ c*X**mu."""
    _, i_a, j_a, i_b, j_b = edge
    di = i_b - i_a
    dj = j_a - j_b
    terms = []
    for (i, j), c in H.items():
        if j_b <= j <= j_a and (i - i_a)*dj == di*(j_a - j):
            terms.append(c*T**(j - j_b))
    return sum(terms)


def _char_roots(P, T, alphas):
    """Roots with multiplicities of the characteristic polynomial P."""
    if alphas:
        _, facs = factor_list(P, T, extension=alphas)
    else:
        _, facs = factor_list(P, T)
    out = []
    for g, m in facs:
        g = expand(g)
        deg = Poly(g, T).degree()
        if deg == 1:
            out.append((_normal(-g.coeff(T, 0)/g.coeff(T, 1), alphas), m))
            continue
        rts = roots(g, T)
        if sum(rts.values()) == deg:
            out.extend((r, m) for r in rts)
        elif not alphas:
            out.extend((CRootOf(g, k), m) for k in range(deg))
        else:
            raise NotImplementedError(
                "cannot determine the roots of the characteristic "
                "polynomial %s over the current algebraic extension" % g)
    return out


def _subst(H, mu, c, alphas):
    """Substitute Y = X**mu*(c + Y1) into H and divide by the X-content."""
    d = {}
    for (i, j), coef in H.items():
        base = i + mu*j
        for k in range(j + 1):
            key = (base, k)
            d[key] = d.get(key, 0) + coef*comb(j, k)*c**(j - k)
    shift = min(i for i, _ in d)
    out = {}
    for (i, j), coef in d.items():
        coef = _normal(coef, alphas)
        if coef != 0:
            out[(i - shift, j)] = coef
    return out


def _normal(e, alphas):
    """Expand e and reduce powers of any CRootOf modulo its minpoly."""
    e = expand(e)
    for a in alphas:
        if isinstance(a, CRootOf):
            t = Dummy('t')
            et = e.xreplace({a: t})
            if not et.is_polynomial(t):
                continue
            f = a.poly.as_expr().xreplace({a.poly.gen: t})
            r = _poly_rem(Poly(et, t), Poly(f, t))
            e = expand(r.as_expr().xreplace({t: a}))
    return e
