"""
Local Puiseux branches of plane algebraic curves.

This module computes all local Puiseux branches of a bivariate polynomial
``F(x, y) = 0`` at a given point using the Newton-Puiseux algorithm.
"""
from __future__ import annotations

from math import gcd, lcm

from sympy.core.add import Add
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.polys.polyerrors import GeneratorsError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.series.order import Order


class PuiseuxBranch:
    r"""A single local branch of ``F(x, y) = 0`` at a point.

    A branch represents a truncated Puiseux series

    .. math::

        y = y_0 + \sum_k c_k (x - x_0)^{e_k} + O((x - x_0)^r)

    with exact algebraic coefficients ``c_k`` and positive rational
    exponents ``e_k``.

    Attributes
    ==========

    x, y : the symbols the expansion is expressed in.
    point : the pair ``(x0, y0)`` the branch is centered at.
    terms : tuple of ``(exponent, coefficient)`` pairs with strictly
        increasing positive rational exponents.
    multiplicity : the number of Puiseux series of ``F`` (counted with
        their multiplicity in ``F``) that coincide with this truncation.
        It is larger than 1 when ``F`` has a multiple factor along this
        branch or when several branches only split apart at an order
        higher than the requested one.
    remainder : the order ``r`` of the first neglected term.  All terms
        with exponent strictly smaller than ``r`` are exact, i.e. the
        truncated series differs from a true branch of the curve only by
        terms of order at least ``r`` in ``x - x0``.  It is ``oo`` when
        the series terminates and the expansion is exact.
    """

    def __init__(self, x, y, point, terms, multiplicity, remainder):
        self.x = x
        self.y = y
        self.point = point
        self.terms = terms
        self.multiplicity = multiplicity
        self.remainder = remainder

    @property
    def exact(self):
        """True if the series terminates, so there is no remainder."""
        return self.remainder is oo

    @property
    def ramification(self):
        """Lcm of the denominators of the exponents of the truncation."""
        r = 1
        for e, _ in self.terms:
            r = lcm(r, e.q)
        return r

    def series(self):
        """Return the truncated series as an expression in ``x``.

        The result includes an ``O((x - x0)**r)`` term unless the
        expansion is exact.
        """
        x0, y0 = self.point
        result = y0
        for e, c in self.terms:
            result = result + c*(self.x - x0)**e
        if not self.exact:
            result = result + Order((self.x - x0)**self.remainder, (self.x, x0))
        return result

    def __repr__(self):
        return (f"PuiseuxBranch(point={self.point!r}, terms={self.terms!r}, "
                f"multiplicity={self.multiplicity!r}, "
                f"remainder={self.remainder!r})")


class PuiseuxResult:
    """Result of :func:`puiseux_branches`.

    Attributes
    ==========

    branches : list of :class:`PuiseuxBranch` with all non-vertical
        local branches of the curve at the point.
    vertical : ``None`` if the curve does not contain the vertical line
        ``x = x0``.  Otherwise the multiplicity of that vertical
        component.  Vertical components are reported separately because
        they are not graphs of Puiseux series in ``x``.
    """

    def __init__(self, branches, vertical):
        self.branches = branches
        self.vertical = vertical

    @property
    def has_vertical_component(self):
        """True if ``x = x0`` is a component of the curve."""
        return self.vertical is not None

    def __iter__(self):
        return iter(self.branches)

    def __len__(self):
        return len(self.branches)

    def __repr__(self):
        return f"PuiseuxResult(branches={self.branches!r}, vertical={self.vertical!r})"


def _lower_edges(points):
    """Negative-slope edges of the lower-left convex hull of points."""
    hull = []
    for pt in sorted(points):
        while len(hull) >= 2:
            (i1, j1), (i2, j2) = hull[-2], hull[-1]
            if (i2 - i1)*(pt[1] - j1) <= (j2 - j1)*(pt[0] - i1):
                hull.pop()
            else:
                break
        hull.append(pt)
    return [(a, b) for a, b in zip(hull, hull[1:]) if b[1] < a[1]]


def _edge_data(G, a, b):
    """Slope data and edge polynomial for the edge from a to b.

    Returns ``(p, q, N, P)`` where the edge corresponds to terms
    ``w ~ c*s**(p/q)``, ``N`` is the common value of ``i*q + j*p`` on the
    edge, and ``P`` maps exponents to coefficients of the edge
    polynomial in ``c``.
    """
    (i1, j1), (i2, j2) = a, b
    di, dj = i2 - i1, j1 - j2
    g = gcd(di, dj)
    p, q = di // g, dj // g
    N = i1*q + j1*p
    P = {}
    for (i, j), coeff in G.items():
        if i1 <= i <= i2 and (i2 - i1)*(j - j1) == (j2 - j1)*(i - i1):
            P[j - j2] = P.get(j - j2, 0) + coeff
    return p, q, N, P


def _to_dict(expr, s, w, ext):
    """Normalized monomial dict of a polynomial over the field ext."""
    if ext:
        p = Poly(expr, s, w, extension=ext)
    else:
        p = Poly(expr, s, w)
    to_sympy = p.domain.to_sympy
    return {m: to_sympy(c) for m, c in p.terms()}


def _shifted(G, p, q, N, c, s, w, ext):
    """Substitute ``s -> s**q``, ``w -> s**p*(c + w)`` and divide by s**N."""
    terms = []
    for (i, j), a in G.items():
        e = i*q + j*p - N
        for k in range(j + 1):
            terms.append(a*binomial(j, k)*c**(j - k) * s**e * w**k)
    return _to_dict(Add(*terms), s, w, ext)


def _find_roots(P, c, ext):
    """All roots of the edge polynomial with their multiplicities."""
    poly = Poly(Add(*[a*c**e for e, a in P.items()]), c, extension=ext or None)
    rts = roots(poly)
    if sum(rts.values()) == poly.degree():
        return rts
    rts = {}
    for r in poly.all_roots(radicals=False):
        rts[r] = rts.get(r, 0) + 1
    return rts


def _newton(G, A, scale, terms, order, ext, out, s, w, c):
    """Recurse over the Newton polygon of G.

    ``G`` is a monomial dict in ``(s, w)`` describing the remaining
    equation, where ``x - x0 = s**scale`` and the tail of the series is
    ``s**A * w``.  Appends ``(terms, multiplicity, remainder)`` triples
    to ``out``.
    """
    k = min(j for (_, j) in G)
    if k:
        # G = w**k*H: the accumulated series is an exact branch of
        # multiplicity k; H may contribute further branches.
        out.append((terms, k, oo))
        G = {(i, j - k): a for (i, j), a in G.items()}
        if (0, 0) in G:
            return
    for a, b in _lower_edges(G):
        p, q, N, P = _edge_data(G, a, b)
        e = Rational(A*q + p, q*scale)
        if e >= order:
            out.append((terms, a[1] - b[1], e))
            continue
        rts = _find_roots(P, c, ext)
        for root in sorted(rts, key=default_sort_key):
            ext1 = ext if root.is_Rational else ext + [root]
            G1 = _shifted(G, p, q, N, root, s, w, ext1)
            _newton(G1, A*q + p, scale*q, terms + ((e, root),), order,
                    ext1, out, s, w, c)


def puiseux_branches(F, x, y, point, order):
    r"""Compute all local Puiseux branches of ``F(x, y) = 0`` at a point.

    Parameters
    ==========

    F : bivariate polynomial in ``x`` and ``y`` with rational
        coefficients.
    x, y : symbols; the expansion is in powers of ``x - x0``.
    point : pair ``(x0, y0)`` with rational coordinates lying on the
        curve.
    order : positive rational number.  Every branch is expanded up to
        (but not including) terms of order ``order`` in ``x - x0``.

    Returns
    =======

    :class:`PuiseuxResult` with

    - ``branches``: all local branches, each a :class:`PuiseuxBranch`.
      Distinct conjugate branches (e.g. coefficients differing by a sign
      of a square root) are returned as separate branches with exact
      algebraic coefficients; they are never merged through numerical
      approximation.
    - ``vertical``: the multiplicity of the component ``x = x0`` if the
      curve contains that whole vertical line, else ``None``.

    Remainder order
    ===============

    A branch with remainder ``r`` satisfies

    .. math::

        y = y_0 + \sum_k c_k (x - x_0)^{e_k} + O((x - x_0)^r),

    i.e. all terms of order strictly less than ``r`` (in particular all
    terms below the requested ``order``) coincide with those of a true
    branch of the curve.  Substituting the truncated series back into
    ``F`` therefore gives ``F(x, y(x)) = O((x - x0)**r)`` whenever the
    branch is simple; for a branch of multiplicity ``m`` the neglected
    tail is amplified accordingly.  When ``remainder`` is ``oo`` the
    series is exact and substituting it back gives zero identically.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.series.puiseux import puiseux_branches
    >>> x, y = symbols('x y')
    >>> res = puiseux_branches(y**2 - x**3, x, y, (0, 0), 4)
    >>> for b in res:
    ...     print(b.series(), b.multiplicity)
    -x**(3/2) 1
    x**(3/2) 1

    Branches that share a tangent keep splitting as needed:

    >>> res = puiseux_branches((y - x)**2 - x**4, x, y, (0, 0), 3)
    >>> for b in res:
    ...     print(b.series())
    -x**2 + x
    x**2 + x

    A curve containing the vertical line through the point reports it
    separately:

    >>> res = puiseux_branches((x - 1)*(y**2 - x), x, y, (1, 1), 2)
    >>> res.vertical
    1
    >>> print(res.branches[0].series())
    1 + (x - 1)/2 + O((x - 1)**2, (x, 1))
    """
    F = sympify(F)
    x0, y0 = point
    x0, y0 = sympify(x0), sympify(y0)
    order = sympify(order, rational=True)
    if not (order.is_Rational and order > 0):
        raise ValueError("order must be a positive rational number")
    if not (x0.is_rational and y0.is_rational):
        raise ValueError("the expansion point must have rational coordinates")
    try:
        p = Poly(F, x, y)
    except GeneratorsError:
        raise ValueError("F must be a polynomial in x and y")
    if p.is_zero:
        raise ValueError("F must be a nonzero polynomial")
    if not (p.domain.is_ZZ or p.domain.is_QQ):
        raise ValueError("F must have rational coefficients")

    s, w, c = p.gens[0], p.gens[1], p.gens[1].dummy
    G = _to_dict(p.as_expr().subs({x: x + x0, y: y + y0}), s, w, [])
    if (0, 0) not in G:
        raise ValueError("the point %s does not lie on the curve" % (point,))

    vertical = None
    a = min(i for (i, _) in G)
    if a:
        vertical = a
        G = {(i - a, j): coeff for (i, j), coeff in G.items()}

    raw = []
    if (0, 0) in G:
        _newton(G, 0, 1, (), order, [], raw, s, w, c)

    merged = {}
    for terms, mult, rem in raw:
        if terms in merged:
            m0, r0 = merged[terms]
            merged[terms] = (m0 + mult, min(rem, r0))
        else:
            merged[terms] = (mult, rem)

    branches = [
        PuiseuxBranch(x, y, (x0, y0), terms, mult, rem)
        for terms, (mult, rem) in sorted(merged.items(),
                                         key=lambda item: default_sort_key(item[0]))
    ]
    return PuiseuxResult(branches, vertical)
