"""
Newton-Puiseux expansions of local branches of plane curves.

Given a bivariate polynomial ``F(x, y)`` with rational coefficients that is
monic and square-free in ``y``, this module computes truncated Puiseux
expansions of all local branches of the curve ``F(x, y) = 0`` near ``x = 0``.

A Taylor expansion of ``y(x)`` only exists for smooth branches.  Curves such
as ``y**2 = x**3`` have branches with fractional exponents that are captured
by Puiseux series instead.  The Newton-Puiseux algorithm used here works
with exact algebraic number coefficients: every branch is returned with
rational exponents, algebraic number coefficients and its ramification
index.  Branches that share their first terms but split apart later are
separated by as many rounds of the Newton polygon construction as needed.
Inputs that are not supported are rejected with an exception rather than by
falling back to floating point root fitting.
"""

from __future__ import annotations

from sympy.core.add import Add
from sympy.core.numbers import Rational, oo
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import binomial
from sympy.polys.domains import QQ, ZZ
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.polytools import Poly, factor_list
from sympy.polys.rootoftools import RootOf
from sympy.simplify.simplify import simplify


class PuiseuxBranch:
    """A truncated Puiseux expansion of one local branch of ``F(x, y) = 0``.

    Attributes
    ----------
    terms : tuple of (Rational, Expr)
        Pairs ``(exponent, coefficient)`` with strictly increasing rational
        exponents.  The branch is ``y = sum(coeff*x**exponent) + O(x**order)``
        where ``order`` is the truncation order passed to ``newton_puiseux``.
    ramification : int
        Ramification index of the branch: the least common denominator of
        the exponents of the full (untruncated) expansion.  A branch with
        ramification ``e`` is a genuine power series in ``x**(1/e)``.
    """

    def __init__(self, terms, ramification):
        self.terms = tuple((Rational(e), c) for e, c in terms)
        self.ramification = int(ramification)

    def series(self, x):
        """Return the truncated expansion as an expression in ``x``."""
        return Add(*[c * x**e for e, c in self.terms])

    def __repr__(self):
        return "PuiseuxBranch(terms=%r, ramification=%r)" % (
            self.terms, self.ramification)

    def __eq__(self, other):
        if not isinstance(other, PuiseuxBranch):
            return NotImplemented
        return self.terms == other.terms and self.ramification == other.ramification

    def __hash__(self):
        return hash((self.terms, self.ramification))


def newton_puiseux(F, x, y, order):
    """Compute truncated Puiseux expansions of all local branches of
    ``F(x, y) = 0`` near ``x = 0``.

    Parameters
    ----------
    F : Expr or Poly
        A bivariate polynomial in ``x`` and ``y`` with rational
        coefficients that is monic and square-free in ``y``.
    x, y : Symbol
        The variables.  The expansions are computed near ``x = 0`` and are
        power series in ``x`` with rational exponents.
    order : int or Rational
        Truncation order.  Every branch is returned with all terms whose
        exponent in ``x`` is strictly less than ``order``.

    Returns
    -------
    list of PuiseuxBranch
        One entry per local branch.  The total number of branches equals
        the degree of ``F`` in ``y``.  Branches whose expansions agree up
        to the truncation order appear as separate entries with identical
        terms.

    Raises
    ------
    ValueError
        If ``F`` is not a polynomial in ``x`` and ``y`` with rational
        coefficients, is not monic in ``y``, is not square-free, or if
        ``order`` is not positive.

    Examples
    --------
    >>> from sympy import symbols
    >>> from sympy.polys.newtonpuiseux import newton_puiseux
    >>> x, y = symbols('x y')

    The cusp ``y**2 = x**3`` has two branches with fractional exponents:

    >>> branches = newton_puiseux(y**2 - x**3, x, y, 3)
    >>> sorted([b.series(x) for b in branches], key=str)
    [-x**(3/2), x**(3/2)]
    >>> [b.ramification for b in branches]
    [2, 2]

    Two branches that are tangent at the origin are separated by as many
    rounds of the Newton polygon construction as needed:

    >>> F = (y - x)*(y - x - x**2)
    >>> sorted([b.series(x) for b in newton_puiseux(F, x, y, 3)], key=str)
    [x, x**2 + x]
    """
    try:
        order = Rational(order)
    except (TypeError, ValueError):
        raise ValueError("order must be a positive rational number")
    if order <= 0:
        raise ValueError("order must be positive")

    if isinstance(F, Poly):
        F = F.as_expr()
    try:
        F = Poly(F, x, y)
    except (PolynomialError, GeneratorsNeeded):
        raise ValueError("F must be a polynomial in x and y")
    if F.domain not in (ZZ, QQ):
        raise ValueError("F must have rational coefficients")
    F = F.set_domain(QQ)
    if F.as_poly(y).LC() != 1:
        raise ValueError("F must be monic in y")
    if any(m != 1 for _, m in F.sqf_list()[1]):
        raise ValueError("F must be square-free")

    Fd = {(Rational(i), j): c for (i, j), c in F.terms()}

    branches = []

    # Branches with a nonzero centre y(0) = c are handled by shifting
    # y -> y + c for every root c of F(0, y).
    f0 = Poly({(j,): c for (i, j), c in Fd.items() if i == 0}, y, domain=QQ)
    for g, _ in factor_list(f0)[1]:
        if g.as_expr() == y:
            continue
        d = g.degree()
        if d == 1:
            centres = [-g.as_expr().coeff(y, 0)/g.as_expr().coeff(y, 1)]
        else:
            centres = [RootOf(g.as_expr(), r) for r in range(d)]
        for c in centres:
            if c.is_rational:
                K, exts = QQ, []
            else:
                K = QQ.algebraic_field(c)
                exts = [c]
            Gd = _shift(Fd, QQ, K.from_sympy(c), K)
            for terms, ram in _newton(Gd, K, exts, order):
                branches.append(PuiseuxBranch([(Rational(0), c)] + terms, ram))

    # Branches with y(0) = 0.
    for terms, ram in _newton(Fd, QQ, [], order):
        branches.append(PuiseuxBranch(terms, ram))

    return branches


def residual_order(F, x, y, branch):
    """Return the order in ``x`` of ``F`` evaluated at a truncated branch.

    Substituting a truncated Puiseux expansion of a branch of ``F(x, y) = 0``
    back into ``F`` gives a series in ``x`` whose lowest exponent is the
    residual order.  For a branch truncated at order ``N`` the residual
    order is at least ``N``; it is ``oo`` if the expansion is an exact root.

    Examples
    --------
    >>> from sympy import symbols, oo
    >>> from sympy.polys.newtonpuiseux import newton_puiseux, residual_order
    >>> x, y = symbols('x y')
    >>> b = newton_puiseux(y**2 - x**3, x, y, 3)[0]
    >>> residual_order(y**2 - x**3, x, y, b)
    oo
    """
    if isinstance(F, Poly):
        F = F.as_expr()
    if isinstance(branch, PuiseuxBranch):
        branch = branch.series(x)
    expr = simplify(F.subs(y, branch).expand())
    if expr == 0:
        return oo
    exponents = [t.as_coeff_exponent(x)[1] for t in Add.make_args(expr)]
    return min(exponents)


def _newton(Fd, K, exts, order):
    """Return ``(terms, ramification)`` pairs for the branches of ``Fd``
    with ``y(0) = 0``.

    ``Fd`` maps ``(x_exponent, y_exponent)`` to coefficients in the domain
    ``K``; ``exts`` is the list of algebraic expressions generating ``K``.
    Only terms with exponent strictly less than ``order`` are kept, but the
    recursion always continues until every branch is separated so that the
    count of branches and their ramification indices are exact.
    """
    out = []
    if all(j >= 1 for (_, j) in Fd):
        # y == 0 is an exact branch; F is square-free so it occurs once.
        out.append(([], 1))
        Fd = {(i, j - 1): c for (i, j), c in Fd.items()}
    for (i0, j0), (i1, j1) in _edges(Fd):
        mu = (i1 - i0) / (j0 - j1)
        qp, qq = mu.p, mu.q
        s = min(qq * i + qp * j for (i, j) in Fd)
        Z = Dummy('Z')
        phi = Poly({(j - j1,): c for (i, j), c in Fd.items()
                    if qq * i + qp * j == s}, Z, domain=K)
        for g, mult in factor_list(phi)[1]:
            d = g.degree()
            if d == 1:
                b = K.from_sympy(-g.as_expr().coeff(Z, 0)/g.as_expr().coeff(Z, 1))
                roots = [(b, K, exts)]
            else:
                roots = []
                g_expr = g.as_expr()
                for r in range(d):
                    b_expr = RootOf(g_expr, r)
                    K2 = QQ.algebraic_field(*(exts + [b_expr]))
                    roots.append((K2.from_sympy(b_expr), K2, exts + [b_expr]))
            for b, K2, exts2 in roots:
                if mult == 1 and qq * order - qp <= 0:
                    terms = [(mu, K2.to_sympy(b))] if mu < order else []
                    out.append((terms, qq))
                    continue
                F1 = _transform(Fd, K, qp, qq, s, b, K2)
                for t_terms, t_ram in _newton(F1, K2, exts2, qq * order - qp):
                    terms = [(mu + e / qq, c) for (e, c) in t_terms]
                    if mu < order:
                        terms.insert(0, (mu, K2.to_sympy(b)))
                    out.append((terms, qq * t_ram))
    return out


def _edges(Fd):
    """Return the edges of the Newton polygon of ``Fd`` with negative slope.

    Each edge is a pair of points ``(i0, j0), (i1, j1)`` of the support with
    ``j0 > j1``; the branches through the edge have leading term
    ``b*x**mu`` with ``mu = (i1 - i0)/(j0 - j1) > 0``.
    """
    best = {}
    for (i, j) in Fd:
        if i not in best or j < best[i]:
            best[i] = j
    hull = []
    for pt in sorted(best.items()):
        while len(hull) >= 2:
            (i0, j0), (i1, j1) = hull[-2], hull[-1]
            i2, j2 = pt
            if (i1 - i0) * (j2 - j0) - (i2 - i0) * (j1 - j0) <= 0:
                hull.pop()
            else:
                break
        hull.append(pt)
    return [(hull[k], hull[k + 1]) for k in range(len(hull) - 1)
            if hull[k][1] > hull[k + 1][1]]


def _transform(Fd, K, qp, qq, s, b, K2):
    """Substitute ``x = t**qq`` and ``y = t**qp*(b + y1)`` into ``Fd``.

    The result is divided by ``t**s`` where ``s`` is the value of
    ``qq*i + qp*j`` on the edge.  Coefficients of ``Fd`` live in ``K`` and
    ``b`` in the extension ``K2``.
    """
    F1 = {}
    for (i, j), c in Fd.items():
        e = qq * i + qp * j - s
        if K is not K2:
            c = K2.from_sympy(K.to_sympy(c))
        for k in range(j + 1):
            term = c * K2.from_sympy(binomial(j, k)) * b ** (j - k)
            key = (e, k)
            F1[key] = F1.get(key, K2.zero) + term
    return {key: v for key, v in F1.items() if v != K2.zero}


def _shift(Fd, K, c, K2):
    """Substitute ``y -> y + c`` into ``Fd``; coefficients of ``Fd`` live in
    ``K`` and ``c`` in the extension ``K2``."""
    out = {}
    for (i, j), a in Fd.items():
        if K is not K2:
            a = K2.from_sympy(K.to_sympy(a))
        for k in range(j + 1):
            term = a * K2.from_sympy(binomial(j, k)) * c ** (j - k)
            key = (i, k)
            out[key] = out.get(key, K2.zero) + term
    return {key: v for key, v in out.items() if v != K2.zero}
