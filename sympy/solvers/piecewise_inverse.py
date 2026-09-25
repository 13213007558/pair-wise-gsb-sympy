"""
Invert a finite Piecewise of low-degree polynomial pieces into a
queryable real inverse relation.

The main entry point is piecewise_inverse.  It takes a Piecewise whose
pieces are polynomials of degree at most three with rational
coefficients and whose conditions describe rational intervals, and
returns a PiecewiseInverse object.  The object exposes the monotonic
branches of the original function together with their ranges, endpoint
openness and inverse expressions, and can be queried for all real
preimages of a given value.
"""
from __future__ import annotations

from sympy.core import S, sympify
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import And, Or
from sympy.polys import Poly, roots, real_roots
from sympy.polys.polyerrors import PolynomialError
from sympy.series.limits import limit
from sympy.sets.sets import Interval, FiniteSet, Union
from sympy.simplify.simplify import simplify
from sympy.solvers.inequalities import solve_univariate_inequality

__all__ = ['piecewise_inverse', 'PiecewiseInverse', 'InverseBranch']


def _condition_as_set(cond, symbol):
    """Convert a Piecewise condition to a set over the reals."""
    if cond is S.true:
        return S.Reals
    if cond is S.false:
        return S.EmptySet
    if isinstance(cond, And):
        result = S.Reals
        for arg in cond.args:
            result = result.intersect(_condition_as_set(arg, symbol))
        return result
    if isinstance(cond, Or):
        return Union(*[_condition_as_set(arg, symbol) for arg in cond.args])
    if isinstance(cond, Relational):
        return solve_univariate_inequality(cond, symbol, relational=False)
    raise ValueError("unsupported Piecewise condition: %r" % (cond,))


def _as_polynomial(expr, symbol):
    """Validate that expr is a rational polynomial of degree at most 3."""
    try:
        poly = Poly(sympify(expr), symbol)
    except PolynomialError:
        raise ValueError(
            "piece %r is not a polynomial in %s" % (expr, symbol))
    if poly.degree() > 3:
        raise ValueError(
            "piece %r has degree %s, only degrees up to 3 are supported"
            % (expr, poly.degree()))
    for coeff in poly.all_coeffs():
        if not coeff.is_rational:
            raise ValueError(
                "piece %r does not have rational coefficients" % (expr,))
    return poly


def _check_interval(ivl):
    for end in (ivl.start, ivl.end):
        if not (end.is_rational or end in (S.Infinity, S.NegativeInfinity)):
            raise ValueError(
                "interval endpoint %r is not rational" % (end,))


def _set_to_intervals(eff):
    """Normalize an effective domain set into intervals and points."""
    if eff is S.EmptySet:
        return []
    if isinstance(eff, Interval):
        _check_interval(eff)
        return [eff]
    if isinstance(eff, FiniteSet):
        return [eff]
    if isinstance(eff, Union):
        parts = []
        for arg in eff.args:
            parts.extend(_set_to_intervals(arg))
        return parts
    raise ValueError("unsupported effective domain: %r" % (eff,))


def _critical_points(poly, ivl, symbol):
    """Exact roots of the derivative in the interior of ivl, sorted."""
    dpoly = poly.diff()
    if dpoly.degree() < 1:
        return []
    interior = ivl.interior
    points = []
    for root in roots(dpoly, symbol):
        if root.is_real and interior.contains(root) is S.true:
            points.append(root)
    points.sort(key=lambda r: r.evalf(40))
    return points


def _split_interval(poly, ivl, symbol):
    """Split ivl at the critical points of poly.

    Critical points are kept as closed endpoints of both neighbouring
    subintervals so that extremum values remain attainable; duplicate
    solutions are removed when the relation is queried.
    """
    crits = _critical_points(poly, ivl, symbol)
    bounds = [ivl.start] + crits + [ivl.end]
    last = len(bounds) - 2
    result = []
    for i in range(last + 1):
        left_open = ivl.left_open if i == 0 else False
        right_open = ivl.right_open if i == last else False
        result.append(Interval(bounds[i], bounds[i + 1],
                               left_open, right_open))
    return result


def _sample_point(ivl):
    if ivl.start is S.NegativeInfinity:
        return ivl.end - 1
    if ivl.end is S.Infinity:
        return ivl.start + 1
    return (ivl.start + ivl.end)/2


def _range_of(poly, ivl, symbol):
    """Range of a monotonic polynomial over an interval."""
    expr = poly.as_expr()
    if ivl.start is S.NegativeInfinity:
        low_at_start = limit(expr, symbol, S.NegativeInfinity)
    else:
        low_at_start = expr.subs(symbol, ivl.start)
    if ivl.end is S.Infinity:
        high_at_end = limit(expr, symbol, S.Infinity)
    else:
        high_at_end = expr.subs(symbol, ivl.end)
    deriv_sign = poly.diff().as_expr().subs(symbol, _sample_point(ivl))
    if deriv_sign > 0:
        return Interval(low_at_start, high_at_end,
                        ivl.left_open, ivl.right_open), 'increasing'
    return Interval(high_at_end, low_at_start,
                    ivl.right_open, ivl.left_open), 'decreasing'


def _same_value(a, b):
    if a == b:
        return True
    return bool(simplify(a - b) == 0)


class InverseBranch:
    """One monotonic branch of a real inverse relation.

    Attributes
    ==========

    expr : Expr
        The polynomial of the branch as an expression in symbol.
    x_interval : Interval or FiniteSet
        The effective domain of the branch.  This already accounts for
        Piecewise priority: parts of a condition shadowed by earlier
        pieces are excluded.
    y_interval : Interval or FiniteSet
        The range of the branch, with endpoint openness mirroring the
        domain and the monotonic direction.
    kind : str
        One of 'increasing', 'decreasing' or 'constant'.
    """

    def __init__(self, expr, x_interval, y_interval, kind, symbol):
        self.expr = expr
        self.x_interval = x_interval
        self.y_interval = y_interval
        self.kind = kind
        self.symbol = symbol

    @property
    def is_constant(self):
        return self.kind == 'constant'

    def inverse_expression(self, y=None):
        """Expressions for x in terms of y on this branch.

        For a nonconstant branch this returns the tuple of exact
        algebraic solutions of expr = y; only those lying in
        x_interval are actual preimages for a given value of y.
        For a constant branch the whole domain interval is returned,
        since every point of it maps to the same value.
        """
        if y is None:
            y = Dummy('y')
        y = sympify(y)
        if self.is_constant:
            return self.x_interval
        poly = Poly(self.expr - y, self.symbol)
        return tuple(roots(poly, self.symbol))

    def solve(self, y):
        """All preimages of the rational value y on this branch."""
        y = sympify(y)
        if self.is_constant:
            if _same_value(self.expr, y):
                return [self.x_interval]
            return []
        poly = Poly(self.expr - y, self.symbol)
        result = []
        for root in real_roots(poly, self.symbol):
            if self.x_interval.contains(root) is S.true:
                result.append(root)
        return result

    def __repr__(self):
        return ("InverseBranch(%s, %s, %s, kind=%s)"
                % (self.expr, self.x_interval, self.y_interval, self.kind))


class PiecewiseInverse:
    """Queryable real inverse relation of a finite Piecewise.

    Examples
    ========

    >>> from sympy import Piecewise
    >>> from sympy.abc import x
    >>> from sympy.solvers.piecewise_inverse import piecewise_inverse
    >>> pw = Piecewise((x, x < 0), (x**2, True))
    >>> inv = piecewise_inverse(pw, x)
    >>> inv.solve(1)
    [1]
    >>> inv.solve(-1)
    [-1]
    """

    def __init__(self, branches, symbol):
        self.branches = tuple(branches)
        self.symbol = symbol

    def __iter__(self):
        return iter(self.branches)

    def __len__(self):
        return len(self.branches)

    @property
    def range(self):
        """The set of all values attained by the original function."""
        return Union(*[b.y_interval for b in self.branches])

    def attains(self, y):
        """Return True if y is attained by the original function."""
        return self.range.contains(sympify(y)) is S.true

    def solve(self, y):
        """All real preimages of y, deduplicated.

        Returns a list whose elements are either exact x values or, for
        constant pieces, the whole x interval mapping to y.
        """
        result = []
        for branch in self.branches:
            for sol in branch.solve(y):
                if isinstance(sol, (Interval, FiniteSet)):
                    result.append(sol)
                elif not any(_same_value(sol, old) for old in result
                             if not isinstance(old, (Interval, FiniteSet))):
                    result.append(sol)
        return result

    def __repr__(self):
        return "PiecewiseInverse(%s)" % (list(self.branches),)


def piecewise_inverse(pw, symbol):
    """Build the real inverse relation of a finite Piecewise.

    Parameters
    ==========

    pw : Piecewise
        Each piece must be a polynomial in symbol of degree at most
        three with rational coefficients, and each condition must
        describe rational intervals of the real line.
    symbol : Symbol
        The independent variable.

    Returns
    =======

    PiecewiseInverse
        The inverse relation, split into monotonic branches.  Branch
        domains respect the priority of the original Piecewise: regions
        where an earlier condition already matched are removed from
        later pieces before inversion.

    Examples
    ========

    >>> from sympy import Piecewise
    >>> from sympy.abc import x
    >>> from sympy.solvers.piecewise_inverse import piecewise_inverse
    >>> pw = Piecewise((x + 1, x <= 0), (2*x + 1, True))
    >>> inv = piecewise_inverse(pw, x)
    >>> inv.solve(1)
    [0]
    """
    if not isinstance(symbol, Symbol):
        raise ValueError("expected a Symbol, got %r" % (symbol,))
    pw = sympify(pw)
    if not isinstance(pw, Piecewise):
        raise ValueError("expected a Piecewise, got %r" % (pw,))

    branches = []
    covered = S.EmptySet
    for expr, cond in pw.args:
        poly = _as_polynomial(expr, symbol)
        cond_set = _condition_as_set(cond, symbol)
        effective = cond_set - covered
        covered = covered.union(cond_set)
        for part in _set_to_intervals(effective):
            if isinstance(part, FiniteSet):
                for point in part.args:
                    value = poly.as_expr().subs(symbol, point)
                    branches.append(InverseBranch(
                        value, FiniteSet(point), FiniteSet(value),
                        'constant', symbol))
                continue
            if poly.degree() == 0:
                value = poly.as_expr()
                branches.append(InverseBranch(
                    value, part, FiniteSet(value), 'constant', symbol))
                continue
            for sub in _split_interval(poly, part, symbol):
                y_interval, kind = _range_of(poly, sub, symbol)
                branches.append(InverseBranch(
                    poly.as_expr(), sub, y_interval, kind, symbol))
    return PiecewiseInverse(branches, symbol)
