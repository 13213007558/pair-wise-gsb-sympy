"""Real inverse relations for finite Piecewise functions.

This module provides :func:`piecewise_inverse`, which turns a univariate
`Piecewise` function whose branches are polynomials of degree at most
three with rational coefficients, defined on rational intervals, into a
queryable real inverse relation.

The relation exposes one :class:`InverseBranch` per monotonic branch, with
the exact range, an inverse expression and the openness of the endpoints.
First-match priority of the original `Piecewise` is honored: parts of a
condition that are covered by earlier branches do not take part in the
inversion.  Querying the relation with :meth:`PiecewiseInverse.preimage`
keeps all real solutions of `f(x) = y` (constant branches contribute
whole intervals) and reports shared endpoint values only once.
"""
from __future__ import annotations

from sympy.core import Rational, S, sympify
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed, PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, real_roots
from sympy.series.limits import limit
from sympy.sets.sets import FiniteSet, Interval, Union
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve

__all__ = ('piecewise_inverse', 'PiecewiseInverse', 'InverseBranch')


class InverseBranch:
    """One monotonic branch of a real inverse relation.

    Attributes
    ==========

    expr : Expr
        The branch expression in `var`.
    var : Symbol
        The independent variable.
    yvar : Symbol
        The variable used by the inverse expression.
    domain : Interval or FiniteSet
        The effective domain of the branch after first-match priority of
        the original `Piecewise` has been applied.  Open endpoints are
        excluded from the domain.
    direction : str
        `'increasing'`, `'decreasing'` or `'constant'`.
    range : Interval or FiniteSet
        The exact image of `domain` under `expr`, including the
        openness of its endpoints.  A `FiniteSet` for constant branches.
    inverse : Expr or Set
        For non-constant branches, an expression in `yvar` that inverts
        `expr` on `domain`; it is valid for values in `range`.  For
        constant branches this is the whole `domain` set, i.e. every
        `x` that maps to the single value in `range`.
    """

    def __init__(self, expr, var, yvar, domain, direction, range_, inverse):
        self.expr = expr
        self.var = var
        self.yvar = yvar
        self.domain = domain
        self.direction = direction
        self.range = range_
        self.inverse = inverse

    @property
    def is_constant(self):
        return self.direction == 'constant'

    def preimage(self, y0):
        """Return the set of values in this branch that map to `y0`."""
        y0 = sympify(y0)
        if self.is_constant:
            if simplify(y0 - self.range.args[0]) == 0:
                return self.domain
            return S.EmptySet
        points = [r for r in _real_roots(self.expr - y0, self.var)
                  if self.domain.contains(r)]
        return FiniteSet(*points)

    def __repr__(self):
        return ('InverseBranch(%r, domain=%r, %s, range=%r, inverse=%r)'
                % (self.expr, self.domain, self.direction, self.range,
                   self.inverse))


class PiecewiseInverse:
    """A queryable real inverse relation of a finite `Piecewise`.

    Use :func:`piecewise_inverse` to construct instances.

    Attributes
    ==========

    piecewise : Piecewise or Expr
        The original function.
    var : Symbol
        The independent variable.
    yvar : Symbol
        The variable used by the inverse expressions of the branches.
    branches : tuple of InverseBranch
        The monotonic branches of the relation, in the order of the
        original `Piecewise`.
    """

    def __init__(self, piecewise, var, yvar, branches):
        self.piecewise = piecewise
        self.var = var
        self.yvar = yvar
        self.branches = tuple(branches)

    @property
    def range(self):
        """The union of the ranges of all branches."""
        return Union(*[branch.range for branch in self.branches])

    def preimage(self, y0):
        """Return the set of all real `x` with `f(x) = y0`.

        All solutions are kept: the result is a `FiniteSet` of the exact
        solutions, possibly united with intervals coming from constant
        branches.  Solutions shared by adjacent branches (for example at
        a common endpoint or a critical point) appear only once.
        """
        y0 = sympify(y0)
        if y0.free_symbols:
            raise ValueError('y0 must be a number, got %r' % (y0,))
        if y0.is_real is False:
            return S.EmptySet
        if y0.is_real is not True:
            raise ValueError('y0 must be a real number, got %r' % (y0,))
        return Union(*[branch.preimage(y0) for branch in self.branches])

    def __repr__(self):
        return 'PiecewiseInverse(%r, %r, branches=%r)' % (
            self.piecewise, self.var, list(self.branches))


def piecewise_inverse(pw, var=None, yvar=None):
    """Build a queryable real inverse relation for a finite `Piecewise`.

    Parameters
    ==========

    pw : Piecewise or Expr
        The function to invert.  A plain expression is treated as a single
        branch defined on the whole real line.  Every branch expression
        must be a polynomial of degree at most three in `var` with
        rational coefficients.
    var : Symbol, optional
        The independent variable.  Inferred when `pw` is univariate.
    yvar : Symbol, optional
        The variable used for the inverse expressions of the branches.
        Defaults to `y` (a dummy if `y` already occurs in `pw`).

    Returns
    =======

    PiecewiseInverse
        The inverse relation.  See :class:`PiecewiseInverse` and
        :class:`InverseBranch` for the available queries.

    Notes
    =====

    Branch conditions must describe unions of real intervals (or isolated
    points) with rational endpoints.  Conditions are resolved with
    first-match priority: the effective domain of a branch is its
    condition minus everything covered by earlier branches, so parts of
    the domain that are shadowed by earlier branches do not take part in
    the inversion.

    Each effective domain interval is split at the real critical points
    of the branch expression (kept exact, as rational or quadratic
    algebraic numbers), so that every reported branch is strictly
    monotonic and has a single inverse expression valid on its whole
    range.

    Examples
    ========

    >>> from sympy import Piecewise, And, Symbol
    >>> from sympy.solvers.piecewise_inverse import piecewise_inverse
    >>> x = Symbol('x')
    >>> pw = Piecewise((x + 1, x < 0), (x**2, And(x >= 0, x <= 2)), (8, x > 2))
    >>> rel = piecewise_inverse(pw, x)
    >>> for branch in rel.branches:
    ...     print(branch)
    InverseBranch(x + 1, domain=Interval.open(-oo, 0), increasing, range=Interval.open(-oo, 1), inverse=y - 1)
    InverseBranch(x**2, domain=Interval(0, 2), increasing, range=Interval(0, 4), inverse=sqrt(y))
    InverseBranch(8, domain=Interval.open(2, oo), constant, range={8}, inverse=Interval.open(2, oo))
    >>> rel.preimage(1)
    {1}
    >>> rel.preimage(8)
    Interval.open(2, oo)
    """
    if isinstance(pw, Piecewise):
        args = pw.args
    else:
        args = ((sympify(pw), S.true),)
    if var is None:
        fs = pw.free_symbols
        if len(fs) != 1:
            raise ValueError(
                'cannot infer the variable, pass it explicitly')
        var = fs.pop()
    var = sympify(var)
    if yvar is None:
        yvar = Symbol('y')
        if yvar in pw.free_symbols:
            yvar = Dummy('y')
    branches = []
    covered = S.EmptySet
    for expr, cond in args:
        expr = sympify(expr)
        _check_branch_expr(expr, var)
        cset = S.Reals if cond is S.true else _condition_set(cond, var)
        effective = cset - covered
        covered = Union(covered, cset)
        if effective is S.EmptySet:
            continue
        for domain in _domain_atoms(effective):
            branches.extend(_monotonic_branches(expr, var, yvar, domain))
    return PiecewiseInverse(pw, var, yvar, branches)


def _check_branch_expr(expr, var):
    if expr.free_symbols - {var}:
        raise ValueError(
            'branch expressions must be univariate in %r, got %r'
            % (var, expr))
    try:
        poly = Poly(expr, var, domain=QQ)
    except (PolynomialError, CoercionFailed):
        raise ValueError(
            'branch expressions must be polynomials with rational '
            'coefficients, got %r' % (expr,))
    if not poly.is_zero and poly.degree() > 3:
        raise ValueError(
            'branch expressions must have degree at most 3, got %r'
            % (expr,))


def _condition_set(cond, var):
    try:
        return cond.as_set()
    except Exception as exc:
        raise ValueError(
            'cannot interpret condition %r as a set of real values'
            % (cond,)) from exc


def _domain_atoms(effective):
    atoms = effective.args if isinstance(effective, Union) else (effective,)
    for atom in atoms:
        if isinstance(atom, Interval):
            yield atom
        elif isinstance(atom, FiniteSet):
            for point in atom.args:
                yield FiniteSet(point)
        else:
            raise ValueError(
                'conditions must reduce to real intervals or isolated '
                'points, got %r' % (atom,))


def _monotonic_branches(expr, var, yvar, domain):
    if isinstance(domain, FiniteSet):
        value = expr.subs(var, domain.args[0])
        return [InverseBranch(expr, var, yvar, domain, 'constant',
                              FiniteSet(value), domain)]
    poly = Poly(expr, var)
    if poly.degree() < 1:
        return [InverseBranch(expr, var, yvar, domain, 'constant',
                              FiniteSet(expr), domain)]
    critical = [r for r in roots(poly.diff().as_expr(), var)
                if r.is_real and domain.inf < r and r < domain.sup]
    critical.sort(key=lambda r: r.evalf(30))
    bounds = [domain.inf] + critical + [domain.sup]
    branches = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        lo_open = domain.left_open if i == 0 else False
        hi_open = domain.right_open if i == len(bounds) - 2 else False
        v_lo = _value_at(expr, var, lo)
        v_hi = _value_at(expr, var, hi)
        if v_lo < v_hi:
            rng = Interval(v_lo, v_hi, lo_open, hi_open)
            direction = 'increasing'
        else:
            rng = Interval(v_hi, v_lo, hi_open, lo_open)
            direction = 'decreasing'
        inverse = _inverse_expression(expr, var, yvar, lo, hi, rng)
        branches.append(InverseBranch(
            expr, var, yvar, Interval(lo, hi, lo_open, hi_open),
            direction, rng, inverse))
    return branches


def _value_at(expr, var, point):
    if point is S.Infinity:
        return limit(expr, var, S.Infinity)
    if point is S.NegativeInfinity:
        return limit(expr, var, S.NegativeInfinity)
    return expr.subs(var, point)


def _inverse_expression(expr, var, yvar, lo, hi, rng):
    solutions = solve(expr - yvar, var)
    if len(solutions) == 1:
        return solutions[0]
    samples = _range_samples(rng, 2)
    selected = [candidate for candidate in solutions
                if all(_maps_into(candidate, yvar, sample, lo, hi)
                       for sample in samples)]
    if len(selected) != 1:
        raise NotImplementedError(
            'could not select a closed-form inverse of %r on (%r, %r)'
            % (expr, lo, hi))
    return selected[0]


def _range_samples(rng, count):
    points = []
    for _ in range(count):
        lo = points[-1] if points else rng.inf
        points.append(_rational_between(lo, rng.sup))
    return points


def _rational_between(a, b):
    """Return a rational number strictly between a and b."""
    if a is S.NegativeInfinity and b is S.Infinity:
        return Rational(0)
    if a is S.NegativeInfinity:
        f = floor(b)
        return f if f < b else f - 1
    if b is S.Infinity:
        return floor(a) + 1
    precision = 15
    while True:
        midpoint = Rational((a + b).evalf(precision) / 2)
        if a < midpoint and midpoint < b:
            return midpoint
        precision *= 2


def _maps_into(candidate, yvar, y0, lo, hi):
    try:
        value = complex(candidate.subs(yvar, y0).evalf(60))
    except (TypeError, ValueError, OverflowError):
        return False
    if abs(value.imag) > 1e-40:
        return False
    if lo is not S.NegativeInfinity:
        if not value.real > float(lo.evalf(60)):
            return False
    if hi is not S.Infinity:
        if not value.real < float(hi.evalf(60)):
            return False
    return True


def _real_roots(f, var):
    try:
        poly = Poly(f, var, domain=QQ)
    except (CoercionFailed, PolynomialError):
        poly = Poly(f, var, extension=True)
    try:
        return real_roots(poly)
    except (NotImplementedError, PolynomialError) as exc:
        raise NotImplementedError(
            'cannot compute the real roots of %r exactly' % (f,)) from exc
