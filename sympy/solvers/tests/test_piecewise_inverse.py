from __future__ import annotations

from sympy.core import S
from sympy.core.numbers import Rational, oo
from sympy.core.symbol import Symbol
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval, FiniteSet, Union, Intersection
from sympy.simplify.simplify import simplify
from sympy.solvers.solveset import solveset
from sympy.core.relational import Eq
from sympy.testing.pytest import raises
from sympy.abc import x, y

from sympy.solvers.piecewise_inverse import (
    piecewise_inverse, PiecewiseInverse, InverseBranch)


def _points(sols):
    return [s for s in sols if not isinstance(s, (Interval, FiniteSet))]


def _oracle_points(oracle):
    if isinstance(oracle, FiniteSet):
        return list(oracle.args)
    if isinstance(oracle, Union):
        points = []
        for arg in oracle.args:
            points.extend(_oracle_points(arg))
        return points
    if isinstance(oracle, Intersection):
        points = []
        for arg in oracle.args:
            points.extend(_oracle_points(arg))
        rest = [a for a in oracle.args if not isinstance(a, FiniteSet)]
        return [p for p in points
                if all(a.contains(p) is S.true for a in rest)]
    return []


def _check_acceptance(pw, inv, value):
    """Forward substitution, domain restriction and completeness."""
    sols = inv.solve(value)
    for sol in _points(sols):
        # forward substitution: f(sol) == value
        assert simplify(pw.subs(x, sol) - value) == 0
        # domain restriction: sol lies in some effective branch domain
        assert any(b.x_interval.contains(sol) is S.true
                   for b in inv.branches)
    # completeness: every solution solveset finds is also found here
    oracle = solveset(Eq(pw, value), x, domain=Interval(-oo, oo))
    for sol in sols:
        if isinstance(sol, (Interval, FiniteSet)):
            assert sol.is_subset(oracle)
    found = sorted(_points(sols), key=lambda s: s.evalf(30))
    expected = sorted(_oracle_points(oracle), key=lambda s: s.evalf(30))
    assert len(found) == len(expected)
    for got, want in zip(found, expected):
        assert abs(got.evalf(30) - want.evalf(30)) < Rational(1, 10)**20


def test_linear_pieces_endpoint_openness():
    pw = Piecewise((x, x < 0), (x + 1, True))
    inv = piecewise_inverse(pw, x)
    assert isinstance(inv, PiecewiseInverse)
    assert all(isinstance(b, InverseBranch) for b in inv.branches)
    b1, b2 = inv.branches
    assert b1.x_interval == Interval.open(-oo, 0)
    assert b1.y_interval == Interval.open(-oo, 0)
    assert b1.kind == 'increasing'
    assert b2.x_interval == Interval.Ropen(0, oo)
    assert b2.y_interval == Interval.Ropen(1, oo)
    assert inv.solve(0) == []
    assert inv.solve(1) == [0]
    assert inv.solve(Rational(3, 2)) == [Rational(1, 2)]
    assert not inv.attains(Rational(1, 2))
    assert inv.attains(1)
    _check_acceptance(pw, inv, 0)
    _check_acceptance(pw, inv, 1)
    _check_acceptance(pw, inv, Rational(3, 2))


def test_priority_shadowing():
    # the part of the second condition with x < 1 is shadowed
    pw = Piecewise((x, x < 1), (2*x, x < 2), (3*x, True))
    inv = piecewise_inverse(pw, x)
    domains = [b.x_interval for b in inv.branches]
    assert Interval.open(-oo, 1) in domains
    assert Interval.Ropen(1, 2) in domains
    assert Interval(2, oo) in domains
    # x = 1/2 only solves through the first piece, not the shadowed ones
    assert inv.solve(Rational(1, 2)) == [Rational(1, 2)]
    # y = 3 comes from 2*x on [1, 2); 3*x = 3 gives x = 1, shadowed
    assert inv.solve(3) == [Rational(3, 2)]
    _check_acceptance(pw, inv, 3)
    _check_acceptance(pw, inv, Rational(1, 2))


def test_constant_piece_gives_interval():
    pw = Piecewise((2, (x >= 0) & (x <= 1)), (x, True))
    inv = piecewise_inverse(pw, x)
    const = [b for b in inv.branches if b.is_constant]
    assert len(const) == 1
    assert const[0].x_interval == Interval(0, 1)
    assert const[0].y_interval == FiniteSet(2)
    sols = inv.solve(2)
    assert Interval(0, 1) in sols
    assert 2 in sols
    # the constant piece only answers to its own value
    assert Interval(0, 1) not in inv.solve(1)
    assert inv.solve(Rational(1, 2)) == []
    # inverse of a constant branch is the whole domain interval
    assert const[0].inverse_expression() == Interval(0, 1)
    _check_acceptance(pw, inv, 2)


def test_quadratic_split_and_dedupe():
    pw = Piecewise((x**2, x <= 1), (x, True))
    inv = piecewise_inverse(pw, x)
    kinds = [b.kind for b in inv.branches]
    assert kinds == ['decreasing', 'increasing', 'increasing']
    # the extremum x = 0 is shared by two branches but reported once
    assert inv.solve(0) == [0]
    # one y with several x
    assert inv.solve(1) == [-1, 1]
    assert inv.solve(4) == [-2, 4]
    _check_acceptance(pw, inv, 0)
    _check_acceptance(pw, inv, 1)
    _check_acceptance(pw, inv, 4)


def test_cubic_exact_roots():
    pw = Piecewise((x**3 - 3*x, x <= 0), (x, True))
    inv = piecewise_inverse(pw, x)
    assert inv.solve(0) == [-sqrt(3), 0]
    # double root at the shared critical point x = -1 appears once
    assert inv.solve(2) == [-1, 2]
    _check_acceptance(pw, inv, 0)
    _check_acceptance(pw, inv, 2)
    _check_acceptance(pw, inv, -1)


def test_irrational_critical_points_exact():
    pw = Piecewise((x**3 - x, x <= 1), (x, True))
    inv = piecewise_inverse(pw, x)
    crit = sqrt(3)/3
    domains = [b.x_interval for b in inv.branches]
    assert Interval(-oo, -crit) in domains
    assert Interval(-crit, crit) in domains
    assert Interval(crit, 1) in domains
    mid = inv.branches[1]
    assert mid.y_interval == Interval(-2*sqrt(3)/9, 2*sqrt(3)/9)
    assert inv.solve(0) == [-1, 0, 1]
    _check_acceptance(pw, inv, 0)


def test_inverse_expression_quadratic():
    pw = Piecewise((x**2, (x >= 0) & (x <= 2)), (x, True))
    inv = piecewise_inverse(pw, x)
    quad = inv.branches[0]
    assert quad.x_interval == Interval(0, 2)
    assert quad.y_interval == Interval(0, 4)
    exprs = quad.inverse_expression(y)
    assert sqrt(y) in exprs
    # substituting a value and filtering by the domain recovers solve()
    vals = [e.subs(y, 1) for e in exprs]
    assert [v for v in vals if quad.x_interval.contains(v)] == [1]
    assert inv.solve(9) == [9]


def test_point_domain_piece():
    pw = Piecewise((x, Eq(x, 1)), (2*x, True))
    inv = piecewise_inverse(pw, x)
    point = [b for b in inv.branches if b.is_constant]
    assert len(point) == 1
    assert point[0].x_interval == FiniteSet(1)
    sols = inv.solve(1)
    assert FiniteSet(1) in sols
    assert Rational(1, 2) in sols


def test_range_property():
    pw = Piecewise((x, x < 0), (x**2, True))
    inv = piecewise_inverse(pw, x)
    assert inv.range == Interval(-oo, oo)
    pw2 = Piecewise((x, x < 0), (x + 2, True))
    inv2 = piecewise_inverse(pw2, x)
    assert inv2.range == Union(Interval.open(-oo, 0), Interval.Ropen(2, oo))


def test_invalid_input():
    raises(ValueError, lambda: piecewise_inverse(x**2, x))
    raises(ValueError, lambda: piecewise_inverse(
        Piecewise((x**4, x < 1), (x, True)), x))
    raises(ValueError, lambda: piecewise_inverse(
        Piecewise((sqrt(2)*x, x < 1), (x, True)), x))
    raises(ValueError, lambda: piecewise_inverse(
        Piecewise((x + Symbol('a'), x < 1), (x, True)), x))
    raises(ValueError, lambda: piecewise_inverse(
        Piecewise((1/x, x < 1), (x, True)), x))


def test_and_or_conditions():
    pw = Piecewise((x + 1, And(x >= 0, x < 1)), (3*x, True))
    inv = piecewise_inverse(pw, x)
    b1 = inv.branches[0]
    assert b1.x_interval == Interval.Ropen(0, 1)
    assert b1.y_interval == Interval.Ropen(1, 2)
    # y = 3/2: x = 1/2 from the first piece; 3*x = 3/2 gives x = 1/2 too,
    # but that region is shadowed, so the solution appears only once
    assert inv.solve(Rational(3, 2)) == [Rational(1, 2)]
    _check_acceptance(pw, inv, Rational(3, 2))
