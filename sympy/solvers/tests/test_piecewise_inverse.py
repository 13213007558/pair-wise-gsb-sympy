from __future__ import annotations
from sympy import (And, Eq, FiniteSet, Interval, Piecewise, Rational, S,
    Ne, Symbol, Union, oo, simplify, sin, solveset, sqrt)
from sympy.sets.sets import Intersection
from sympy.solvers.piecewise_inverse import (
    InverseBranch, PiecewiseInverse, piecewise_inverse)
from sympy.testing.pytest import raises

x = Symbol('x')
y = Symbol('y')


def _points(s):
    if isinstance(s, FiniteSet):
        return list(s.args)
    if isinstance(s, Interval):
        mid = [(s.inf + s.sup)/2] if s.inf.is_finite and s.sup.is_finite else []
        ends = [e for e, open_ in ((s.inf, s.left_open), (s.sup, s.right_open))
                if e.is_finite and not open_]
        return ends + mid
    out = []
    for arg in s.args:
        out.extend(_points(arg))
    return out


def _assert_forward(pw, var, xv, y0):
    val = pw.subs(var, xv)
    if simplify(val - y0) != 0:
        assert abs(complex((val - y0).evalf(40))) < 1e-30


def _assert_same_solution_set(got, expected):
    def split(s):
        points, others = [], []
        atoms = s.args if isinstance(s, Union) else (s,)
        for atom in atoms:
            if isinstance(atom, FiniteSet):
                points.extend(atom.args)
            elif isinstance(atom, Intersection):
                finite = [a for a in atom.args if isinstance(a, FiniteSet)]
                if len(finite) == 1:
                    points.extend(finite[0].args)
                else:
                    others.append(atom)
            elif atom is not S.EmptySet:
                others.append(atom)
        return points, others
    g_pts, g_other = split(got)
    e_pts, e_other = split(expected)
    ga = sorted((complex(v.evalf(40)) for v in g_pts), key=lambda c: (c.real, c.imag))
    ea = sorted((complex(v.evalf(40)) for v in e_pts), key=lambda c: (c.real, c.imag))
    assert len(ga) == len(ea)
    for g, e in zip(ga, ea):
        assert abs(g - e) < 1e-30
    assert sorted(map(str, g_other)) == sorted(map(str, e_other))


def _check_query(rel, y0):
    """Forward substitution, domain restriction and completeness."""
    pre = rel.preimage(y0)
    pw, var = rel.piecewise, rel.var
    for pt in _points(pre):
        _assert_forward(pw, var, pt, y0)
    expected = solveset(Eq(pw, y0), var, S.Reals)
    _assert_same_solution_set(pre, expected)
    return pre


def _check_inverse_expressions(rel, samples):
    for branch in rel.branches:
        if branch.is_constant:
            continue
        for y0 in samples:
            if not branch.range.contains(y0):
                continue
            xv = branch.inverse.subs(rel.yvar, y0)
            val = branch.expr.subs(branch.var, xv)
            if simplify(val - y0) != 0:
                assert abs(complex((val - y0).evalf(50))) < 1e-35
            lo = branch.domain.inf
            hi = branch.domain.sup
            cx = complex(xv.evalf(50))
            assert abs(cx.imag) < 1e-35
            if lo is not S.NegativeInfinity:
                assert cx.real > float(lo.evalf(50)) - 1e-30
            if hi is not S.Infinity:
                assert cx.real < float(hi.evalf(50)) + 1e-30


def test_branch_structure_basic():
    pw = Piecewise((x + 1, x < 0), (x**2, And(x >= 0, x <= 2)), (8, x > 2))
    rel = piecewise_inverse(pw, x)
    assert isinstance(rel, PiecewiseInverse)
    assert rel.yvar is y or rel.yvar == Symbol('y')
    b1, b2, b3 = rel.branches
    assert b1.domain == Interval.open(-oo, 0)
    assert b1.direction == 'increasing'
    assert b1.range == Interval.open(-oo, 1)
    assert b1.inverse == rel.yvar - 1
    assert b2.domain == Interval(0, 2)
    assert b2.direction == 'increasing'
    assert b2.range == Interval(0, 4)
    assert b2.inverse == sqrt(rel.yvar)
    assert b3.is_constant
    assert b3.range == FiniteSet(8)
    assert b3.inverse == Interval.open(2, oo)
    assert rel.range == Union(Interval(-oo, 4), FiniteSet(8))


def test_preimage_basic():
    pw = Piecewise((x + 1, x < 0), (x**2, And(x >= 0, x <= 2)), (8, x > 2))
    rel = piecewise_inverse(pw, x)
    assert rel.preimage(1) == FiniteSet(1)
    assert rel.preimage(0) == FiniteSet(-1, 0)
    assert rel.preimage(Rational(1, 2)) == FiniteSet(Rational(-1, 2), sqrt(2)/2)
    assert rel.preimage(4) == FiniteSet(2)
    assert rel.preimage(-3) == FiniteSet(-4)
    assert rel.preimage(5) == S.EmptySet
    assert rel.preimage(8) == Interval.open(2, oo)


def test_first_match_priority():
    pw = Piecewise((x, x < 2), (2*x, And(x >= 1, x < 3)), (x**2, x >= 2))
    rel = piecewise_inverse(pw, x)
    domains = [b.domain for b in rel.branches]
    assert Interval.open(-oo, 2) in domains
    assert Interval.Ropen(2, 3) in domains
    assert Interval(3, oo) in domains
    # x = 3/2 is covered by the first branch only
    assert rel.preimage(Rational(3, 2)) == FiniteSet(Rational(3, 2))
    # the shadowed part [1, 2) of the second branch does not take part
    assert rel.preimage(3) == S.EmptySet
    assert rel.preimage(5) == FiniteSet(Rational(5, 2))
    assert rel.preimage(9) == FiniteSet(3)


def test_constant_branch_gives_interval():
    pw = Piecewise((x, x < 0), (3, And(x >= 0, x <= 1)), (x + 2, x > 1))
    rel = piecewise_inverse(pw, x)
    pre = rel.preimage(3)
    assert pre == Interval(0, 1)
    assert rel.preimage(Rational(7, 2)) == FiniteSet(Rational(3, 2))
    assert rel.preimage(Rational(5, 2)) == S.EmptySet
    assert rel.preimage(-1) == FiniteSet(-1)


def test_cubic_three_monotonic_branches():
    rel = piecewise_inverse(Piecewise((x**3 - 3*x, True)), x)
    assert len(rel.branches) == 3
    b1, b2, b3 = rel.branches
    assert b1.domain == Interval(-oo, -1)
    assert b1.range == Interval(-oo, 2)
    assert b1.direction == 'increasing'
    assert b2.domain == Interval(-1, 1)
    assert b2.range == Interval(-2, 2)
    assert b2.direction == 'decreasing'
    assert b3.domain == Interval(1, oo)
    assert b3.range == Interval(-2, oo)
    assert b3.direction == 'increasing'
    assert rel.preimage(0) == FiniteSet(-sqrt(3), 0, sqrt(3))
    # critical values: the shared endpoint appears only once
    assert rel.preimage(2) == FiniteSet(-1, 2)
    assert rel.preimage(-2) == FiniteSet(-2, 1)
    assert len(rel.preimage(5)) == 1
    assert len(rel.preimage(-5)) == 1


def test_stationary_point_splits_branches():
    rel = piecewise_inverse(x**3, x)
    assert len(rel.branches) == 2
    b1, b2 = rel.branches
    assert b1.domain == Interval(-oo, 0)
    assert b1.range == Interval(-oo, 0)
    assert b2.domain == Interval(0, oo)
    assert b2.range == Interval(0, oo)
    assert rel.preimage(0) == FiniteSet(0)
    assert rel.preimage(-8) == FiniteSet(-2)
    assert rel.preimage(8) == FiniteSet(2)
    # the inverse expressions are valid on their whole ranges
    for branch, y0 in ((b1, -8), (b1, -1), (b2, 1), (b2, 8)):
        xv = branch.inverse.subs(rel.yvar, y0)
        assert simplify(branch.expr.subs(branch.var, xv) - y0) == 0


def test_algebraic_critical_points_stay_exact():
    rel = piecewise_inverse(x**3 - x, x)
    c = sqrt(3)/3
    assert [b.domain for b in rel.branches] == [
        Interval(-oo, -c), Interval(-c, c), Interval(c, oo)]
    v = 2*sqrt(3)/9
    assert rel.branches[0].range == Interval(-oo, v)
    assert rel.branches[1].range == Interval(-v, v)
    assert rel.branches[2].range == Interval(-v, oo)
    # algebraic query value: double root at the critical point, listed once
    assert rel.preimage(-v) == FiniteSet(-2*c, c)
    assert rel.preimage(v) == FiniteSet(-c, 2*c)


def test_endpoint_openness():
    pw = Piecewise((x**2, And(x > 0, x <= 2)), (x, x > 2))
    rel = piecewise_inverse(pw, x)
    b1, b2 = rel.branches
    assert b1.domain == Interval.Lopen(0, 2)
    assert b1.range == Interval.Lopen(0, 4)
    assert b2.domain == Interval.open(2, oo)
    assert b2.range == Interval.open(2, oo)
    assert rel.preimage(0) == S.EmptySet
    assert rel.preimage(4) == FiniteSet(2, 4)
    assert rel.preimage(2) == FiniteSet(sqrt(2))


def test_shared_endpoint_not_duplicated():
    pw = Piecewise((x**2, x <= 1), (2 - x, x > 1))
    rel = piecewise_inverse(pw, x)
    pre = rel.preimage(1)
    assert pre == FiniteSet(-1, 1)
    assert len(pre.args) == 2


def test_isolated_point_domain():
    pw = Piecewise((x**2, Eq(x, 2)), (x, True))
    rel = piecewise_inverse(pw, x)
    assert rel.preimage(4) == FiniteSet(2, 4)
    # x = 2 is excluded from the second branch by priority
    assert rel.preimage(2) == S.EmptySet


def test_plain_expression_input():
    rel = piecewise_inverse(x**2, x)
    assert len(rel.branches) == 2
    assert rel.preimage(4) == FiniteSet(-2, 2)


def test_custom_yvar():
    t = Symbol('t')
    rel = piecewise_inverse(x**2, x, yvar=t)
    assert rel.branches[1].inverse == sqrt(t)


def test_forward_and_completeness():
    pws = [
        Piecewise((x + 1, x < 0), (x**2, And(x >= 0, x <= 2)), (8, x > 2)),
        Piecewise((x, x < 2), (2*x, And(x >= 1, x < 3)), (x**2, x >= 2)),
        Piecewise((x**3 - 3*x, True)),
        Piecewise((x, x < 0), (3, And(x >= 0, x <= 1)), (x + 2, x > 1)),
        Piecewise((x**2, x <= 1), (2 - x, x > 1)),
        Piecewise((x**3 - x, And(x >= -2, x < 2)), (5, x >= 2)),
    ]
    ys = [Rational(-3), Rational(-2), Rational(-1), Rational(-1, 2),
          Rational(0), Rational(1, 3), Rational(1), Rational(2),
          Rational(3), Rational(5), Rational(8)]
    for pw in pws:
        rel = piecewise_inverse(pw, x)
        for y0 in ys:
            _check_query(rel, y0)
        _check_inverse_expressions(rel, [Rational(-3, 2), Rational(-1, 3),
                                         Rational(1, 4), Rational(3, 2),
                                         Rational(7, 2)])


def test_errors():
    raises(ValueError, lambda: piecewise_inverse(x**4, x))
    raises(ValueError, lambda: piecewise_inverse(sin(x), x))
    raises(ValueError, lambda: piecewise_inverse(sqrt(2)*x, x))
    raises(ValueError, lambda: piecewise_inverse(x + y, x))
    raises(ValueError, lambda: piecewise_inverse(
        Piecewise((x, sin(x) > 0), (2*x, True)), x))
    rel = piecewise_inverse(x, x)
    raises(ValueError, lambda: rel.preimage(y))


def test_edge_cases():
    # fully shadowed branch is skipped
    rel = piecewise_inverse(Piecewise((x, True), (2*x, True)), x)
    assert len(rel.branches) == 1
    assert rel.preimage(3) == FiniteSet(3)
    # negative leading coefficient
    rel = piecewise_inverse(Piecewise((-x**2, And(x >= 1, x <= 3))), x)
    b = rel.branches[0]
    assert b.direction == 'decreasing'
    assert b.range == Interval(-9, -1)
    assert b.inverse == sqrt(-rel.yvar)
    assert rel.preimage(-4) == FiniteSet(2)
    # cubic with no real critical points: a single branch
    rel = piecewise_inverse(x**3 + x, x)
    assert len(rel.branches) == 1
    assert rel.branches[0].domain == Interval(-oo, oo)
    assert rel.preimage(2) == FiniteSet(1)
    assert rel.preimage(-10) == FiniteSet(-2)
    # conditions may exclude isolated points
    rel = piecewise_inverse(Piecewise((x**2, Ne(x, 0))), x)
    assert [b.domain for b in rel.branches] == [
        Interval.open(-oo, 0), Interval.open(0, oo)]
    assert rel.preimage(4) == FiniteSet(-2, 2)
    assert rel.preimage(0) == S.EmptySet


def test_branch_repr():
    rel = piecewise_inverse(x + 1, x)
    assert isinstance(rel.branches[0], InverseBranch)
    assert 'InverseBranch' in repr(rel.branches[0])
    assert 'PiecewiseInverse' in repr(rel)
