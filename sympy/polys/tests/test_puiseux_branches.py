"""Tests for local Puiseux branch expansion."""

from __future__ import annotations

from sympy import Add, Float, I, Rational, expand, sqrt
from sympy.abc import x, y, X
from sympy.polys.puiseux_branches import VerticalComponent, puiseux_branches
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import raises


def terms(branch):
    return {e: c for c, e in branch.terms}


def subs_residual(F, branch, x0, y0):
    """F(x, S(x)) for the truncated series S of the branch."""
    S = y0 + sum(c*X**e for c, e in branch.terms)
    return expand(F.subs({x: x0 + X, y: S}))


def valuation(e):
    """Lowest exponent of X in an expanded expression, None if zero."""
    if e == 0:
        return None
    return min(t.as_coeff_exponent(X)[1] for t in Add.make_args(e))


def check_residual(F, branch, x0, y0):
    """Check that substituting back kills all terms below mult*rem."""
    r = subs_residual(F, branch, x0, y0)
    if branch.rem is None:
        assert r == 0
    else:
        assert valuation(r) is not None
        assert valuation(r) >= branch.mult*branch.rem


def test_cusp():
    F = y**2 - x**3
    branches, vertical = puiseux_branches(F, x, y, 0, 0, 3)
    assert vertical == ()
    assert len(branches) == 2
    for b in branches:
        assert b.mult == 1
        assert b.rem is None
        assert list(terms(b)) == [Rational(3, 2)]
    assert {terms(b)[Rational(3, 2)] for b in branches} == {1, -1}
    for b in branches:
        check_residual(F, b, 0, 0)


def test_tangent_branches_split():
    F = (y - x)**2 - x**3
    branches, _ = puiseux_branches(F, x, y, 0, 0, 3)
    assert len(branches) == 2
    for b in branches:
        assert b.mult == 1 and b.rem is None
        assert terms(b)[1] == 1
    assert {terms(b)[Rational(3, 2)] for b in branches} == {1, -1}
    for b in branches:
        check_residual(F, b, 0, 0)


def test_tangent_branches_clustered():
    F = (y - x)**2 - x**6
    branches, _ = puiseux_branches(F, x, y, 0, 0, 2)
    assert len(branches) == 1
    b = branches[0]
    assert b.terms == ((1, 1),)
    assert b.rem == 3
    assert b.mult == 2
    check_residual(F, b, 0, 0)
    branches, _ = puiseux_branches(F, x, y, 0, 0, 4)
    assert len(branches) == 2
    for b in branches:
        assert b.mult == 1 and b.rem is None
        assert terms(b)[1] == 1
    assert {terms(b)[3] for b in branches} == {1, -1}
    for b in branches:
        check_residual(F, b, 0, 0)


def test_nonzero_point():
    F = y**2 - x**3
    branches, vertical = puiseux_branches(F, x, y, 1, 1, 2)
    assert vertical == ()
    assert len(branches) == 1
    b = branches[0]
    assert b.terms == ((Rational(3, 2), 1),)
    assert b.rem == 2
    assert b.mult == 1
    check_residual(F, b, 1, 1)
    branches, _ = puiseux_branches(F, x, y, 1, -1, 2)
    assert branches[0].terms == ((Rational(-3, 2), 1),)
    check_residual(F, branches[0], 1, -1)


def test_vertical_component():
    F = x*(y - x**2)
    branches, vertical = puiseux_branches(F, x, y, 0, 0, 3)
    assert vertical == (VerticalComponent(0, 1),)
    assert len(branches) == 1
    assert branches[0].terms == ((1, 2),)
    assert branches[0].rem is None
    check_residual(F, branches[0], 0, 0)

    F = (x - 1)**2*y
    branches, vertical = puiseux_branches(F, x, y, 1, 0, 3)
    assert vertical == (VerticalComponent(1, 2),)
    assert len(branches) == 1
    assert branches[0].terms == ()
    assert branches[0].mult == 1
    check_residual(F, branches[0], 1, 0)


def test_purely_vertical():
    branches, vertical = puiseux_branches(x**2, x, y, 0, 0, 3)
    assert branches == ()
    assert vertical == (VerticalComponent(0, 2),)


def test_algebraic_conjugates_not_merged():
    F = y**2 - 2*x**2
    branches, _ = puiseux_branches(F, x, y, 0, 0, 2)
    assert len(branches) == 2
    coeffs = {terms(b)[1] for b in branches}
    assert coeffs == {sqrt(2), -sqrt(2)}
    for b in branches:
        assert b.rem is None and b.mult == 1
        assert not terms(b)[1].has(Float)
        check_residual(F, b, 0, 0)

    F = y**2 + x**2
    branches, _ = puiseux_branches(F, x, y, 0, 0, 2)
    assert {terms(b)[1] for b in branches} == {I, -I}
    for b in branches:
        check_residual(F, b, 0, 0)


def test_repeated_factor_multiplicity():
    F = (y**2 - x**3)**2
    branches, _ = puiseux_branches(F, x, y, 0, 0, 3)
    assert len(branches) == 2
    for b in branches:
        assert b.mult == 2
        assert b.rem is None
        check_residual(F, b, 0, 0)
    assert {terms(b)[Rational(3, 2)] for b in branches} == {1, -1}


def test_remainder_order_semantics():
    F = y**2 - x**3 - x**5
    branches, _ = puiseux_branches(F, x, y, 0, 0, 4)
    assert len(branches) == 2
    for b in branches:
        assert list(terms(b)) == [Rational(3, 2), Rational(7, 2)]
        assert b.rem == Rational(11, 2)
        assert b.mult == 1
        check_residual(F, b, 0, 0)
    b = [b for b in branches if terms(b)[Rational(3, 2)] == 1][0]
    assert terms(b)[Rational(7, 2)] == Rational(1, 2)
    assert subs_residual(F, b, 0, 0) == X**7/4


def test_crootof_coefficients():
    F = y**5 - x**4*y - x**5
    branches, _ = puiseux_branches(F, x, y, 0, 0, 2)
    assert len(branches) == 5
    for b in branches:
        assert b.mult == 1 and b.rem is None
        c = terms(b)[1]
        assert isinstance(c, CRootOf)
        assert c.poly.as_expr().expand() == c.poly.gen**5 - c.poly.gen - 1


def test_horizontal_branch():
    branches, _ = puiseux_branches(y*(y - x), x, y, 0, 0, 3)
    assert len(branches) == 2
    exact = [b for b in branches if b.terms == ()]
    assert len(exact) == 1
    assert exact[0].rem is None and exact[0].mult == 1


def test_point_not_on_curve():
    branches, vertical = puiseux_branches(y**2 - x**3, x, y, 1, 0, 2)
    assert branches == ()
    assert vertical == ()


def test_errors():
    raises(ValueError, lambda: puiseux_branches(sqrt(2)*y - x, x, y, 0, 0, 2))
    raises(ValueError, lambda: puiseux_branches(y - x, x, y, sqrt(2), 0, 2))
    raises(ValueError, lambda: puiseux_branches(y - x, x, y, 0, 0, 0))
    raises(ValueError, lambda: puiseux_branches(0, x, y, 0, 0, 2))


def test_as_expr():
    branches, _ = puiseux_branches(y**2 - x**3 - x**5, x, y, 0, 0, 4)
    b = [b for b in branches if terms(b)[Rational(3, 2)] == 1][0]
    from sympy import O
    assert b.as_expr(x) == x**Rational(3, 2) + x**Rational(7, 2)/2 + O(x**Rational(11, 2))
