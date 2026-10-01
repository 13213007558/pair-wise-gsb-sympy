"""Tests for the Newton-Puiseux algorithm in sympy.polys.puiseux_series."""

from sympy.core import I, Rational, oo, Dummy, expand
from sympy.functions import sqrt
from sympy.polys.polytools import Poly
from sympy.simplify import simplify
from sympy.testing.pytest import raises
from sympy.abc import x, y

from sympy.polys.puiseux_series import PuiseuxBranch, puiseux_series


def residual_order(F, branch):
    """Order of vanishing of F(x, y(x)) for a truncated branch y(x)."""
    e = branch.ramification
    t = Dummy('t')
    s = sum(c*t**(e*k) for k, c in branch.terms)
    r = simplify(expand(F.subs({x: t**e, y: s})))
    if r == 0:
        return oo
    return Rational(min(m[0] for m in Poly(r, t).monoms()), e)


def check_branches(F, n):
    """Check residuals, truncation and the branch count of F."""
    branches = puiseux_series(F, x, y, n)
    d = Poly(F, x, y).degree(y)
    assert sum(b.ramification for b in branches) == d
    for b in branches:
        exps = [e for e, _ in b.terms]
        assert all(e < n for e in exps)
        assert exps == sorted(exps)
        assert residual_order(F, b) >= n
    return branches


def terms_of(branch):
    return list(branch.terms)


def test_cusp():
    branches = check_branches(y**2 - x**3, 4)
    assert len(branches) == 1
    assert branches[0].ramification == 2
    assert terms_of(branches[0]) == [(Rational(3, 2), 1)]


def test_cusp_higher_order():
    # y = x**(3/2)*(1 + x)**(1/2) = x**(3/2) + x**(5/2)/2 - x**(7/2)/8 + ...
    branches = check_branches(y**2 - x**3 - x**4, 5)
    assert len(branches) == 1
    assert branches[0].ramification == 2
    assert terms_of(branches[0]) == [
        (Rational(3, 2), 1), (Rational(5, 2), Rational(1, 2)),
        (Rational(7, 2), Rational(-1, 8)), (Rational(9, 2), Rational(1, 16))]


def test_node_two_branches():
    # y = +/-x*(1 + x)**(1/2) = +/-(x + x**2/2 - x**3/8 + ...)
    branches = check_branches(y**2 - x**2 - x**3, 4)
    assert len(branches) == 2
    assert all(b.ramification == 1 for b in branches)
    assert terms_of(branches[0]) == [
        (1, 1), (2, Rational(1, 2)), (3, Rational(-1, 8))]
    assert terms_of(branches[1]) == [
        (1, -1), (2, Rational(-1, 2)), (3, Rational(1, 8))]


def test_tangent_branches():
    # Two branches y = x +/- x**2 tangent to first order.
    branches = check_branches((y - x)**2 - x**4, 4)
    assert len(branches) == 2
    assert terms_of(branches[0]) == [(1, 1), (2, 1)]
    assert terms_of(branches[1]) == [(1, 1), (2, -1)]
    assert all(residual_order((y - x)**2 - x**4, b) is oo for b in branches)


def test_multi_round_split():
    # y = x + x**2 +/- x**3: the double root of the first characteristic
    # polynomial only splits after two more rounds of the algorithm.
    F = (y - x - x**2)**2 - x**6
    branches = check_branches(F, 5)
    assert len(branches) == 2
    assert terms_of(branches[0]) == [(1, 1), (2, 1), (3, 1)]
    assert terms_of(branches[1]) == [(1, 1), (2, 1), (3, -1)]


def test_branches_not_merged_prematurely():
    # Below the splitting order the truncations coincide, but the two
    # branches must still be reported separately.
    F = (y - x - x**2)**2 - x**6
    branches = check_branches(F, 2)
    assert len(branches) == 2
    assert terms_of(branches[0]) == terms_of(branches[1]) == [(1, 1)]
    branches = check_branches(F, 3)
    assert len(branches) == 2
    assert terms_of(branches[0]) == terms_of(branches[1]) == [(1, 1), (2, 1)]


def test_double_root_fractional_split():
    # (y**2 - x**3)**2 = x**7, i.e. y = +/-x**(3/2)*(1 +/- x**(1/2))**(1/2).
    F = (y**2 - x**3)**2 - x**7
    branches = check_branches(F, 4)
    assert len(branches) == 2
    assert all(b.ramification == 2 for b in branches)
    assert terms_of(branches[0]) == [
        (Rational(3, 2), 1), (2, Rational(1, 2)),
        (Rational(5, 2), Rational(-1, 8)), (3, Rational(1, 16)),
        (Rational(7, 2), Rational(-5, 128))]
    assert terms_of(branches[1]) == [
        (Rational(3, 2), 1), (2, Rational(-1, 2)),
        (Rational(5, 2), Rational(-1, 8)), (3, Rational(-1, 16)),
        (Rational(7, 2), Rational(-5, 128))]


def test_algebraic_coefficients():
    branches = check_branches(y**2 - 2*x**3, 4)
    assert len(branches) == 1
    assert branches[0].ramification == 2
    assert terms_of(branches[0]) == [(Rational(3, 2), sqrt(2))]

    # y = +/-x*(2 + x)**(1/2) = +/-(sqrt(2)*x + sqrt(2)*x**2/4 + ...)
    branches = check_branches(y**2 - 2*x**2 - x**3, 3)
    assert len(branches) == 2
    assert all(b.ramification == 1 for b in branches)
    assert terms_of(branches[0]) == [(1, -sqrt(2)), (2, -sqrt(2)/4)]
    assert terms_of(branches[1]) == [(1, sqrt(2)), (2, sqrt(2)/4)]


def test_nested_ramification():
    # y = x**(1/2) + x**(3/4): a single branch of ramification 4.
    F = y**4 - 2*x*y**2 + x**2 - 4*x**2*y - x**3
    branches = check_branches(F, 2)
    assert len(branches) == 1
    assert branches[0].ramification == 4
    assert terms_of(branches[0]) == [(Rational(1, 2), 1), (Rational(3, 4), 1)]


def test_constant_term_branches():
    # Branches centered at y = +/-1 rather than at the origin.
    branches = check_branches(y**2 - 1 - x, 3)
    assert len(branches) == 2
    assert terms_of(branches[0]) == [
        (0, 1), (1, Rational(1, 2)), (2, Rational(-1, 8))]
    assert terms_of(branches[1]) == [
        (0, -1), (1, Rational(-1, 2)), (2, Rational(1, 8))]


def test_zero_branch():
    branches = check_branches(y*(y - x), 3)
    assert len(branches) == 2
    assert terms_of(branches[0]) == []
    assert terms_of(branches[1]) == [(1, 1)]


def test_higher_ramification():
    branches = check_branches(y**3 - x**2, 2)
    assert len(branches) == 1
    assert branches[0].ramification == 3
    assert terms_of(branches[0]) == [(Rational(2, 3), 1)]


def test_branch_count_and_residuals():
    curves = [
        y**2 - x**3,
        y**2 - x**3 - x**4,
        y**2 - x**2 - x**3,
        (y - x)**2 - x**4,
        (y - x - x**2)**2 - x**6,
        (y**2 - x**3)**2 - x**7,
        y**2 - 2*x**3,
        y**2 - 2*x**2 - x**3,
        y**3 - x**2,
        y**3 - x**4 - x**5,
        y**4 - 2*x*y**2 + x**2 - 4*x**2*y - x**3,
        y**2 - 1 - x,
        (y - 1)**2 - x**2,
        y*(y - x)*(y - x**2),
        y**2 - x**5,
    ]
    for F in curves:
        check_branches(F, 4)


def test_as_expr():
    [b] = puiseux_series(y**2 - x**3, x, y, 3)
    assert b.as_expr() == x**Rational(3, 2)
    assert b.as_expr(y) == y**Rational(3, 2)
    assert b.x is x


def test_invalid_input():
    # Not monic in y.
    raises(ValueError, lambda: puiseux_series(2*y**2 - x**3, x, y, 3))
    raises(ValueError, lambda: puiseux_series(x*y**2 - x**3, x, y, 3))
    raises(ValueError, lambda: puiseux_series(y**2 - x*y**2 - x**3, x, y, 3))

def test_nested_algebraic_coefficients():
    # y = +/-sqrt(2)*x + 2**(1/4)*x**(3/2) and the conjugate cycle
    # y = +/-sqrt(2)*x +/- 2**(1/4)*I*x**(3/2).
    F = y**4 - 4*x**2*y**2 - 8*x**4*y + 4*x**4 - 2*x**6
    branches = check_branches(F, 2)
    assert len(branches) == 2
    assert all(b.ramification == 2 for b in branches)
    assert terms_of(branches[0]) == [
        (1, -sqrt(2)), (Rational(3, 2), 2**Rational(1, 4)*I)]
    assert terms_of(branches[1]) == [
        (1, sqrt(2)), (Rational(3, 2), 2**Rational(1, 4))]

    # y = +/-sqrt(2)*x*(1 +/- x**(1/2)/2)**(1/2): coefficients stay in
    # QQ(sqrt(2)) through several rounds.
    F = (y**2 - 2*x**2)**2 - x**5
    branches = check_branches(F, 3)
    assert len(branches) == 2
    assert all(b.ramification == 2 for b in branches)
    assert terms_of(branches[0]) == [
        (1, -sqrt(2)), (Rational(3, 2), sqrt(2)/4),
        (2, sqrt(2)/32), (Rational(5, 2), sqrt(2)/128)]
    assert terms_of(branches[1]) == [
        (1, sqrt(2)), (Rational(3, 2), sqrt(2)/4),
        (2, -sqrt(2)/32), (Rational(5, 2), sqrt(2)/128)]


def test_complex_coefficients():
    # y**3 = x**3 + x**4: three branches with cube roots of unity.
    F = y**3 - x**3 - x**4
    branches = check_branches(F, 3)
    assert len(branches) == 3
    assert all(b.ramification == 1 for b in branches)
    leading = [b.terms[0][1] for b in branches]
    assert all(simplify(c**3) == 1 for c in leading)
    assert len(set(leading)) == 3

    # Not squarefree.
    raises(ValueError, lambda: puiseux_series((y - x)**2, x, y, 3))
    raises(ValueError, lambda: puiseux_series((y**2 - x**3)**2, x, y, 3))
    # Not a polynomial in x and y with rational coefficients.
    raises(ValueError, lambda: puiseux_series(y**2 - x**3 + 1/x, x, y, 3))
    raises(ValueError, lambda: puiseux_series(y**2 - sqrt(2)*x**3, x, y, 3))
    raises(ValueError, lambda: puiseux_series(y**2 - x**3 + y**-1, x, y, 3))
    # No y dependence.
    raises(ValueError, lambda: puiseux_series(x**2 - 1, x, y, 3))
    # Bad truncation order.
    raises(ValueError, lambda: puiseux_series(y**2 - x**3, x, y, 0))
    raises(ValueError, lambda: puiseux_series(y**2 - x**3, x, y, -2))
