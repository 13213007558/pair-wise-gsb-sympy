from sympy.core.numbers import Rational, oo, I
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.polys.newtonpuiseux import (
    PuiseuxBranch, newton_puiseux, residual_order)
from sympy.testing.pytest import raises

x, y = symbols('x y')


def check_branches(F, order, count):
    branches = newton_puiseux(F, x, y, order)
    assert len(branches) == count
    for b in branches:
        assert residual_order(F, x, y, b) >= order
    return branches


def series_set(branches):
    return set(b.series(x) for b in branches)


def test_cusp():
    branches = check_branches(y**2 - x**3, 3, 2)
    assert series_set(branches) == {x**Rational(3, 2), -x**Rational(3, 2)}
    assert [b.ramification for b in branches] == [2, 2]
    assert residual_order(y**2 - x**3, x, y, branches[0]) is oo


def test_tangent_branches():
    F = (y - x)*(y - x - x**2)
    branches = check_branches(F, 3, 2)
    assert series_set(branches) == {x, x + x**2}
    assert all(b.ramification == 1 for b in branches)


def test_branches_splitting_late():
    # The two branches agree up to order 3 but must not be merged.
    F = (y - x)*(y - x - x**3)
    branches = check_branches(F, 2, 2)
    assert [b.terms for b in branches] == [((1, 1),), ((1, 1),)]
    branches = check_branches(F, 4, 2)
    assert series_set(branches) == {x, x + x**3}


def test_multi_round_splitting():
    # Four branches: y = +/-x +/- x**(7/2)/2 + ...  The first round of
    # the Newton polygon only separates the sign of the x term.
    F = (y**2 - x**2)**2 - x**9
    branches = check_branches(F, 3, 4)
    assert series_set(branches) == {x, -x}
    branches = check_branches(F, 4, 4)
    assert series_set(branches) == {
        x + x**Rational(7, 2)/2, x - x**Rational(7, 2)/2,
        -x + x**Rational(7, 2)/2, -x - x**Rational(7, 2)/2}
    assert all(b.ramification == 2 for b in branches)


def test_algebraic_coefficients():
    branches = check_branches(y**2 - 2*x**3, 3, 2)
    assert series_set(branches) == {
        sqrt(2)*x**Rational(3, 2), -sqrt(2)*x**Rational(3, 2)}
    assert all(b.ramification == 2 for b in branches)


def test_algebraic_coefficients_multi_round():
    # y = +/-sqrt(2)*x**(3/2) +/- sqrt(2)/4*x**2 + ...: the branches
    # split in the second round inside the field QQ(sqrt(2)).
    F = (y**2 - 2*x**3)**2 - x**7
    branches = check_branches(F, 3, 4)
    assert all(b.ramification == 2 for b in branches)
    assert set(b.terms[0] for b in branches) == {
        (Rational(3, 2), sqrt(2)), (Rational(3, 2), -sqrt(2))}
    assert set(b.terms[1] for b in branches) == {
        (2, sqrt(2)/4), (2, -sqrt(2)/4)}


def test_ramified_algebraic_centre():
    # Above each of the centres +/-sqrt(2) there are two branches with
    # ramification 2.
    F = (y**2 - 2)**2 - x**3
    branches = check_branches(F, 3, 4)
    assert all(b.ramification == 2 for b in branches)
    assert set(b.terms[0] for b in branches) == {
        (0, sqrt(2)), (0, -sqrt(2))}
    assert set(b.terms[1] for b in branches) == {
        (Rational(3, 2), sqrt(2)/4), (Rational(3, 2), -sqrt(2)/4)}


def test_complex_coefficients():
    branches = check_branches(y**3 - x**2, 3, 3)
    assert all(b.ramification == 3 for b in branches)
    assert set(b.terms[0][1] for b in branches) == {
        1, (-1 - sqrt(3)*I)/2, (-1 + sqrt(3)*I)/2}
    assert all(b.terms[0][0] == Rational(2, 3) for b in branches)


def test_ramification_four():
    branches = check_branches(y**4 - x**5, 2, 4)
    assert all(b.ramification == 4 for b in branches)
    assert set(b.terms[0][1] for b in branches) == {1, -1, I, -I}
    assert all(b.terms[0][0] == Rational(5, 4) for b in branches)


def test_nonzero_centres():
    F = (y**2 - 2)*(y - x)
    branches = check_branches(F, 2, 3)
    assert series_set(branches) == {sqrt(2), -sqrt(2), x}
    assert all(b.ramification == 1 for b in branches)


def test_exact_zero_branch():
    branches = check_branches(y*(y - x), 3, 2)
    assert series_set(branches) == {0, x}


def test_splitting_after_first_term():
    # y = x**2 +/- x**(5/2): the two branches split only in the second round.
    F = (y - x**2)**2 - x**5
    branches = check_branches(F, 4, 2)
    assert series_set(branches) == {
        x**2 + x**Rational(5, 2), x**2 - x**Rational(5, 2)}
    assert all(b.ramification == 2 for b in branches)


def test_poly_input():
    F = Poly(y**2 - x**3, x, y)
    branches = newton_puiseux(F, x, y, 3)
    assert len(branches) == 2


def test_residual_order():
    F = y**2 - x**3 - x**5
    b = newton_puiseux(F, x, y, 3)[0]
    assert residual_order(F, x, y, b) >= 3


def test_invalid_inputs():
    raises(ValueError, lambda: newton_puiseux(2*y**2 - x**3, x, y, 3))
    raises(ValueError, lambda: newton_puiseux((y - x)**2, x, y, 3))
    raises(ValueError, lambda: newton_puiseux(y**2 - sqrt(2)*x, x, y, 3))
    raises(ValueError, lambda: newton_puiseux(y**2 - x**3, x, y, 0))
    raises(ValueError, lambda: newton_puiseux(y**2 - x**3, x, y, -1))


def test_branch_equality():
    b1 = PuiseuxBranch([(Rational(3, 2), 1)], 2)
    b2 = PuiseuxBranch([(Rational(3, 2), 1)], 2)
    assert b1 == b2
    assert hash(b1) == hash(b2)
    assert b1 != PuiseuxBranch([(Rational(3, 2), -1)], 2)
    assert repr(b1) == 'PuiseuxBranch(terms=((3/2, 1),), ramification=2)'
