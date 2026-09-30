from __future__ import annotations

from sympy.core.symbol import symbols
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.exponential import exp
from sympy.liealgebras import baker_campbell_hausdorff
from sympy.matrices import Matrix
from sympy.physics.quantum import Commutator
from sympy.testing.pytest import raises

x, y = symbols("x y", commutative=False)


def test_baker_campbell_hausdorff_low_orders():
    c = Commutator(x, y)
    assert baker_campbell_hausdorff(x, y, order=1) == x + y
    assert baker_campbell_hausdorff(x, y, order=2) == x + y + c/2
    assert baker_campbell_hausdorff(x, y, order=3) == (
        x + y + c/2 + Commutator(x, c)/12 - Commutator(y, c)/12)
    assert baker_campbell_hausdorff(x, y, order=4) == (
        x + y + c/2 + Commutator(x, c)/12 - Commutator(y, c)/12
        - Commutator(y, Commutator(x, c))/24)


def test_baker_campbell_hausdorff_truncation_is_stable():
    order4 = baker_campbell_hausdorff(x, y, order=4)
    order5 = baker_campbell_hausdorff(x, y, order=5)
    difference = (order5 - order4).expand()
    assert difference != 0
    assert difference.subs({x: 0, y: 0}) == 0
    assert (difference.subs(x, 0) == 0 and
            difference.subs(y, 0) == 0)


def test_baker_campbell_hausdorff_commuting_inputs():
    a, b = symbols("a b")
    assert baker_campbell_hausdorff(a, b, order=8) == a + b
    assert baker_campbell_hausdorff(x, x, order=8) == 2*x


def test_baker_campbell_hausdorff_invalid_arguments():
    raises(ValueError, lambda: baker_campbell_hausdorff(x, y, order=0))
    raises(ValueError, lambda: baker_campbell_hausdorff(
        x, y, order=9, max_order=8))
    raises(ValueError, lambda: baker_campbell_hausdorff(
        x, y, order="x", max_order=8))


def test_baker_campbell_hausdorff_nilpotent_matrices():
    left = Matrix([
        [0, 1, 2, 3],
        [0, 0, 4, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 0],
    ])
    right = Matrix([
        [0, 7, 8, 9],
        [0, 0, 10, 11],
        [0, 0, 0, 12],
        [0, 0, 0, 0],
    ])
    assert (left**4).is_zero_matrix
    assert (right**4).is_zero_matrix

    approximation = baker_campbell_hausdorff(x, y, order=3).subs({
        x: left,
        y: right,
    })
    approximation = bottom_up(
        approximation,
        lambda value: value.doit() if isinstance(value, Commutator) else value)
    assert (exp(approximation) - exp(left)*exp(right)).is_zero_matrix


def test_lyndon_basis_uses_jacobi_relation():
    c = Commutator(x, y)
    expansion = baker_campbell_hausdorff(x, y, order=4)

    assert expansion.has(Commutator(y, Commutator(x, c)))
    assert not expansion.has(Commutator(x, Commutator(y, c)))
    assert (Commutator(x, Commutator(y, c)) -
            Commutator(y, Commutator(x, c))).doit().expand() == 0


def test_equivalent_nested_commutators_are_not_duplicated():
    c = Commutator(x, y)
    order5 = baker_campbell_hausdorff(x, y, order=5)
    assert len(order5.atoms(Commutator)) == len(set(order5.atoms(Commutator)))
    assert not order5.has(Commutator(c, c))
    assert not order5.has(Commutator(c, Commutator(c, y)))
