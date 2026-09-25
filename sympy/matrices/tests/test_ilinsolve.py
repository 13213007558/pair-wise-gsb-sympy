from __future__ import annotations
from itertools import product

from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix, zeros
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.ilinsolve import DivisibilityContradiction, ilinsolve
from sympy.solvers.solveset import linsolve
from sympy.testing.pytest import raises


def _box(n, bound):
    return [Matrix(v) for v in product(range(-bound, bound + 1), repeat=n)]


def _check_lattice(A, b, xbound=2, zbound=6):
    (x0,), B = ilinsolve(A, b)
    assert isinstance(x0, Matrix)
    assert A*x0 == b
    assert A*B == zeros(A.rows, B.cols)
    exact = {tuple(x) for x in _box(A.cols, xbound) if A*x == b}
    zbox = _box(B.cols, zbound) if B.cols else [zeros(0, 1)]
    generated = {tuple(x0 + B*z) for z in zbox}
    assert all(A*Matrix(x) == b for x in generated)
    assert exact <= generated


def test_ilinsolve_square_full_rank():
    A = Matrix([[2, 1], [1, 3]])
    _check_lattice(A, Matrix([4, 7]))
    (x0,), B = ilinsolve(A, Matrix([4, 7]))
    assert B.cols == 0


def test_ilinsolve_rectangular():
    _check_lattice(Matrix([[2, 1, -1], [8, 4, -4]]), Matrix([3, 12]))
    _check_lattice(Matrix([[1, 2], [3, 1], [2, 2]]), Matrix([3, 4, 4]))
    _check_lattice(Matrix([[6, 9, 15]]), Matrix([12]))


def test_ilinsolve_rank_deficient():
    _check_lattice(Matrix([[1, 2], [2, 4]]), Matrix([3, 6]))
    _check_lattice(Matrix([[2, 4, 2], [4, 8, 4], [1, 2, 1]]),
                   Matrix([2, 4, 1]))
    _check_lattice(Matrix([[0, 0], [0, 0]]), Matrix([0, 0]))


def test_ilinsolve_zero_matrix_nullspace():
    (x0,), B = ilinsolve(zeros(2, 3), zeros(2, 1))
    assert x0 == zeros(3, 1)
    assert B.shape == (3, 3)
    assert abs(B.det()) == 1


def test_ilinsolve_rational_solution_but_no_integer_solution():
    A = Matrix([[2]])
    b = Matrix([1])
    x = symbols('x')
    assert linsolve((A, b), x) == {(Rational(1, 2),)}
    (result,), B = ilinsolve(A, b)
    assert isinstance(result, DivisibilityContradiction)
    assert result.verify(A, b)
    assert result.d == 2 and result.c == 1

    A2 = Matrix([[4, 6]])
    b2 = Matrix([3])
    (result2,), B2 = ilinsolve(A2, b2)
    assert isinstance(result2, DivisibilityContradiction)
    assert result2.verify(A2, b2)


def test_ilinsolve_inconsistent_rank_deficient():
    A = Matrix([[1, 2], [2, 4]])
    b = Matrix([1, 1])
    (result,), B = ilinsolve(A, b)
    assert isinstance(result, DivisibilityContradiction)
    assert result.verify(A, b)
    assert B.shape == (2, 1)
    assert A*B == zeros(2, 1)


def test_ilinsolve_zero_rhs_column_of_zero_matrix():
    A = zeros(2, 2)
    b = Matrix([0, 5])
    (result,), B = ilinsolve(A, b)
    assert isinstance(result, DivisibilityContradiction)
    assert result.d == 0
    assert result.verify(A, b)


def test_ilinsolve_multiple_rhs_shared_factorization():
    A = Matrix([[2, 4], [1, 3]])
    b = Matrix([[6, 3, 2], [4, 1, 1]])
    sols, B = ilinsolve(A, b)
    assert len(sols) == 3
    assert isinstance(sols[0], Matrix)
    assert A*sols[0] == b[:, 0]
    assert isinstance(sols[1], DivisibilityContradiction)
    assert sols[1].verify(A, b[:, 1])
    assert isinstance(sols[2], Matrix)
    assert A*sols[2] == b[:, 2]
    assert A*B == zeros(2, B.cols)


def test_ilinsolve_certificate_soundness():
    A = Matrix([[2, 4], [1, 3]])
    b = Matrix([3, 1])
    (bad,), _ = ilinsolve(A, b)
    assert bad.verify(A, b)
    assert not bad.verify(A, Matrix([4, 1]))
    good = DivisibilityContradiction(Matrix([[0, 1]]), 1, 1)
    assert not good.verify(A, b)


def test_ilinsolve_noninteger_entries_rejected():
    raises(ValueError, lambda: ilinsolve(Matrix([[Rational(1, 2)]]),
                                         Matrix([1])))
    raises(ValueError, lambda: ilinsolve(Matrix([[1]]),
                                         Matrix([[symbols('x')]])))
    raises(ShapeError, lambda: ilinsolve(Matrix([[1, 2]]), Matrix([1, 2])))


def test_ilinsolve_random_small():
    import random
    rng = random.Random(1234)
    for _ in range(20):
        m = rng.randint(1, 3)
        n = rng.randint(1, 3)
        A = Matrix(m, n, lambda i, j: rng.randint(-3, 3))
        x_true = Matrix(n, 1, lambda i, j: rng.randint(-2, 2))
        _check_lattice(A, A*x_true)
