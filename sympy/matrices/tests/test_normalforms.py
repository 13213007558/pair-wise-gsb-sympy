from __future__ import annotations
from sympy.testing.pytest import warns_deprecated_sympy

from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly
from sympy.matrices import Matrix, randMatrix
from sympy.matrices.normalforms import (
    invariant_factors,
    smith_normal_form,
    smith_normal_decomp,
    hermite_normal_form,
    is_smith_normal_form,
    integer_linsolve,
    IntegerSolution,
    DivisibilityContradiction,
)
from sympy.matrices.exceptions import ShapeError
from sympy.polys.domains import ZZ, QQ
from sympy.core.numbers import Integer, Rational
from sympy.solvers.solveset import linsolve
from sympy.testing.pytest import raises

import random
from itertools import product


def test_smith_normal():
    m = Matrix([[12,6,4,8],[3,9,6,12],[2,16,14,28],[20,10,10,20]])
    smf = Matrix([[1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 30, 0], [0, 0, 0, 0]])
    assert smith_normal_form(m) == smf

    a, s, t = smith_normal_decomp(m)
    assert a == s * m * t

    x = Symbol('x')
    with warns_deprecated_sympy():
        m = Matrix([[Poly(x-1), Poly(1, x),Poly(-1,x)],
                    [0, Poly(x), Poly(-1,x)],
                    [Poly(0,x),Poly(-1,x),Poly(x)]])
    invs = 1, x - 1, x**2 - 1
    assert invariant_factors(m, domain=QQ[x]) == invs

    m = Matrix([[2, 4]])
    smf = Matrix([[2, 0]])
    assert smith_normal_form(m) == smf

    prng = random.Random(0)
    for i in range(6):
        for j in range(6):
            for _ in range(10 if i*j else 1):
                m = randMatrix(i, j, max=5, percent=50, prng=prng)
                a, s, t = smith_normal_decomp(m)
                assert a == s * m * t
                assert is_smith_normal_form(a)
                s.inv().to_DM(ZZ)
                t.inv().to_DM(ZZ)

                a, s, t = smith_normal_decomp(m, QQ)
                assert a == s * m * t
                assert is_smith_normal_form(a)
                s.inv()
                t.inv()


def test_hermite_normal():
    m = Matrix([[2, 7, 17, 29, 41], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]])
    hnf = Matrix([[1, 0, 0], [0, 2, 1], [0, 0, 1]])
    assert hermite_normal_form(m) == hnf

    tr_hnf = Matrix([[37, 0, 19], [222, -6, 113], [48, 0, 25], [0, 2, 1], [0, 0, 1]])
    assert hermite_normal_form(m.transpose()) == tr_hnf

    m = Matrix([[8, 28, 68, 116, 164], [3, 11, 19, 31, 43], [5, 13, 23, 37, 47]])
    hnf = Matrix([[4, 0, 0], [0, 2, 1], [0, 0, 1]])
    assert hermite_normal_form(m) == hnf
    assert hermite_normal_form(m, D=8) == hnf
    assert hermite_normal_form(m, D=ZZ(8)) == hnf
    assert hermite_normal_form(m, D=Integer(8)) == hnf

    m = Matrix([[10, 8, 6, 30, 2], [45, 36, 27, 18, 9], [5, 4, 3, 2, 1]])
    hnf = Matrix([[26, 2], [0, 9], [0, 1]])
    assert hermite_normal_form(m) == hnf

    m = Matrix([[2, 7], [0, 0], [0, 0]])
    hnf = Matrix([[1], [0], [0]])
    assert hermite_normal_form(m) == hnf


def test_issue_23410():
    A = Matrix([[1, 12], [0, 8], [0, 5]])
    H = Matrix([[1, 0], [0, 8], [0, 5]])
    assert hermite_normal_form(A) == H


def _in_integer_span(B, v):
    # True if v is in the integer column span of B.
    if B.cols == 0:
        return v.is_zero_matrix
    try:
        z, _ = B.gauss_jordan_solve(v)
    except ValueError:
        return False
    return all(entry.is_integer for entry in z)


def _check_contradiction_certificate(A, b, cert):
    # A DivisibilityContradiction must be independently verifiable.
    D, S, T = smith_normal_decomp(A, domain=ZZ)
    assert cert.transform == S
    c = cert.transform * b
    assert c[cert.row] == cert.residual
    m, n = A.shape
    divisor = D[cert.row, cert.row] if cert.row < min(m, n) else Integer(0)
    assert cert.divisor == divisor
    if cert.divisor == 0:
        assert cert.residual != 0
    else:
        assert cert.residual % cert.divisor != 0


def test_integer_linsolve_enumeration():
    # For small systems, enumerate all integer solutions in a box and
    # check both inclusions against the returned solution lattice.
    systems = [
        (Matrix([[2, 4], [3, 6]]), Matrix([2, 3])),
        (Matrix([[2, 4], [3, 6]]), Matrix([2, 4])),
        (Matrix([[1, 2, 3], [4, 5, 6]]), Matrix([1, 1])),
        (Matrix([[6, 3], [4, 2], [2, 1]]), Matrix([3, 2, 1])),
        (Matrix([[2, 0], [0, 3]]), Matrix([1, 1])),
        (Matrix([[2, 0], [0, 3]]), Matrix([4, 9])),
        (Matrix([[1, 2], [3, 4], [5, 6]]), Matrix([1, 1, 1])),
        (Matrix.zeros(2, 3), Matrix([0, 0])),
        (Matrix.zeros(2, 3), Matrix([0, 5])),
    ]
    box = list(range(-3, 4))
    for A, b in systems:
        n = A.cols
        sols = [Matrix(x) for x in product(box, repeat=n) if A * Matrix(x) == b]
        res = integer_linsolve(A, b)
        if isinstance(res, IntegerSolution):
            x0, B = res
            assert A * x0 == b
            assert (A * B).is_zero_matrix
            assert B.cols == n - A.rank()
            # Every lattice point is a solution.
            if B.cols:
                for z in product(range(-2, 3), repeat=B.cols):
                    assert A * (x0 + B * Matrix(z)) == b
            # Every enumerated solution lies in the lattice.
            for x in sols:
                assert _in_integer_span(B, x - x0)
        else:
            assert isinstance(res, DivisibilityContradiction)
            assert sols == []
            _check_contradiction_certificate(A, b, res)


def test_integer_linsolve_rational_but_no_integer_solution():
    # 2*x = 1 has the rational solution 1/2 but no integer solution.
    x = Symbol('x')
    assert linsolve([2*x - 1], x) == {(Rational(1, 2),)}
    res = integer_linsolve(Matrix([[2]]), Matrix([1]))
    assert isinstance(res, DivisibilityContradiction)
    assert res.divisor == 2
    assert res.residual == 1
    assert res.residual % res.divisor != 0
    assert (res.transform * Matrix([1]))[res.row] == res.residual


def test_integer_linsolve_multiple_rhs():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    b = Matrix([[1, 1, 3], [1, 2, 3]])
    results = integer_linsolve(A, b)
    assert len(results) == 3
    assert isinstance(results[0], IntegerSolution)
    assert isinstance(results[1], DivisibilityContradiction)
    assert isinstance(results[2], IntegerSolution)
    for j in (0, 2):
        assert A * results[j].particular == b[:, j]
        assert (A * results[j].nullbasis).is_zero_matrix
    # The decomposition is shared, so the nullspace bases coincide.
    assert results[0].nullbasis == results[2].nullbasis
    _check_contradiction_certificate(A, b[:, 1], results[1])
    assert results[1].column == 1


def test_integer_linsolve_errors():
    raises(ValueError, lambda: integer_linsolve(
        Matrix([[Rational(1, 2)]]), Matrix([1])))
    raises(ValueError, lambda: integer_linsolve(
        Matrix([[Symbol('x')]]), Matrix([1])))
    raises(ShapeError, lambda: integer_linsolve(
        Matrix([[1, 2]]), Matrix([1, 2])))
