from random import Random

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix, eye, zeros
from sympy.physics.quantum.commutator import Commutator
from sympy.testing.pytest import raises

from sympy.algebras.bch import (
    _hall_data, _lyndon_words, _reduce_to_hall, bch_expansion, hall_basis)


def _expm(M, order):
    R = eye(M.rows)
    P = eye(M.rows)
    for k in range(1, order + 1):
        P = P*M/k
        R = R + P
    return R


def _logm(E, order):
    A = E - eye(E.rows)
    R = zeros(E.rows)
    P = A.copy()
    for k in range(1, order + 1):
        if k > 1:
            P = P*A
        R = R + Rational((-1)**(k + 1), k)*P
    return R


def _eval_lie(expr, X, Y, MX, MY):
    """Evaluate a nested commutator expression on matrices."""
    if isinstance(expr, Commutator):
        A = _eval_lie(expr.args[0], X, Y, MX, MY)
        B = _eval_lie(expr.args[1], X, Y, MX, MY)
        return A*B - B*A
    if isinstance(expr, Add):
        return sum((_eval_lie(a, X, Y, MX, MY) for a in expr.args),
                   zeros(MX.rows))
    if isinstance(expr, Mul):
        result = eye(MX.rows)
        for a in expr.args:
            result = result*_eval_lie(a, X, Y, MX, MY)
        return result
    if expr == X:
        return MX
    if expr == Y:
        return MY
    if expr.is_Number:
        return expr*eye(MX.rows)
    raise ValueError("cannot evaluate %r" % (expr,))


def test_lyndon_words():
    assert _lyndon_words(1) == [(0,), (1,)]
    assert _lyndon_words(3) == [(0,), (0, 0, 1), (0, 1), (0, 1, 1), (1,)]
    assert _lyndon_words(4) == [(0,), (0, 0, 0, 1), (0, 0, 1), (0, 0, 1, 1),
                                (0, 1), (0, 1, 1), (0, 1, 1, 1), (1,)]


def test_hall_basis():
    X, Y = symbols("X Y", commutative=False)
    C = Commutator(X, Y)
    assert hall_basis(1, X, Y) == [X, Y]
    assert hall_basis(2, X, Y) == [X, Y, C]
    assert hall_basis(3, X, Y) == [X, Y, C, Commutator(X, C),
                                   Commutator(C, Y)]


def test_reduce_to_hall():
    # antisymmetry: [Y, X] = -[X, Y]
    assert _reduce_to_hall({(1, 0): 1, (0, 1): -1}, 2) == [((0, 1), -1)]
    # every Lyndon basis element reduces to itself
    words, _, expansion = _hall_data(4)
    for w in words:
        if len(w) > 1:
            assert _reduce_to_hall(expansion[w], 4) == [(w, 1)]


def test_bch_expansion_low_order():
    X, Y = symbols("X Y", commutative=False)
    C = Commutator(X, Y)
    assert bch_expansion(X, Y, 1) == X + Y
    assert bch_expansion(X, Y, 2) == X + Y + C/2
    assert (bch_expansion(X, Y, 3) ==
            X + Y + C/2 + Commutator(X, C)/12 - Commutator(Y, C)/12)
    assert (bch_expansion(X, Y, 4) ==
            X + Y + C/2 + Commutator(X, C)/12 - Commutator(Y, C)/12
            - Commutator(X, Commutator(Y, C))/24)


def test_bch_expansion_nilpotent_matrices():
    rng = Random(42)
    n = 6
    MX = Matrix(n, n, lambda i, j: Rational(rng.randint(-3, 3)) if j > i else 0)
    MY = Matrix(n, n, lambda i, j: Rational(rng.randint(-3, 3)) if j > i else 0)
    X, Y = symbols("X Y", commutative=False)
    exact = _logm(_expm(MX, n)*_expm(MY, n), n)
    # strictly upper triangular n x n matrices have nilpotency index at
    # most n, so words of degree >= n vanish and the truncated series is
    # exact once the order reaches n - 1
    for order in (n - 1, n):
        Z = bch_expansion(X, Y, order)
        assert _eval_lie(Z, X, Y, MX, MY) == exact
    # degree 5 words are generically nonzero here, so order 4 truncates
    # away a nonzero contribution
    assert _eval_lie(bch_expansion(X, Y, 4), X, Y, MX, MY) != exact


def test_bch_expansion_commuting():
    x, y = symbols("x y")
    for order in (1, 2, 5):
        assert bch_expansion(x, y, order) == x + y
    X, Y = symbols("X Y", commutative=False)
    E12 = Matrix(3, 3, lambda i, j: 1 if (i, j) == (0, 1) else 0)
    E13 = Matrix(3, 3, lambda i, j: 1 if (i, j) == (0, 2) else 0)
    assert E12*E13 == E13*E12
    Z = bch_expansion(X, Y, 5)
    assert _eval_lie(Z, X, Y, E12, E13) == E12 + E13


def test_bch_expansion_invalid_order():
    X, Y = symbols("X Y", commutative=False)
    for bad in (0, -1, Rational(3, 2), "3"):
        raises(ValueError, lambda: bch_expansion(X, Y, bad))
        raises(ValueError, lambda: hall_basis(bad, X, Y))
