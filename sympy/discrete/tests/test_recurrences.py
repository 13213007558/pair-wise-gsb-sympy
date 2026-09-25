from __future__ import annotations
from itertools import product
from random import Random
from sympy.core.numbers import Rational
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.core import S, symbols, Integer
from sympy.polys import Poly, GF
from sympy.series.series import series
from sympy.testing.pytest import raises
from sympy.discrete.recurrences import (linrec, infer_linear_recurrence,
    LinearRecurrence)

def test_linrec():
    assert linrec(coeffs=[1, 1], init=[1, 1], n=20) == 10946
    assert linrec(coeffs=[1, 2, 3, 4, 5], init=[1, 1, 0, 2], n=10) == 1040
    assert linrec(coeffs=[0, 0, 11, 13], init=[23, 27], n=25) == 59628567384
    assert linrec(coeffs=[0, 0, 1, 1, 2], init=[1, 5, 3], n=15) == 165
    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=70) == \
        56889923441670659718376223533331214868804815612050381493741233489928913241
    assert linrec(coeffs=[0]*55 + [1, 1, 2, 3], init=[0]*50 + [1, 2, 3], n=4000) == \
        702633573874937994980598979769135096432444135301118916539

    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=10**4)
    assert linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4], n=10**5)

    assert all(linrec(coeffs=[1, 1], init=[0, 1], n=n) == fibonacci(n)
                                                    for n in range(95, 115))

    assert all(linrec(coeffs=[1, 1], init=[1, 1], n=n) == fibonacci(n + 1)
                                                    for n in range(595, 615))

    a = [S.Half, Rational(3, 4), Rational(5, 6), 7, Rational(8, 9), Rational(3, 5)]
    b = [1, 2, 8, Rational(5, 7), Rational(3, 7), Rational(2, 9), 6]
    x, y, z = symbols('x y z')

    assert linrec(coeffs=a[:5], init=b[:4], n=80) == \
        Rational(1726244235456268979436592226626304376013002142588105090705187189,
            1960143456748895967474334873705475211264)

    assert linrec(coeffs=a[:4], init=b[:4], n=50) == \
        Rational(368949940033050147080268092104304441, 504857282956046106624)

    assert linrec(coeffs=a[3:], init=b[:3], n=35) == \
        Rational(97409272177295731943657945116791049305244422833125109,
            814315512679031689453125)

    assert linrec(coeffs=[0]*60 + [Rational(2, 3), Rational(4, 5)], init=b, n=3000) == \
        Rational(26777668739896791448594650497024, 48084516708184142230517578125)

    raises(TypeError, lambda: linrec(coeffs=[11, 13, 15, 17], init=[1, 2, 3, 4, 5], n=1))
    raises(TypeError, lambda: linrec(coeffs=a[:4], init=b[:5], n=10000))
    raises(ValueError, lambda: linrec(coeffs=a[:4], init=b[:4], n=-10000))
    raises(TypeError, lambda: linrec(x, b, n=10000))
    raises(TypeError, lambda: linrec(a, y, n=10000))

    assert linrec(coeffs=[x, y, z], init=[1, 1, 1], n=4) == \
        x**2  + x*y + x*z + y + z
    assert linrec(coeffs=[1, 2, 1], init=[x, y, z], n=20) == \
        269542*x + 664575*y + 578949*z
    assert linrec(coeffs=[0, 3, 1, 2], init=[x, y], n=30) == \
        58516436*x + 56372788*y
    assert linrec(coeffs=[0]*50 + [1, 2, 3], init=[x, y, z], n=1000) == \
        11477135884896*x + 25999077948732*y + 41975630244216*z
    assert linrec(coeffs=[], init=[1, 1], n=20) == 0
    assert linrec(coeffs=[x, y, z], init=[1, 2, 3], n=2) == 3


def test_infer_linear_recurrence_statuses():
    p = 7

    r = infer_linear_recurrence([], p)
    assert r.status == LinearRecurrence.EMPTY
    assert r.order is None
    assert r.coefficients == ()
    assert r.start is None
    raises(ValueError, lambda: r.extend(3))
    raises(ValueError, lambda: r.connection_polynomial())
    raises(ValueError, lambda: r.characteristic_polynomial())

    r = infer_linear_recurrence([0, 0, 0, 0], p)
    assert r.status == LinearRecurrence.ZERO
    assert r.order == 0
    assert r.coefficients == ()
    assert r.start == 0
    assert r.extend(3) == [0, 0, 0]
    assert r.validate([0, 0]) is None
    assert r.validate([0, 1]) == 5

    # 3 terms cannot uniquely determine an order 2 recurrence
    r = infer_linear_recurrence([1, 2, 3], p)
    assert r.status == LinearRecurrence.UNDERDETERMINED
    assert r.order == 2
    assert r.start == 2

    r = infer_linear_recurrence([1, 1, 2, 3, 5, 1, 6, 0], p)
    assert r.status == LinearRecurrence.UNIQUE
    assert r.order == 2
    assert r.coefficients == (1, 1)
    assert r.start == 2

    # the recurrence applies to the whole prefix from index start on
    s = [1, 1, 2, 3, 5, 1, 6, 0]
    assert all(sum(c*s[k - 1 - i] for i, c in enumerate(r.coefficients)) % p
               == s[k] for k in range(r.start, len(s)))

    # terms are reduced modulo p and symbolic integers are accepted
    r = infer_linear_recurrence([Integer(8), -1, 8, -1, 8], p)
    assert r.prefix == (1, 6, 1, 6, 1)
    assert r.coefficients == (6,)
    assert r.extend(2) == [6, 1]

    raises(ValueError, lambda: infer_linear_recurrence([1, 2, 3], 1))
    raises(ValueError, lambda: infer_linear_recurrence([1, 2, 3], 4))
    raises(ValueError, lambda: infer_linear_recurrence([1, 2, 3], -7))


def test_infer_linear_recurrence_holdout():
    p = 7
    prefix = [1, 1, 2, 3, 5, 1, 6, 0]

    r = infer_linear_recurrence(prefix, p, holdout=[6, 6, 5, 4, 2])
    assert r.holdout == (6, 6, 5, 4, 2)
    assert r.holdout_mismatch is None

    r = infer_linear_recurrence(prefix, p, holdout=[6, 6, 4])
    # first inconsistent term is at absolute index 10
    assert r.holdout_mismatch == 10
    # a failed holdout must not change the inferred recurrence
    assert r.coefficients == (1, 1)
    assert r.order == 2
    assert r.status == LinearRecurrence.UNIQUE

    # validate() rechecks arbitrary data against the same recurrence
    assert r.validate([6, 6, 5]) is None
    assert r.validate([0]) == 8
    assert r.validate([]) is None

    r = infer_linear_recurrence([], p, holdout=[1, 2])
    assert r.status == LinearRecurrence.EMPTY
    assert r.holdout_mismatch is None


def test_infer_linear_recurrence_extend():
    p = 11
    init = [1, 1, 2]
    coeffs = [1, 2, 3]
    s = [linrec(coeffs=coeffs, init=init, n=n) % p for n in range(30)]

    r = infer_linear_recurrence(s[:20], p)
    assert r.status == LinearRecurrence.UNIQUE
    assert r.order == 3
    assert r.coefficients == (1, 2, 3)
    assert r.extend(10) == s[20:]

    # an underdetermined recurrence still extends consistently
    r = infer_linear_recurrence(s[:5], p)
    assert r.status == LinearRecurrence.UNDERDETERMINED
    ext = r.extend(5)
    full = list(r.prefix) + ext
    assert all(sum(c*full[k - 1 - i] for i, c in enumerate(r.coefficients))
               % p == full[k] for k in range(r.start, len(full)))

    raises(ValueError, lambda: r.extend(-1))


def test_infer_linear_recurrence_rational_generating_functions():
    x = symbols('x')
    p = 101
    cases = [
        ((1 + 2*x)/(1 - x - x**2), (1, 1)),
        ((1 + x)/(1 - 3*x + x**2 + 2*x**3), (3, -1, -2)),
        (1/(1 - 2*x), (2,)),
        ((2 - x)/((1 - x)**2*(1 + x)), None),
    ]
    for f, coeffs in cases:
        ser = series(f, x, 0, 20).removeO().expand()
        s = [int(ser.coeff(x, i)) % p for i in range(20)]
        r = infer_linear_recurrence(s[:12], p, holdout=s[12:])
        assert r.status == LinearRecurrence.UNIQUE
        assert r.holdout_mismatch is None
        if coeffs is not None:
            assert r.order == len(coeffs)
            assert r.coefficients == tuple(c % p for c in coeffs)
        # the extended terms keep matching the generating function
        assert r.extend(8) == s[12:]


def test_infer_linear_recurrence_polynomials():
    x = symbols('x')
    p = 7
    r = infer_linear_recurrence([1, 1, 2, 3, 5, 1, 6, 0], p)

    char = r.characteristic_polynomial(x)
    assert char == Poly(x**2 - x - 1, x, modulus=p)
    assert char.get_domain() == GF(p)

    conn = r.connection_polynomial(x)
    assert conn == Poly(1 - x - x**2, x, modulus=p)
    assert conn.get_domain() == GF(p)

    # the connection polynomial annihilates the sequence
    s = [1, 1, 2, 3, 5, 1, 6, 0]
    c = [int(conn.nth(i)) for i in range(r.order + 1)]
    assert all(sum(c[i]*s[k - i] for i in range(r.order + 1)) % p == 0
               for k in range(r.start, len(s)))

    # default symbol and the zero recurrence
    assert r.connection_polynomial().gens[0].is_symbol
    z = infer_linear_recurrence([0, 0, 0], p)
    assert z.connection_polynomial(x) == Poly(1, x, modulus=p)
    assert z.characteristic_polynomial(x) == Poly(1, x, modulus=p)


def _shortest_recurrences_brute_force(s, p):
    # all shortest recurrences of s over GF(p) by exhaustive search
    n = len(s)
    for L in range(n + 1):
        sols = [c for c in product(range(p), repeat=L)
                if all(sum(c[i]*s[k - 1 - i] for i in range(L)) % p == s[k]
                       for k in range(L, n))]
        if sols:
            return L, sols
    raise AssertionError("unreachable")


def test_infer_linear_recurrence_brute_force():
    rng = Random(238475)
    for p, n, trials in [(2, 8, 40), (3, 8, 30), (5, 5, 20)]:
        for _ in range(trials):
            s = [rng.randrange(p) for _ in range(n)]
            r = infer_linear_recurrence(s, p)
            L, sols = _shortest_recurrences_brute_force(s, p)
            if not any(s):
                assert r.status == LinearRecurrence.ZERO
                assert L == 0
                continue
            # the inferred order is minimal and the coefficients are
            # one of the shortest recurrences
            assert r.order == L
            assert tuple(r.coefficients) in [tuple(c) for c in sols]
            # uniqueness of the shortest recurrence decides the status
            if len(sols) == 1:
                assert r.status == LinearRecurrence.UNIQUE
            else:
                assert r.status == LinearRecurrence.UNDERDETERMINED
