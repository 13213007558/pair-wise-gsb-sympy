from __future__ import annotations
from sympy.core.numbers import Rational
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.core import S, symbols
from sympy.testing.pytest import raises
from sympy.discrete.recurrences import linrec
from sympy.abc import x
from sympy.polys.polytools import Poly, gcd
from sympy.series.series import series
from sympy.discrete.recurrences import (
    LinearRecurrence, find_linear_recurrence)

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


def test_find_linear_recurrence_basic():
    p = 7
    fib = [0, 1]
    for _ in range(18):
        fib.append((fib[-1] + fib[-2]) % p)
    r = find_linear_recurrence(fib, p)
    assert r.modulus == p
    assert r.coeffs == (1, 1)
    assert r.order == 2
    assert r.start == 2
    assert r.status == 'unique'
    assert r.prefix_length == 20
    assert r.holdout_failure is None
    assert r.check(fib) is None
    assert r.extend(fib, 4) == [3, 5, 1, 6]
    assert r.extend(fib, 0) == []
    assert r.characteristic_polynomial() == \
        Poly(x**2 - x - 1, x, modulus=p)
    assert LinearRecurrence.from_characteristic_polynomial(
        r.characteristic_polynomial()) == r
    # cross-check extension against linrec
    assert r.extend(fib, 5) == [linrec([1, 1], fib[:2], n) % p
                                for n in range(20, 25)]


def test_find_linear_recurrence_statuses():
    r = find_linear_recurrence([], 5)
    assert r.status == 'empty'
    assert r.order is None and r.coeffs is None and r.start is None
    raises(ValueError, lambda: r.extend([], 1))
    raises(ValueError, lambda: r.check([0]))
    raises(ValueError, lambda: r.characteristic_polynomial())

    r = find_linear_recurrence([0, 0, 0, 0], 5)
    assert r.status == 'zero'
    assert r.order == 0 and r.coeffs == () and r.start == 0
    assert r.extend([0, 0, 0, 0], 3) == [0, 0, 0]
    assert r.check([0, 0, 0, 0, 0]) is None
    assert r.check([0, 0, 1]) == 2
    assert r.characteristic_polynomial() == Poly(1, x, modulus=5)

    # 2*order > len(prefix): the continuation is not determined
    r = find_linear_recurrence([0, 0, 1], 5)
    assert r.status == 'underdetermined'
    assert r.order == 3
    assert r.check([0, 0, 1]) is None
    r = find_linear_recurrence([2], 3)
    assert r.status == 'underdetermined'
    assert r.order == 1
    r = find_linear_recurrence([1, 2, 4, 3], 5)
    assert r.status == 'unique'
    assert r.coeffs == (2,)


def test_find_linear_recurrence_holdout():
    p = 7
    fib = [0, 1]
    for _ in range(18):
        fib.append((fib[-1] + fib[-2]) % p)
    train, hold = fib[:16], fib[16:]
    r = find_linear_recurrence(train, p, holdout=hold)
    assert r.holdout_failure is None
    bad = hold[:3] + [(hold[3] + 1) % p]
    r = find_linear_recurrence(train, p, holdout=bad)
    assert r.holdout_failure == 19
    # the coefficients are inferred from the training prefix only and
    # are not refitted to the holdout data
    assert r.coeffs == find_linear_recurrence(train, p).coeffs
    # holdout validation of the all-zero recurrence
    r = find_linear_recurrence([0, 0, 0], 3, holdout=[0, 1, 0])
    assert r.holdout_failure == 4
    raises(ValueError, lambda: find_linear_recurrence([], 3, holdout=[1]))


def test_find_linear_recurrence_errors():
    raises(ValueError, lambda: find_linear_recurrence([1, 2], 1))
    raises(ValueError, lambda: find_linear_recurrence([1, 2], 4))
    raises(ValueError, lambda: find_linear_recurrence([1, 2], -7))
    raises(TypeError, lambda: find_linear_recurrence(3, 7))
    r = find_linear_recurrence([1, 1, 2, 3], 5)
    raises(ValueError, lambda: r.extend([1], 2))
    raises(ValueError, lambda: r.extend([1, 1], -1))
    raises(ValueError, lambda: LinearRecurrence((1, 1), 10))

def test_find_linear_recurrence_from_characteristic_polynomial():
    r = LinearRecurrence.from_characteristic_polynomial(
        Poly(x**3 + 2*x + 1, x, modulus=5))
    assert r.coeffs == (0, 3, 4)
    assert r.modulus == 5
    assert r.status == 'assumed'
    assert r.characteristic_polynomial() == Poly(x**3 + 2*x + 1, x, modulus=5)
    raises(ValueError, lambda: LinearRecurrence.from_characteristic_polynomial(
        Poly(x**2 + 1, x)))
    raises(ValueError, lambda: LinearRecurrence.from_characteristic_polynomial(
        Poly(2*x**2 + 1, x, modulus=7)))
    y = symbols('y')
    raises(ValueError, lambda: LinearRecurrence.from_characteristic_polynomial(
        Poly(x**2 + 1, x, y, modulus=7)))


def test_find_linear_recurrence_rational_gf():
    # sequences with a known rational generating function P(x)/Q(x)
    # over GF(p): the coefficients of the series satisfy the recurrence
    # given by Q, and the characteristic polynomial is x**L*Q(1/x)
    p = 11
    cases = [
        ((1,), (1, 3, 5)),
        ((2, 1), (1, 0, 4, 3)),
        ((1, 2), (1, 7, 2)),
        ((3, 0, 1), (1, 1, 0, 6)),
    ]
    for P, Q in cases:
        L = len(Q) - 1
        Px = sum(c*x**i for i, c in enumerate(P))
        Qx = sum(c*x**i for i, c in enumerate(Q))
        assert gcd(Px, Qx, modulus=p) == 1
        n = 2*L + 6
        ser = series(Px/Qx, x, 0, n).removeO()
        s = [(ser.coeff(x, i).p*pow(ser.coeff(x, i).q, -1, p)) % p
             for i in range(n)]
        r = find_linear_recurrence(s, p)
        assert r.status == 'unique'
        assert r.order == L
        assert list(r.coeffs) == [(-Q[i + 1]) % p for i in range(L)]
        assert r.characteristic_polynomial(x) == \
            Poly((x**L*Qx.subs(x, 1/x)).expand(), x, modulus=p)
        assert r.extend(s, 4) == [linrec(list(r.coeffs), s[:L], k) % p
                                  for k in range(n, n + 4)]


def _gf_solve_consistency(rows, ncols, p):
    # Gaussian elimination over GF(p) on an augmented matrix; returns
    # (consistent, rank) for the system rows[i][:-1] * c == rows[i][-1]
    rows = [r[:] for r in rows]
    rank = 0
    for col in range(ncols):
        piv = next((i for i in range(rank, len(rows)) if rows[i][col] % p),
                   None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = pow(rows[rank][col] % p, -1, p)
        rows[rank] = [v*inv % p for v in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][col] % p:
                f = rows[i][col] % p
                rows[i] = [(a - f*b) % p for a, b in zip(rows[i], rows[rank])]
        rank += 1
    consistent = all(any(v % p for v in r[:ncols]) or r[ncols] % p == 0
                     for r in rows)
    return consistent, rank


def _brute_min_recurrence(s, p):
    # minimal L and coefficient rank for the system
    # s[j] == sum(c[i]*s[j - 1 - i] for i in range(L)), L <= j < len(s)
    n = len(s)
    for L in range(n + 1):
        rows = [[s[j - 1 - i] for i in range(L)] + [s[j]]
                for j in range(L, n)]
        consistent, rank = _gf_solve_consistency(rows, L, p)
        if consistent:
            return L, rank


def test_find_linear_recurrence_brute_force():
    # exhaustive cross-check of the inferred order, coefficients and
    # uniqueness status against direct solution of the linear equations
    import random
    rng = random.Random(1234)
    for p in (2, 3, 5):
        for n in range(1, 10):
            for _ in range(10):
                s = [rng.randrange(p) for _ in range(n)]
                r = find_linear_recurrence(s, p)
                L, rank = _brute_min_recurrence(s, p)
                if not any(s):
                    assert r.status == 'zero' and r.order == 0
                    continue
                assert r.order == L
                assert r.check(s) is None
                assert r.status == ('unique' if 2*L <= n
                                    else 'underdetermined')
                if 2*L <= n:
                    assert rank == L
                # coefficients satisfy every equation of the prefix
                for j in range(L, n):
                    assert sum(r.coeffs[i]*s[j - 1 - i]
                               for i in range(L)) % p == s[j] % p


def test_find_linear_recurrence_generated():
    # sequences generated by iterating a random recurrence: the
    # inference must recover a recurrence that reproduces the future
    import random
    rng = random.Random(7)
    p = 13
    for _ in range(20):
        L = rng.randrange(1, 6)
        c = [rng.randrange(p) for _ in range(L)]
        init = [rng.randrange(p) for _ in range(L)]
        n = 2*L + 5
        s = [linrec(c, init, k) % p for k in range(n)]
        if not any(s):
            continue
        r = find_linear_recurrence(s[:2*L], p)
        assert r.status == 'unique'
        assert r.order <= L
        assert r.extend(s[:2*L], 5) == s[2*L:]
        assert find_linear_recurrence(
            s[:2*L], p, holdout=s[2*L:]).holdout_failure is None
