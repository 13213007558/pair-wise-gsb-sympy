from __future__ import annotations
from sympy.core.function import (Function, Lambda, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.combinatorial.numbers import (fibonacci, euler, harmonic)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.solvers.recurr import (rsolve, rsolve_hyper, rsolve_poly,
    rsolve_ratio, rsolve_periodic)
from sympy.testing.pytest import raises, slow, XFAIL
from sympy.abc import a, b

y = Function('y')
n, k = symbols('n,k', integer=True)
C0, C1, C2 = symbols('C0,C1,C2')


def test_rsolve_poly():
    assert rsolve_poly([-1, -1, 1], 0, n) == 0
    assert rsolve_poly([-1, -1, 1], 1, n) == -1

    assert rsolve_poly([-1, n + 1], n, n) == 1
    assert rsolve_poly([-1, 1], n, n) == C0 + (n**2 - n)/2
    assert rsolve_poly([-n - 1, n], 1, n) == C0*n - 1
    assert rsolve_poly([-4*n - 2, 1], 4*n + 1, n) == -1

    assert rsolve_poly([-1, 1], n**5 + n**3, n) == \
        C0 - n**3 / 2 - n**5 / 2 + n**2 / 6 + n**6 / 6 + 2*n**4 / 3


def test_rsolve_ratio():
    solution = rsolve_ratio([-2*n**3 + n**2 + 2*n - 1, 2*n**3 + n**2 - 6*n,
        -2*n**3 - 11*n**2 - 18*n - 9, 2*n**3 + 13*n**2 + 22*n + 8], 0, n)
    assert solution == C0*(2*n - 3)/(n**2 - 1)/2


def test_rsolve_hyper():
    assert rsolve_hyper([-1, -1, 1], 0, n) in [
        C0*(S.Half - S.Half*sqrt(5))**n + C1*(S.Half + S.Half*sqrt(5))**n,
        C1*(S.Half - S.Half*sqrt(5))**n + C0*(S.Half + S.Half*sqrt(5))**n,
    ]

    assert rsolve_hyper([n**2 - 2, -2*n - 1, 1], 0, n) in [
        C0*rf(sqrt(2), n) + C1*rf(-sqrt(2), n),
        C1*rf(sqrt(2), n) + C0*rf(-sqrt(2), n),
    ]

    assert rsolve_hyper([n**2 - k, -2*n - 1, 1], 0, n) in [
        C0*rf(sqrt(k), n) + C1*rf(-sqrt(k), n),
        C1*rf(sqrt(k), n) + C0*rf(-sqrt(k), n),
    ]

    assert rsolve_hyper(
        [2*n*(n + 1), -n**2 - 3*n + 2, n - 1], 0, n) == C1*factorial(n) + C0*2**n

    assert rsolve_hyper(
        [n + 2, -(2*n + 3)*(17*n**2 + 51*n + 39), n + 1], 0, n) == 0

    assert rsolve_hyper([-n - 1, -1, 1], 0, n) == 0

    assert rsolve_hyper([-1, 1], n, n).expand() == C0 + n**2/2 - n/2

    assert rsolve_hyper([-1, 1], 1 + n, n).expand() == C0 + n**2/2 + n/2

    assert rsolve_hyper([-1, 1], 3*(n + n**2), n).expand() == C0 + n**3 - n

    assert rsolve_hyper([-a, 1],0,n).expand() == C0*a**n

    assert rsolve_hyper([-a, 0, 1], 0, n).expand() == (-1)**n*C1*a**(n/2) + C0*a**(n/2)

    assert rsolve_hyper([1, 1, 1], 0, n).expand() == \
        C0*(Rational(-1, 2) - sqrt(3)*I/2)**n + C1*(Rational(-1, 2) + sqrt(3)*I/2)**n

    assert rsolve_hyper([1, -2*n/a - 2/a, 1], 0, n) == 0

    # issue 27901
    assert rsolve_hyper([-1, 1], factorial(n**2), n) is None
    assert rsolve_hyper([-1, 1], n**n, n) is None
    assert rsolve_hyper([-1, 1], sqrt(n), n) is None
    assert rsolve_hyper([-1, 1], 1/k, k) is None

    assert rsolve_hyper([-2, 1], 2**(n**3 + 1), n) is None
    assert rsolve_hyper([-2, 1], 2**(a*n + 1), n) is None

    assert rsolve_hyper([-1, 1], fibonacci(n), n) is None
    assert rsolve_hyper([-1, 1], euler(n), n) is None
    assert rsolve_hyper([-1, 1], harmonic(n), n) is None
    assert rsolve_hyper([-(2*k - 1), 1], 1, k) is None

@XFAIL
def test_rsolve_ratio_missed():
    # this arises during computation
    # assert rsolve_hyper([-1, 1], 3*(n + n**2), n).expand() == C0 + n**3 - n
    assert rsolve_ratio([-n, n + 2], n, n) is not None


def recurrence_term(c, f):
    """Compute RHS of recurrence in f(n) with coefficients in c."""
    return sum(c[i]*f.subs(n, n + i) for i in range(len(c)))


@slow
def test_rsolve_bulk():
    """Some bulk-generated tests."""
    funcs = [ n, n + 1, n**2, n**3, n**4, n + n**2, 27*n + 52*n**2 - 3*
        n**3 + 12*n**4 - 52*n**5 ]
    coeffs = [ [-2, 1], [-2, -1, 1], [-1, 1, 1, -1, 1], [-n, 1], [n**2 -
        n + 12, 1] ]
    for p in funcs:
        # compute difference
        for c in coeffs:
            q = recurrence_term(c, p)
            if p.is_polynomial(n):
                assert rsolve_poly(c, q, n) == p
            # See issue 3956:
            if p.is_hypergeometric(n) and len(c) <= 3:
                assert rsolve_hyper(c, q, n).subs(zip(symbols('C:3'), [0, 0, 0])).expand() == p


def test_rsolve_0_sol_homogeneous():
    # fixed by cherry-pick from
    # https://github.com/diofant/diofant/commit/e1d2e52125199eb3df59f12e8944f8a5f24b00a5
    assert rsolve_hyper([n**2 - n + 12, 1], n*(n**2 - n + 12) + n + 1, n) == n


def test_rsolve():
    f = y(n + 2) - y(n + 1) - y(n)
    h = sqrt(5)*(S.Half + S.Half*sqrt(5))**n \
        - sqrt(5)*(S.Half - S.Half*sqrt(5))**n

    assert rsolve(f, y(n)) in [
        C0*(S.Half - S.Half*sqrt(5))**n + C1*(S.Half + S.Half*sqrt(5))**n,
        C1*(S.Half - S.Half*sqrt(5))**n + C0*(S.Half + S.Half*sqrt(5))**n,
    ]

    assert rsolve(f, y(n), [0, 5]) == h
    assert rsolve(f, y(n), {0: 0, 1: 5}) == h
    assert rsolve(f, y(n), {y(0): 0, y(1): 5}) == h
    assert rsolve(y(n) - y(n - 1) - y(n - 2), y(n), [0, 5]) == h
    assert rsolve(Eq(y(n), y(n - 1) + y(n - 2)), y(n), [0, 5]) == h

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)
    g = C1*factorial(n) + C0*2**n
    h = -3*factorial(n) + 3*2**n

    assert rsolve(f, y(n)) == g
    assert rsolve(f, y(n), []) == g
    assert rsolve(f, y(n), {}) == g

    assert rsolve(f, y(n), [0, 3]) == h
    assert rsolve(f, y(n), {0: 0, 1: 3}) == h
    assert rsolve(f, y(n), {y(0): 0, y(1): 3}) == h

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = y(n) - y(n - 1) - 2

    assert rsolve(f, y(n), {y(0): 0}) == 2*n
    assert rsolve(f, y(n), {y(0): 1}) == 2*n + 1
    assert rsolve(f, y(n), {y(0): 0, y(1): 1}) is None

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = 3*y(n - 1) - y(n) - 1

    assert rsolve(f, y(n), {y(0): 0}) == -3**n/2 + S.Half
    assert rsolve(f, y(n), {y(0): 1}) == 3**n/2 + S.Half
    assert rsolve(f, y(n), {y(0): 2}) == 3*3**n/2 + S.Half

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = y(n) - 1/n*y(n - 1)
    assert rsolve(f, y(n)) == C0/factorial(n)
    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = y(n) - 1/n*y(n - 1) - 1
    assert rsolve(f, y(n)) is None

    f = 2*y(n - 1) + (1 - n)*y(n)/n

    assert rsolve(f, y(n), {y(1): 1}) == 2**(n - 1)*n
    assert rsolve(f, y(n), {y(1): 2}) == 2**(n - 1)*n*2
    assert rsolve(f, y(n), {y(1): 3}) == 2**(n - 1)*n*3

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    f = (n - 1)*(n - 2)*y(n + 2) - (n + 1)*(n + 2)*y(n)

    assert rsolve(f, y(n), {y(3): 6, y(4): 24}) == n*(n - 1)*(n - 2)
    assert rsolve(
        f, y(n), {y(3): 6, y(4): -24}) == -n*(n - 1)*(n - 2)*(-1)**(n)

    assert f.subs(y, Lambda(k, rsolve(f, y(n)).subs(n, k))).simplify() == 0

    assert rsolve(Eq(y(n + 1), a*y(n)), y(n), {y(1): a}).simplify() == a**n

    assert rsolve(y(n) - a*y(n-2),y(n), \
            {y(1): sqrt(a)*(a + b), y(2): a*(a - b)}).simplify() == \
            a**(n/2 + 1) - b*(-sqrt(a))**n

    f = (-16*n**2 + 32*n - 12)*y(n - 1) + (4*n**2 - 12*n + 9)*y(n)

    yn = rsolve(f, y(n), {y(1): binomial(2*n + 1, 3)})
    sol = 2**(2*n)*n*(2*n - 1)**2*(2*n + 1)/12
    assert factor(expand(yn, func=True)) == sol

    sol = rsolve(y(n) + a*(y(n + 1) + y(n - 1))/2, y(n))
    assert str(sol) == 'C0*((-sqrt(1 - a**2) - 1)/a)**n + C1*((sqrt(1 - a**2) - 1)/a)**n'

    assert rsolve((k + 1)*y(k), y(k)) is None
    assert (rsolve((k + 1)*y(k) + (k + 3)*y(k + 1) + (k + 5)*y(k + 2), y(k))
            is None)

    assert rsolve(y(n) + y(n + 1) + 2**n + 3**n, y(n)) == (-1)**n*C0 - 2**n/3 - 3**n/4


def test_rsolve_raises():
    x = Function('x')
    raises(ValueError, lambda: rsolve(y(n) - y(k + 1), y(n)))
    raises(ValueError, lambda: rsolve(y(n) - y(n + 1), x(n)))
    raises(ValueError, lambda: rsolve(y(n) - x(n + 1), y(n)))
    raises(ValueError, lambda: rsolve(y(n) - sqrt(n)*y(n + 1), y(n)))
    raises(ValueError, lambda: rsolve(y(n) - y(n + 1), y(n), {x(0): 0}))
    raises(ValueError, lambda: rsolve(y(n) + y(n + 1) + 2**n + cos(n), y(n)))


def test_issue_6844():
    f = y(n + 2) - y(n + 1) + y(n)/4
    assert rsolve(f, y(n)) == 2**(-n + 1)*C1*n + 2**(-n)*C0
    assert rsolve(f, y(n), {y(0): 0, y(1): 1}) == 2**(1 - n)*n


def test_issue_18751():
    r = Symbol('r', positive=True)
    theta = Symbol('theta', real=True)
    f = y(n) - 2 * r * cos(theta) * y(n - 1) + r**2 * y(n - 2)
    assert rsolve(f, y(n)) == \
        C0*(r*(cos(theta) - I*Abs(sin(theta))))**n + C1*(r*(cos(theta) + I*Abs(sin(theta))))**n


def test_constant_naming():
    #issue 8697
    assert rsolve(y(n+3) - y(n+2) - y(n+1) + y(n), y(n)) == (-1)**n*C1 + C0 + C2*n
    assert rsolve(y(n+3)+3*y(n+2)+3*y(n+1)+y(n), y(n)).expand() == (-1)**n*C0 - (-1)**n*C1*n - (-1)**n*C2*n**2
    assert rsolve(y(n) - 2*y(n - 3) + 5*y(n - 2) - 4*y(n - 1),y(n),[1,3,8]) == 3*2**n - n - 2

    #issue 19630
    assert rsolve(y(n+3) - 3*y(n+1) + 2*y(n), y(n), {y(1):0, y(2):8, y(3):-2}) == (-2)**n + 2*n


@slow
def test_issue_15751():
    f = y(n) + 21*y(n + 1) - 273*y(n + 2) - 1092*y(n + 3) + 1820*y(n + 4) + 1092*y(n + 5) - 273*y(n + 6) - 21*y(n + 7) + y(n + 8)
    assert rsolve(f, y(n)) is not None


def test_issue_17990():
    f = -10*y(n) + 4*y(n + 1) + 6*y(n + 2) + 46*y(n + 3)
    sol = rsolve(f, y(n))
    expected = C0*((86*18**(S(1)/3)/69 + (-12 + (-1 + sqrt(3)*I)*(290412 +
        3036*sqrt(9165))**(S(1)/3))*(1 - sqrt(3)*I)*(24201 + 253*sqrt(9165))**
        (S(1)/3)/276)/((1 - sqrt(3)*I)*(24201 + 253*sqrt(9165))**(S(1)/3))
        )**n + C1*((86*18**(S(1)/3)/69 + (-12 + (-1 - sqrt(3)*I)*(290412 + 3036
        *sqrt(9165))**(S(1)/3))*(1 + sqrt(3)*I)*(24201 + 253*sqrt(9165))**
        (S(1)/3)/276)/((1 + sqrt(3)*I)*(24201 + 253*sqrt(9165))**(S(1)/3))
        )**n + C2*(-43*18**(S(1)/3)/(69*(24201 + 253*sqrt(9165))**(S(1)/3)) -
        S(1)/23 + (290412 + 3036*sqrt(9165))**(S(1)/3)/138)**n
    assert sol == expected
    e = sol.subs({C0: 1, C1: 1, C2: 1, n: 1}).evalf()
    assert abs(e + 0.130434782608696) < 1e-13

@XFAIL
def test_issue_27975a():
    assert rsolve_hyper([-2*2**n, 2*2**n], -2*binomial(n, k) + binomial(n + 1, k), k) is not None

@XFAIL
def test_issue_27975b():
    term = -2**(1 - k)*factorial(k + 2)/factorial(k - 1) + factorial(k + 3)/(2**k*factorial(k))
    assert rsolve_hyper([-2**n, 2**n], term, k) is not None

def test_issue_8697():
    a = Function('a')
    eq = a(n + 3) - a(n + 2) - a(n + 1) + a(n)
    assert rsolve(eq, a(n)) == (-1)**n*C1 + C0 + C2*n
    eq2 = a(n + 3) + 3*a(n + 2) + 3*a(n + 1) + a(n)
    assert (rsolve(eq2, a(n)) ==
            (-1)**n*C0 + (-1)**(n + 1)*C1*n + (-1)**(n + 1)*C2*n**2)

    assert rsolve(a(n) - 2*a(n - 3) + 5*a(n - 2) - 4*a(n - 1),
                  a(n), {a(0): 1, a(1): 3, a(2): 8}) == 3*2**n - n - 2

    # From issue thread (but fixed by https://github.com/diofant/diofant/commit/da9789c6cd7d0c2ceeea19fbf59645987125b289):
    assert rsolve(a(n) - 2*a(n - 1) - n, a(n), {a(0): 1}) == 3*2**n - n - 2


def test_diofantissue_294():
    f = y(n) - y(n - 1) - 2*y(n - 2) - 2*n
    assert rsolve(f, y(n)) == (-1)**n*C0 + 2**n*C1 - n - Rational(5, 2)
    # issue sympy/sympy#11261
    assert rsolve(f, y(n), {y(0): -1, y(1): 1}) == (-(-1)**n/2 + 2*2**n -
                                                    n - Rational(5, 2))
    # issue sympy/sympy#7055
    assert rsolve(-2*y(n) + y(n + 1) + n - 1, y(n)) == 2**n*C0 + n


def test_issue_15553():
    f = Function("f")
    assert rsolve(Eq(f(n), 2*f(n - 1) + n), f(n)) == 2**n*C0 - n - 2
    assert rsolve(Eq(f(n + 1), 2*f(n) + n**2 + 1), f(n)) == 2**n*C0 - n**2 - 2*n - 4
    assert rsolve(Eq(f(n + 1), 2*f(n) + n**2 + 1), f(n), {f(1): 0}) == 7*2**n/2 - n**2 - 2*n - 4
    assert rsolve(Eq(f(n), 2*f(n - 1) + 3*n**2), f(n)) == 2**n*C0 - 3*n**2 - 12*n - 18
    assert rsolve(Eq(f(n), 2*f(n - 1) + n**2), f(n)) == 2**n*C0 - n**2 - 4*n - 6
    assert rsolve(Eq(f(n), 2*f(n - 1) + n), f(n), {f(0): 1}) == 3*2**n - n - 2


def _iterate_periodic(coeffs, f, p, init, count):
    k = len(coeffs[0]) - 1
    seq = [S(init[i]) for i in range(k)]
    while len(seq) < count:
        i = len(seq) - k
        r = i % p
        a = coeffs[r]
        seq.append((S(f[r]) - sum(a[j]*seq[i + j] for j in range(k)))/a[k])
    return seq


def _check_periodic_phases(sol, coeffs, f, p):
    m = Symbol('m', integer=True, nonnegative=True)
    k = len(coeffs[0]) - 1
    for r in range(p):
        residual = sum(coeffs[r][i]*sol[(r + i) % p].subs(n, p*m + r + i)
                       for i in range(k + 1)) - f[r]
        assert expand(residual, func=True).simplify() == 0


def test_rsolve_periodic_order3_period3():
    # y(n + 3) = 2*y(n) + 1 for n % 3 == 0,
    # y(n + 3) = y(n + 1)     for n % 3 == 1,
    # y(n + 3) = y(n + 2)     for n % 3 == 2.
    coeffs = [[-2, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]]
    f = [1, 0, 0]
    sol = rsolve_periodic(coeffs, f, y(n), 3)
    assert set(sol) == {0, 1, 2}
    assert sol[0].expand() == 2**(n/3)*C0 + 2**(n/3) - 1
    assert sol[2] == C2
    # finite transient from the zero eigenvalue of the state matrix
    assert (sol[1] - (C2 + (C1 - C2)*KroneckerDelta(0, (n - 1)/3))).expand() == 0

    _check_periodic_phases(sol, coeffs, f, 3)

    sol = rsolve_periodic(coeffs, f, y(n), 3, {0: 1, 1: 2, 2: 3})
    assert sol[0] == 2**(n/3 + 1) - 1
    assert sol[1] == 3 - KroneckerDelta(0, (n - 1)/3)
    assert sol[2] == 3

    seq = _iterate_periodic(coeffs, f, 3, {0: 1, 1: 2, 2: 3}, 40)
    assert [sol[i % 3].subs(n, i) for i in range(40)] == seq


def test_rsolve_periodic_jordan_block():
    # The period state matrix is a nontrivial Jordan block with
    # eigenvalue 1, so the general term involves binomial coefficients.
    # y(n + 2) = y(n + 1) + y(n) for even n, y(n + 2) = y(n) + 1 for odd n.
    coeffs = [[-1, -1, 1], [-1, 0, 1]]
    f = [0, 1]
    sol = rsolve_periodic(coeffs, f, y(n), 2)
    assert sol[0] == C0 + C1*n/2 + binomial(n/2, 2)
    assert sol[1] == C1 + (n - 1)/2

    _check_periodic_phases(sol, coeffs, f, 2)

    sol = rsolve_periodic(coeffs, f, y(n), 2, {y(0): 1, y(1): 0})
    seq = _iterate_periodic(coeffs, f, 2, {0: 1, 1: 0}, 40)
    assert [sol[i % 2].subs(n, i) for i in range(40)] == seq


def test_rsolve_periodic_zero_eigenvalue():
    # The period state matrix is nilpotent, so the general term keeps a
    # finite transient expressed with KroneckerDelta terms.
    # y(n + 2) = y(n + 1) for even n, y(n + 2) = 0 for odd n.
    coeffs = [[0, -1, 1], [0, 0, 1]]
    f = [0, 0]
    sol = rsolve_periodic(coeffs, f, y(n), 2)
    assert sol[0].has(KroneckerDelta)
    assert sol[1].has(KroneckerDelta)

    sol = rsolve_periodic(coeffs, f, y(n), 2, {0: 5, 1: 7})
    seq = _iterate_periodic(coeffs, f, 2, {0: 5, 1: 7}, 30)
    assert [sol[i % 2].subs(n, i) for i in range(30)] == seq
    assert seq[:6] == [5, 7, 7, 0, 0, 0]

    # the transient vanishes and the sequence stays at zero
    for r in range(2):
        for i in range(2, 12):
            residual = sum(coeffs[r][j]*sol[(r + j) % 2].subs(n, 2*i + r + j)
                           for j in range(3))
            assert residual == 0


def test_rsolve_periodic_init():
    coeffs = [[-1, 1], [-2, 1]]
    f = [0, 0]
    sol = rsolve_periodic(coeffs, f, y(n), 2, [3])
    assert sol == {0: 3*2**(n/2), 1: 3*2**(n/2 - S(1)/2)}

    # not enough initial values: report which ones are missing
    raises(ValueError, lambda: rsolve_periodic(
        [[-1, -1, 1], [-1, 0, 1]], [0, 0], y(n), 2, {0: 3}))
    try:
        rsolve_periodic([[-2, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 1]],
                        [1, 0, 0], y(n), 3, {0: 1})
    except ValueError as exc:
        assert 'y(1)' in str(exc) and 'y(2)' in str(exc)
    else:
        assert False

    # inconsistent initial values
    assert rsolve_periodic([[0, -1, 1], [0, 0, 1]], [0, 0], y(n), 2,
                           {0: 5, 1: 7, 2: 1}) is None


def test_rsolve_periodic_errors():
    raises(ValueError, lambda: rsolve_periodic([[-1, 1]], [0], y(n), 0))
    raises(ValueError, lambda: rsolve_periodic([[-1, 1]], [0], y(n), 2))
    raises(ValueError, lambda: rsolve_periodic([[-1, 1]], [0, 0], y(n), 1))
    raises(ValueError, lambda: rsolve_periodic([[1, 0]], [0], y(n), 1))
    raises(ValueError, lambda: rsolve_periodic([[-1, 1]], [n], y(n), 1))
    raises(ValueError, lambda: rsolve_periodic([[sqrt(2), 1]], [0], y(n), 1))
