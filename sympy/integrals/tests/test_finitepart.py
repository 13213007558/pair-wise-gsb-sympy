"""Tests for Hadamard finite part integrals of rational functions."""
from __future__ import annotations

from sympy.core import Rational, S, pi
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.finitepart import hadamard_finite_part_integral
from sympy.testing.pytest import raises
from sympy.abc import x

fp = hadamard_finite_part_integral


def test_second_order_pole():
    # F.p. int_{-1}^{1} x**-2 dx = [-1/x] with finite parts = -1 - 1
    assert fp(1/x**2, x, -1, 1) == -2
    # asymmetric interval: -1/2 - 1
    assert fp(1/x**2, x, -1, 2) == Rational(-3, 2)
    # shifted pole
    assert fp(1/(x - 1)**2, x, 0, 3) == -Rational(1, 2) - 1


def test_third_order_pole():
    # antiderivative -1/(2*x**2); -1/8 - (-1/2)
    assert fp(1/x**3, x, -1, 2) == Rational(3, 8)
    assert fp(1/(x - 1)**3, x, 0, 3) == Rational(3, 8)


def test_simple_pole():
    # for a simple pole the finite part coincides with the Cauchy PV
    assert fp(1/x, x, -1, 2) == log(2)
    assert fp(1/x, x, -3, -1) == -log(3)


def test_polynomial_and_regular():
    assert fp(x**2, x, 0, 1) == Rational(1, 3)
    assert fp(1/(x**2 + 1), x, 0, 1) == pi/4
    assert fp((x**3 + 1)/(x - 2), x, 0, 1) == Rational(16, 3) - 9*log(2)


def test_mixed_orders():
    # x/(x - 1)**2 = 1/(x - 1) + 1/(x - 1)**2 -> 0 + (-2)
    assert fp(x/(x - 1)**2, x, 0, 2) == -2


def test_multiple_poles():
    # 1/(x**2*(x - 1)**2) = 1/x**2 + 2/x + 1/(x - 1)**2 - 2/(x - 1)
    # F.p. = -3/2 + 2*log(2) - 3/2 + 2*log(2)
    assert fp(1/(x**2*(x - 1)**2), x, -1, 2) == -3 + 4*log(2)


def test_algebraic_pole():
    res = fp(1/(x**2 - 2)**2, x, 0, 2)
    expected = sqrt(2)*log(1 + sqrt(2))/8 - Rational(1, 4)
    assert abs((res - expected).evalf(50)) < S(10)**-45


def test_algebraic_pole_rootof():
    res = fp((x + 1)/((x**5 - x - 1)*(x - 3)), x, 1, 2)
    assert res.has(log)
    assert abs(res.evalf(50) - _numeric_fp(
        (x + 1)/((x**5 - x - 1)*(x - 3)), 1, 2)) < S(10)**-8


def test_endpoint_poles():
    assert fp(1/x**2, x, 0, 1, allow_endpoint_poles=True) == -1
    assert fp(1/x, x, 0, 2, allow_endpoint_poles=True) == log(2)
    assert fp(1/(x - 1)**2, x, 0, 1, allow_endpoint_poles=True) == -1
    # poles at both endpoints cancel by symmetry
    assert fp(1/(x*(1 - x)), x, 0, 1, allow_endpoint_poles=True) == 0


def test_endpoint_pole_requires_opt_in():
    raises(ValueError, lambda: fp(1/x**2, x, 0, 1))
    raises(ValueError, lambda: fp(1/(x - 1), x, 0, 1))
    try:
        fp(1/x**2, x, 0, 1)
    except ValueError as exc:
        assert '0' in str(exc) and 'allow_endpoint_poles' in str(exc)


def test_unsupported_input():
    raises(ValueError, lambda: fp(1/(x + sqrt(2)), x, 0, 1))
    raises(ValueError, lambda: fp(1/x, x, 0, sqrt(2)))
    raises(ValueError, lambda: fp(1/x, x, 2, 1))


def _synthetic_div(coeffs, r):
    # Quotient of a polynomial (descending coefficients) divided by (t - r).
    out = [coeffs[0]]
    for c in coeffs[1:]:
        out.append(c + out[-1]*r)
    return out[:-1]


def _numeric_fp(f, a, b, eps='1e-12', dps=50):
    # Independent numerical finite part: truncate eps-neighborhoods of the
    # poles, integrate with mpmath, and subtract the divergent terms whose
    # coefficients are obtained by numerical differentiation.
    import mpmath as mp
    from sympy.polys.polytools import Poly, real_roots
    mp.mp.dps = dps
    eps = mp.mpf(eps)
    num, den = f.as_numer_denom()
    denp = Poly(den, x)
    nump = Poly(num, x)
    ncs = [mp.mpf(c.numerator)/mp.mpf(c.denominator)
           for c in nump.all_coeffs()]
    dcs = [mp.mpf(c.numerator)/mp.mpf(c.denominator)
           for c in denp.all_coeffs()]
    poles = []
    for fac, mult in denp.factor_list()[1]:
        for r in real_roots(fac):
            if a <= r <= b:
                poles.append((mp.mpf(str(r.evalf(dps))), mult))
    poles.sort()
    fn = lambda t: mp.polyval(ncs, t)/mp.polyval(dcs, t)
    ma, mb = mp.mpf(a), mp.mpf(b)
    segments = []
    left = ma
    for r, _ in poles:
        if r == ma:
            left = r + eps
        elif r == mb:
            segments.append((left, r - eps))
            left = None
        else:
            segments.append((left, r - eps))
            left = r + eps
    if left is not None:
        segments.append((left, mb))
    total = mp.mpf(0)
    for lo, hi in segments:
        if lo < hi:
            total += mp.quad(fn, [lo, hi])
    for r, m in poles:
        qcs = dcs
        for _ in range(m):
            qcs = _synthetic_div(qcs, r)
        g = lambda t: mp.polyval(ncs, t)/mp.polyval(qcs, t)
        coeffs = [mp.diff(g, r, n=m - k)/mp.factorial(m - k)
                  for k in range(1, m + 1)]
        for k, A in enumerate(coeffs, start=1):
            phi = (lambda t: mp.log(abs(t))) if k == 1 else (
                lambda t: -t**(1 - k)/(k - 1))
            if r == mp.mpf(a):
                total += A*phi(eps)
            elif r == mp.mpf(b):
                total -= A*phi(-eps)
            else:
                total += A*(phi(eps) - phi(-eps))
    return total


def test_numerical_crosscheck():
    cases = [
        (1/x**2, -1, 1),
        (1/x**3, -1, 2),
        (1/(x**2*(x - 1)**2), -1, 2),
        (1/(x**2 - 2)**2, 0, 2),
        (x/(x - 1)**2, 0, 2),
    ]
    for f, a, b in cases:
        exact = fp(f, x, a, b)
        numeric = _numeric_fp(f, a, b)
        assert abs(exact.evalf(50) - numeric) < S(10)**-8


def test_numerical_crosscheck_endpoint():
    exact = fp(1/x**2, x, 0, 1, allow_endpoint_poles=True)
    assert abs(exact.evalf(50) - _numeric_fp(1/x**2, 0, 1)) < S(10)**-8
