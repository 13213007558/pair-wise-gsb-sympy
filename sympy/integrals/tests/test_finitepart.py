"""Tests for Hadamard finite part integration of rational functions."""
from __future__ import annotations

from sympy import (cancel, lambdify, log, pi, real_roots, simplify, sqrt,
    symbols, sin, S)
from sympy.integrals.finitepart import (
    UnsupportedSingularityError, _laurent_coefficients,
    hadamard_finite_part_integrate)
from sympy.testing.pytest import raises

from mpmath import mp, quad

x = symbols('x')
fp = hadamard_finite_part_integrate


def test_second_order_pole():
    assert fp(1/x**2, x, -1, 1) == -2
    assert fp(1/(x - 1)**2, x, 0, 2) == -2
    assert fp((x + 1)/x**2, x, 1, 2) == log(2) + S.Half
    assert fp((x + 1)/(x - 1)**2, x, 0, 2) == -4


def test_third_order_pole():
    assert fp(1/x**3, x, -1, 1) == 0
    assert fp(1/x**3, x, -1, 2) == S(3)/8
    assert fp(1/(x - 1)**3, x, 0, 3) == S(3)/8


def test_multiple_poles():
    assert fp(1/x**2 + 1/(x - 1)**3, x, -1, 2) == S(-15)/8
    assert fp(1/x + 1/(x - 1)**2, x, -1, 2) == log(2) - S(3)/2


def test_simple_pole_matches_principal_value():
    assert fp(1/x, x, -1, 2) == log(2)
    assert fp(1/(x - 1), x, 0, 3) == log(2)


def test_endpoint_poles_require_opt_in():
    raises(UnsupportedSingularityError, lambda: fp(1/x**2, x, 0, 1))
    raises(UnsupportedSingularityError, lambda: fp(1/(x - 1)**2, x, 0, 1))
    try:
        fp(1/x**2, x, 0, 1)
    except UnsupportedSingularityError as err:
        assert err.pole == 0
    else:
        raise AssertionError('expected UnsupportedSingularityError')


def test_endpoint_poles_one_sided():
    assert fp(1/x**2, x, 0, 1, allow_endpoint_poles=True) == -1
    assert fp(1/(x - 1)**2, x, 0, 1, allow_endpoint_poles=True) == -1
    assert fp(1/x, x, 0, 2, allow_endpoint_poles=True) == log(2)
    assert fp(1/(x - 2), x, 0, 2, allow_endpoint_poles=True) == -log(2)
    assert fp(1/(x*(x - 2)), x, 0, 2, allow_endpoint_poles=True) == -log(2)
    assert fp(1/(x**2*(x - 1)), x, 0, 1, allow_endpoint_poles=True) == 1


def test_algebraic_poles():
    res = fp(1/(x**2 - 2), x, 0, 2)
    expected = sqrt(2)*(log(2 - sqrt(2)) - log(2 + sqrt(2)))/4
    assert simplify(res - expected) == 0

    res = fp(1/(x**2 - 2)**2, x, 0, 2)
    expected = (-1/(8*(2 - sqrt(2))) - 1/(8*(2 + sqrt(2)))
        - sqrt(2)*(log(2 - sqrt(2)) - log(sqrt(2)))/16
        + sqrt(2)*(log(2 + sqrt(2)) - log(sqrt(2)))/16)
    assert simplify(res - expected) == 0


def test_regular_cases():
    assert fp(x**2, x, 0, 1) == S(1)/3
    assert fp(x + 1/(x - 3), x, 0, 2) == 2 - log(3)
    assert fp((x**2 - 1)/(x - 1), x, 0, 2) == 4
    assert simplify(fp(1/(x**2 + 1), x, 0, 1) - pi/4) == 0


def test_algebraic_pole_rootof():
    # Real algebraic poles that have no radical expression are isolated
    # exactly as RootOf objects.
    res = fp(1/(x**5 - x - 1), x, 0, 2)
    assert abs(complex(res.evalf(30)) + 0.7445139795701202) < 1e-15
    res = fp((x**2 + 1)/(x**5 - x - 1)**2, x, 0, 2)
    assert abs(complex(res.evalf(30)) - 0.48680457104434974) < 1e-15


def test_invalid_input():
    raises(ValueError, lambda: fp(sin(x), x, 0, 1))
    raises(ValueError, lambda: fp(1/(x - sqrt(2)), x, 0, 1))
    raises(ValueError, lambda: fp(1/x, x, sqrt(2), 2))
    raises(ValueError, lambda: fp(1/x, x, 2, 1))


def _truncated_fp_numeric(f, a, b, allow_endpoint_poles=False, dps=60):
    """Numerical finite part: integrate over [a, b] minus eps-neighbourhoods
    of the poles at high precision and subtract the divergent terms
    computed from the exact Laurent coefficients."""
    mp.dps = dps
    fc = cancel(f)
    den = fc.as_numer_denom()[1]
    poles = []
    for r in real_roots(den, x):
        if (a < r) and (r < b):
            pos = 'interior'
        elif allow_endpoint_poles and r == a:
            pos = 'left'
        elif allow_endpoint_poles and r == b:
            pos = 'right'
        else:
            continue
        for i, (r0, m, _) in enumerate(poles):
            if r0 == r:
                poles[i] = (r0, m + 1, pos)
                break
        else:
            poles.append((r, 1, pos))
    maxorder = max([m for _, m, _ in poles], default=1)
    eps = mp.mpf(10)**(-dps//(maxorder + 2))

    fnum = lambdify(x, fc, 'mpmath')
    cuts = [mp.mpf(S(a).evalf(dps))]
    for r, m, pos in sorted(poles, key=lambda p: p[0].evalf()):
        rm = mp.mpf(r.evalf(dps))
        if pos == 'interior':
            cuts += [rm - eps, rm + eps]
    cuts.append(mp.mpf(S(b).evalf(dps)))
    if any(pos == 'left' for _, _, pos in poles):
        cuts[0] += eps
    if any(pos == 'right' for _, _, pos in poles):
        cuts[-1] -= eps

    total = mp.mpf(0)
    for i in range(0, len(cuts) - 1, 2):
        total += quad(fnum, [cuts[i], cuts[i + 1]])

    num, den = fc.as_numer_denom()
    for r, m, pos in poles:
        for k, ak in _laurent_coefficients(num, den, r, m, x).items():
            ak = mp.mpf(ak.evalf(dps))
            if pos == 'interior':
                if k >= 2:
                    total -= ak*eps**(1 - k)*((-1)**(1 - k) - 1)/(1 - k)
            elif pos == 'left':
                if k == 1:
                    total -= -ak*mp.log(eps)
                else:
                    total -= -ak*eps**(1 - k)/(1 - k)
            else:
                if k == 1:
                    total -= ak*mp.log(eps)
                else:
                    total -= ak*(-eps)**(1 - k)/(1 - k)
    return total


def _check_numeric(f, a, b, allow_endpoint_poles=False, tol='1e-8'):
    exact = fp(f, x, a, b, allow_endpoint_poles=allow_endpoint_poles)
    dps = 60
    mp.dps = dps
    numeric = _truncated_fp_numeric(f, a, b,
        allow_endpoint_poles=allow_endpoint_poles, dps=dps)
    re, im = exact.evalf(dps).as_real_imag()
    exact_mp = mp.mpc(mp.mpf(re), mp.mpf(im))
    assert mp.fabs(numeric - exact_mp) < mp.mpf(tol)


def test_numeric_crosscheck():
    _check_numeric(1/x**2, -1, 1)
    _check_numeric(1/x**3, -1, 2)
    _check_numeric((x + 1)/(x - 1)**2, 0, 2)
    _check_numeric(1/x**2 + 1/(x - 1)**3, -1, 2)
    _check_numeric(1/(x**2 - 2), 0, 2)
    _check_numeric(1/(x**2 - 2)**2, 0, 2)
    _check_numeric(1/(x**5 - x - 1), 0, 2)
    _check_numeric((x**2 + 1)/(x**5 - x - 1)**2, 0, 2)
    _check_numeric(1/(x**2*(x - 1)), 0, 1, allow_endpoint_poles=True)
    _check_numeric(1/(x*(x - 2)), 0, 2, allow_endpoint_poles=True)
