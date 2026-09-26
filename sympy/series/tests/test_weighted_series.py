from sympy.core import S, Rational, symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.testing.pytest import raises

from sympy.series.weighted_series import (
    WeightedSeries, SeriesTruncationError)

x, y, z = symbols('x y z')


def monoms_below(weights, order):
    result = []

    def rec(i, remaining, current):
        if i == len(weights):
            result.append(tuple(current))
            return
        k = 0
        while k * weights[i] < remaining:
            current.append(k)
            rec(i + 1, remaining - k * weights[i], current)
            current.pop()
            k += 1

    rec(0, order, [])
    return result


def assert_same_known_part(f, g, order=None):
    assert f.gens == g.gens and f.weights == g.weights
    if order is None:
        order = min(f.order, g.order)
    for monom in monoms_below(f.weights, order):
        assert f.coefficient(monom) == g.coefficient(monom), monom


def test_add_mul_exact_coefficients():
    gens, weights = (x, y, z), (1, 2, 3)
    f = WeightedSeries.from_expr(1 + 2*x + 3*y + x**2*z + Rational(1, 3)*y**2,
                                 gens, weights, 12)
    g = WeightedSeries.from_expr(2 - x + y*z + z**2, gens, weights, 12)
    h = f + g
    assert h.order == 12
    assert_same_known_part(h, WeightedSeries.from_expr(
        3 + x + 3*y + x**2*z + Rational(1, 3)*y**2 + y*z + z**2,
        gens, weights, 12))
    p = f * g
    assert p.order == 12
    expected = WeightedSeries.from_expr(
        (1 + 2*x + 3*y + x**2*z + Rational(1, 3)*y**2)
        * (2 - x + y*z + z**2), gens, weights, 12)
    assert_same_known_part(p, expected)
    # terms at or above the weighted order are simply not computed
    assert p.coefficient((0, 0, 2)) == 1  # z**2 has degree 6 < 12
    raises(SeriesTruncationError, lambda: p.coefficient((0, 0, 4)))
    # scalar arithmetic
    assert (2 * f).coefficient((1, 0, 0)) == 4
    assert (f + 1).coefficient((0, 0, 0)) == 2
    assert (f - 1).coefficient((0, 0, 0)) == 0
    q = f - f
    assert q.is_zero and q.order == 12


def test_mixed_orders_min_wins():
    gens, weights = (x, y), (1, 1)
    f = WeightedSeries.from_expr(1 + x + y, gens, weights, 6)
    g = WeightedSeries.from_expr(1 - x, gens, weights, 3)
    p = f * g
    assert p.order == 3
    assert p.coefficient((1, 0)) == 0
    raises(SeriesTruncationError, lambda: p.coefficient((2, 1)))


def test_inverse_exact_coefficients():
    gens, weights = (x, y), (1, 1)
    n = 8
    f = WeightedSeries.from_expr(1 + x + y, gens, weights, n)
    finv = f.inverse()
    assert finv.order == n
    # coefficient of x**a*y**b in 1/(1+x+y) is (-1)**(a+b)*(a+b)!/(a!*b!)
    for monom in monoms_below(weights, n):
        a, b = monom
        assert finv.coefficient(monom) ==             (-1)**(a + b) * factorial(a + b) / (factorial(a) * factorial(b))
    one = f * finv
    assert one.order == n
    assert one.coefficient((0, 0)) == 1
    for monom in monoms_below(weights, n):
        if monom != (0, 0):
            assert one.coefficient(monom) == 0
    # non-unit constant term
    g = WeightedSeries.from_expr(2 + x, gens, weights, 5)
    assert (g * g.inverse()).coefficient((0, 0)) == 1
    # non-invertible series
    h = WeightedSeries.from_expr(x + y, gens, weights, 5)
    raises(ValueError, lambda: h.inverse())


def test_exp_log_identities():
    gens, weights = (x, y, z), (1, 2, 3)
    n = 10
    f = WeightedSeries.from_expr(x + y + x*y, gens, weights, n)
    g = WeightedSeries.from_expr(z + x**2, gens, weights, n)
    # log(exp(f)) == f up to the guaranteed order
    e = f.exp()
    assert e.order == n
    assert e.coefficient((0, 0, 0)) == 1
    loge = e.log()
    assert loge.order == n
    assert_same_known_part(loge, f)
    # exp(f + g) == exp(f) * exp(g)
    left = (f + g).exp()
    right = f.exp() * g.exp()
    assert_same_known_part(left, right)
    # exp(log(1 + h)) == 1 + h
    h = WeightedSeries.from_expr(1 + x + z, gens, weights, n)
    assert_same_known_part(h.log().exp(), h)
    # exact coefficients of exp(x + y) with weights (1, 1)
    gens2, weights2 = (x, y), (1, 1)
    u = WeightedSeries.from_expr(x + y, gens2, weights2, 6).exp()
    for a, b in monoms_below(weights2, 6):
        assert u.coefficient((a, b)) == 1 / (factorial(a) * factorial(b))
    # log of series with zero constant term is rejected
    raises(ValueError, lambda: f.log())


def test_diff_lowers_order_by_weight():
    gens, weights = (x, y, z), (1, 2, 3)
    f = WeightedSeries.from_expr(x**2*y + y**2 + z**2 + x + 1,
                                 gens, weights, 10)
    df = f.diff(y)
    assert df.order == 8
    assert df.to_expr() == x**2 + 2*y
    dz = f.diff(z)
    assert dz.order == 7
    assert dz.to_expr() == 2*z
    dx = f.diff(x)
    assert dx.order == 9
    assert dx.to_expr() == 2*x*y + 1
    # differentiating past the constant term leaves nothing known
    d = f.diff(z).diff(z).diff(z).diff(z)
    assert d.order == 0
    assert d.is_zero
    raises(SeriesTruncationError, lambda: d.coefficient((0, 0, 0)))


def test_compose_zero_constant_term():
    gens, weights = (x, y), (1, 1)
    n = 6
    f = WeightedSeries.from_expr(1 + x + 2*y + x*y + x**2, gens, weights, n)
    gx = WeightedSeries.from_expr(x + x**2, gens, weights, n)
    gy = WeightedSeries.from_expr(y - x, gens, weights, n)
    r = f.compose({x: gx, y: gy})
    assert r.order == n
    expected = WeightedSeries.from_expr(
        1 + (x + x**2) + 2*(y - x) + (x + x**2)*(y - x) + (x + x**2)**2,
        gens, weights, n)
    assert_same_known_part(r, expected)
    # nonzero constant term in a substituted series is rejected
    bad = WeightedSeries.from_expr(1 + x, gens, weights, n)
    raises(ValueError, lambda: f.compose({x: bad}))


def test_compose_leading_degree_changes_order():
    # substituting a variable by a higher-order series raises the
    # guaranteed order; substituting by a lower-weight one lowers it
    f = WeightedSeries.from_expr(x + x**2, (x,), (2,), 10)
    g = WeightedSeries.from_expr(x**2, (x,), (2,))  # leading degree 4
    r = f.compose({x: g})
    assert r.order == 20  # ceiling(10 * 4/2)
    assert r.coefficient((2,)) == 1
    assert r.coefficient((4,)) == 1
    # finite order of the substituted series caps the result
    g2 = WeightedSeries.from_expr(x, (x,), (2,), 6)
    r2 = f.compose({x: g2})
    assert r2.order == 6
    # substituting a weight-3 variable by a weight-1 one deflates the order
    gens, weights = (x, y), (1, 3)
    h = WeightedSeries.from_expr(1 + y + x**3, gens, weights, 12)
    r3 = h.compose({y: WeightedSeries.var(gens, weights, 0)})
    assert r3.order == 4  # ceiling(12 * 1/3); cannot pretend to know 12
    assert_same_known_part(r3, WeightedSeries.from_expr(
        1 + x + x**3, gens, weights, 4))
    raises(SeriesTruncationError, lambda: r3.coefficient((0, 2)))


def test_compose_cancellation_keeps_honest_order():
    gens, weights = (x, y), (1, 1)
    n = 6
    f = WeightedSeries.from_expr(x - y + x**2, gens, weights, n)
    g = WeightedSeries.from_expr(y + y**2, gens, weights, n)
    r = f.compose({x: g, y: g})
    # the leading terms cancel: the computed coefficient is exactly zero
    assert r.order == n
    assert r.coefficient((0, 1)) == 0
    assert r.coefficient((0, 2)) == 1
    assert r.coefficient((0, 3)) == 2
    assert r.coefficient((0, 4)) == 1
    # but degrees at or beyond n are still unknown, not zero
    raises(SeriesTruncationError, lambda: r.coefficient((0, 6)))


def test_coefficient_zero_vs_not_computed():
    gens, weights = (x, y), (1, 2)
    f = WeightedSeries.from_expr(1 + x, gens, weights, 4)
    assert f.coefficient((0, 1)) == 0  # computed zero (degree 2 < 4)
    assert f.coefficient((3, 0)) == 0  # computed zero (degree 3 < 4)
    raises(SeriesTruncationError, lambda: f.coefficient((0, 2)))
    raises(SeriesTruncationError, lambda: f.coefficient((4, 0)))
    exact = WeightedSeries.from_expr(1 + x, gens, weights)
    assert exact.coefficient((5, 5)) == 0  # exact series: everything known


def test_truncate_cannot_raise_order():
    gens, weights = (x,), (1,)
    f = WeightedSeries.from_expr(1 + x + x**2, gens, weights, 2)
    assert f.truncate(1).to_expr() == 1
    raises(ValueError, lambda: f.truncate(5))


def test_from_expr_to_expr_roundtrip():
    gens, weights = (x, y), (2, 3)
    expr = 1 + x**2 + 3*y - x*y
    f = WeightedSeries.from_expr(expr, gens, weights)
    assert f.to_expr() == expr
    assert f.order is S.Infinity
    t = WeightedSeries.from_expr(expr, gens, weights, 6)
    assert t.to_expr() == 1 + x**2 + 3*y - x*y  # x*y has degree 5 < 6
    t7 = WeightedSeries.from_expr(expr, gens, weights, 5)
    assert t7.to_expr() == 1 + x**2 + 3*y
    raises(ValueError, lambda: WeightedSeries.from_expr(1/x, gens, weights))


def test_univariate_series_interface_unchanged():
    from sympy import series, O, sin
    s = series(sin(x), x, 0, 6)
    assert s == x - x**3/6 + x**5/120 + O(x**6)
