"""Hadamard finite part integrals of rational functions.

This module implements the Hadamard finite part of definite integrals of
rational functions over a real interval that may contain poles of arbitrary
order.  The finite part is defined by truncating the integration domain
around each singularity, expanding the result in the truncation parameter,
and taking the constant (divergence-free) part.  It is neither an ordinary
integral nor, except for simple poles where the two notions coincide, a
Cauchy principal value.
"""
from __future__ import annotations

from sympy.core import S, Symbol, sympify
from sympy.core.numbers import I, pi
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.integrals.integrals import integrate
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (CoercionFailed, GeneratorsNeeded,
                                    PolynomialError)
from sympy.polys.polytools import Poly, real_roots

__all__ = ['hadamard_finite_part_integral']


def _basis_antiderivative(k, t):
    # Antiderivative of t**(-k) used in the finite part formulas.
    if k == 1:
        return log(Abs(t))
    return -t**(1 - k)/(k - 1)


def _fp_basis_integral(r, k, a, b):
    # Hadamard finite part of the integral of (x - r)**(-k) over [a, b]
    # where r is a point of [a, b].  The finite part of the antiderivative
    # at the singular endpoint is zero because its expansion there contains
    # only divergent power or logarithmic terms and no constant term.
    upper = S.Zero if r == b else _basis_antiderivative(k, b - r)
    lower = S.Zero if r == a else _basis_antiderivative(k, a - r)
    return upper - lower


def _principal_part_coeffs(N, D, r, m, x):
    # Coefficients A_k of the principal part sum A_k/(x - r)**k, k = 1..m,
    # of the rational function N/D at the pole r of order m.  Computed by
    # exact polynomial arithmetic over the algebraic field QQ(r): after
    # shifting x -> x + r the coefficients are the first m terms of the
    # power series quotient N(x + r)/Q(x + r) with D = (x - r)**m*Q.
    K = QQ if r.is_Rational else QQ.algebraic_field(r)
    Np = Poly(N, x, domain=K)
    Qp = Poly(D, x, domain=K)
    t = Poly(x - r, x, domain=K)
    for _ in range(m):
        Qp = Qp.exquo(t)
    nc = [K.to_sympy(K.convert(c)) for c in Np.shift(r).all_coeffs()][::-1]
    qc = [K.to_sympy(K.convert(c)) for c in Qp.shift(r).all_coeffs()][::-1]
    cs = []
    for k in range(m):
        s = nc[k] if k < len(nc) else S.Zero
        for j in range(k):
            s -= cs[j]*(qc[k - j] if k - j < len(qc) else S.Zero)
        cs.append(s/qc[0])
    return {k: cs[m - k] for k in range(1, m + 1)}


def hadamard_finite_part_integral(f, x, a, b, allow_endpoint_poles=False):
    r"""Hadamard finite part of the integral of a rational function.

    Computes the Hadamard finite part

    .. math::

        \mathrm{p.f.} \int_a^b f(x)\, dx

    where `f` is a rational function of `x` with rational coefficients and
    `[a, b]` is a real interval with rational endpoints that may contain
    poles of `f` of arbitrary order.  Interior poles may be rational or
    real algebraic numbers; they are isolated exactly.

    The finite part is defined by removing an `\varepsilon`-neighborhood
    of every pole, expanding the truncated integral in `\varepsilon`, and
    taking the constant term after subtracting all divergent (power and
    logarithmic) terms.  For a pole of order two or higher this differs
    from any principal value prescription.  If a pole coincides with an
    endpoint the one-sided finite part is used, but only when
    ``allow_endpoint_poles=True`` is passed explicitly.

    Parameters
    ==========

    f : Expr
        Rational function of `x` with rational coefficients.
    x : Symbol
        Integration variable.
    a, b : Rational
        Endpoints of the integration interval with ``a < b``.
    allow_endpoint_poles : bool, optional
        Whether poles located exactly at ``a`` or ``b`` are accepted
        and treated with the one-sided finite part definition.  Default is
        ``False``.

    Returns
    =======

    Expr
        The exact finite part of the integral.

    Raises
    ======

    ValueError
        If `f` is not a rational function of `x` with rational
        coefficients, if the endpoints are not rational, or if a pole
        coincides with an endpoint while ``allow_endpoint_poles`` is
        ``False``.  The location of the offending singularity is
        included in the message.

    Examples
    ========

    >>> from sympy.integrals.finitepart import hadamard_finite_part_integral
    >>> from sympy.abc import x

    A second order pole inside the interval:

    >>> hadamard_finite_part_integral(1/x**2, x, -1, 1)
    -2

    A third order pole:

    >>> hadamard_finite_part_integral(1/x**3, x, -1, 2)
    3/8

    A pole at an endpoint requires an explicit opt-in and is then treated
    with the one-sided definition:

    >>> hadamard_finite_part_integral(1/x**2, x, 0, 1)
    Traceback (most recent call last):
    ...
    ValueError: pole of order 2 at the endpoint 0; pass
    allow_endpoint_poles=True to use the one-sided finite part
    >>> hadamard_finite_part_integral(1/x**2, x, 0, 1,
    ...                               allow_endpoint_poles=True)
    -1

    References
    ==========

    .. [1] J. Hadamard, "Lectures on Cauchy's Problem in Linear Partial
           Differential Equations", Yale University Press, 1923.
    .. [2] https://en.wikipedia.org/wiki/Hadamard_regularization
    """
    x = sympify(x)
    if not isinstance(x, Symbol):
        raise ValueError("the integration variable must be a Symbol")
    a, b = sympify(a), sympify(b)
    if not (a.is_Rational and b.is_Rational):
        raise ValueError(
            "the endpoints must be rational numbers, got %s and %s" % (a, b))
    if not a < b:
        raise ValueError("expected a < b, got %s and %s" % (a, b))

    f = sympify(f)
    num, den = f.as_numer_denom()
    try:
        N = Poly(num, x, domain=QQ)
        D = Poly(den, x, domain=QQ)
    except (PolynomialError, GeneratorsNeeded, CoercionFailed):
        raise ValueError(
            "expected a rational function of %s with rational "
            "coefficients, got %s" % (x, f))
    if D.is_zero:
        raise ValueError("the denominator of %s vanishes identically" % f)
    g = N.gcd(D)
    N, D = N.exquo(g), D.exquo(g)
    f = N.as_expr()/D.as_expr()

    poles = {}
    for fac, mult in D.factor_list()[1]:
        for r in real_roots(fac):
            if bool(a <= r) and bool(r <= b):
                poles[r] = mult

    for r, m in poles.items():
        if (r == a or r == b) and not allow_endpoint_poles:
            raise ValueError(
                "pole of order %d at the endpoint %s; pass "
                "allow_endpoint_poles=True to use the one-sided "
                "finite part" % (m, r))

    endpoint_poles = {r: m for r, m in poles.items() if r == a or r == b}
    interior_poles = {r: m for r, m in poles.items() if r != a and r != b}

    # Poles at the (rational) endpoints have rational principal parts.
    # Subtracting them leaves a rational function over QQ that is regular
    # at both endpoints, so its antiderivative can be evaluated there.
    endpoint_parts = {}
    for r, m in endpoint_poles.items():
        endpoint_parts[r] = _principal_part_coeffs(N, D, r, m, x)
    if endpoint_parts:
        f = f - sum(
            A/(x - r)**k
            for r, coeffs in endpoint_parts.items()
            for k, A in coeffs.items())
        f = f.cancel()

    # For an antiderivative F with real-valued logarithms the finite part
    # over [a, b] equals F(b) - F(a): the divergent terms of F at interior
    # poles carry no constant term and the analytic parts cancel between
    # the two sides of every pole.  SymPy's complex logarithms pick up a
    # spurious I*pi at each interior real pole crossed, which is corrected
    # by adding I*pi times the residue at each such pole.
    F = integrate(f, x)
    result = F.subs(x, b) - F.subs(x, a)
    for r, m in interior_poles.items():
        residue = _principal_part_coeffs(N, D, r, m, x)[1]
        if residue:
            result += I*pi*residue

    for r, coeffs in endpoint_parts.items():
        for k, A in coeffs.items():
            result += A*_fp_basis_integral(r, k, a, b)
    return result
