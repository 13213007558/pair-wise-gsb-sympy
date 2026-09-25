"""Hadamard finite part integration of rational functions.

This module provides :func:`hadamard_finite_part_integrate`, an entry
point for evaluating Hadamard finite part integrals of rational
functions over a real interval that may contain poles of arbitrary
order in its interior (or, on explicit request, at its endpoints).

The finite part is defined by removing symmetric (or one-sided, at an
endpoint) neighbourhoods of each pole and discarding the divergent
terms of the resulting expansion, keeping the constant term.  It is
not an ordinary integral and, except for simple poles, it is not a
Cauchy principal value either.
"""
from __future__ import annotations

from sympy.core import Dummy, I, S, pi, sympify
from sympy.core.symbol import Symbol
from sympy.functions import log
from sympy.polys import Poly, PolynomialError, cancel, real_roots
from sympy.series import series
from sympy.integrals.integrals import integrate
from sympy.integrals.rationaltools import ratint

__all__ = ['UnsupportedSingularityError', 'hadamard_finite_part_integrate']


class UnsupportedSingularityError(ValueError):
    """Raised when a singularity cannot be handled by finite part
    integration.

    The ``pole`` attribute gives the location of the offending
    singularity.
    """

    def __init__(self, pole, message):
        self.pole = pole
        super().__init__(message)


def _laurent_coefficients(num, den, r, m, x):
    r"""Return the coefficients ``{k: A_k}`` of the singular part
    ``\sum_{k=1}^{m} A_k (x - r)^{-k}`` of ``num/den`` at its pole
    ``r`` of order ``m``.

    The coefficients are exact algebraic numbers.
    """
    if r.is_rational:
        s = series(num/den, x, r, 0).removeO()
        return {k: s.coeff(x - r, -k) for k in range(1, m + 1)}
    # Exact arithmetic in the algebraic extension Q(r): write
    # den = (x - r)**m*h and expand num(x)/h(x) around x = r.
    denK = Poly(den, x, extension=r)
    facK = Poly((x - r)**m, x, extension=r)
    hK, rem = denK.div(facK)
    numK = Poly(num, x, extension=r)
    t = Dummy('t')
    numt = Poly(numK.as_expr().subs(x, t + r), t, extension=r)
    ht = Poly(hK.as_expr().subs(x, t + r), t, extension=r)
    h0 = ht.nth(0)
    coeffs = []
    for j in range(m):
        cj = numt.nth(j)
        for i in range(j):
            cj = cj - coeffs[i]*ht.nth(j - i)
        coeffs.append(cj/h0)
    return {m - j: coeffs[j] for j in range(m)}


def hadamard_finite_part_integrate(f, x, a, b, allow_endpoint_poles=False):
    r"""Compute the Hadamard finite part of the integral of a rational
    function over a real interval that may contain poles.

    Explanation
    ===========

    For a rational function f with a pole of order m at a point c in
    (a, b) the ordinary integral of f over [a, b] diverges.  Its
    Hadamard finite part is defined by excising a symmetric
    neighbourhood (c - eps, c + eps) of the pole and keeping the
    constant term of the expansion of

    .. math::

        I(\varepsilon) = \int_a^{c-\varepsilon} f(x)\,dx
            + \int_{c+\varepsilon}^b f(x)\,dx
            = \sum_{j} d_j\,\varepsilon^{-j} + \mathrm{F.p.} + o(1),

    i.e. the divergent terms are subtracted and the finite constant
    part is the result.  When the pole sits at an endpoint the
    neighbourhood is one-sided and the same "subtract the divergent
    terms" prescription applies; this is only done when
    ``allow_endpoint_poles=True`` is passed explicitly.  For a simple
    pole in the interior the finite part coincides with the Cauchy
    principal value, but for higher order poles it does not.

    Parameters
    ==========

    f : Expr
        Rational function of ``x`` with rational coefficients.
    x : Symbol
        Integration variable.
    a, b : Rational
        Endpoints of the interval with ``a < b``.
    allow_endpoint_poles : bool, optional
        Must be set to ``True`` to allow poles exactly at ``a`` or
        ``b``, which are then treated with the one-sided finite part
        definition.  Default is ``False``.

    Returns
    =======

    Expr
        The exact finite part of the integral.

    Raises
    ======

    UnsupportedSingularityError
        If a pole lies at an endpoint and ``allow_endpoint_poles`` is
        ``False``, or if a pole cannot be located exactly; the
        ``pole`` attribute of the exception gives its location.
    ValueError
        If ``f`` is not a rational function of ``x`` with rational
        coefficients, or the endpoints are not rational.

    Examples
    ========

    >>> from sympy.integrals.finitepart import hadamard_finite_part_integrate
    >>> from sympy.abc import x

    A second order pole in the interior:

    >>> hadamard_finite_part_integrate(1/x**2, x, -1, 1)
    -2

    A third order pole:

    >>> hadamard_finite_part_integrate(1/x**3, x, -1, 2)
    3/8

    Poles at the endpoints require an explicit opt-in and are treated
    with the one-sided definition:

    >>> hadamard_finite_part_integrate(1/x**2, x, 0, 1,
    ...     allow_endpoint_poles=True)
    -1

    Interior poles may be real algebraic numbers:

    >>> from sympy import simplify, log, sqrt
    >>> hadamard_finite_part_integrate(1/(x**2 - 2), x, 0, 2)
    -sqrt(2)*log(sqrt(2) + 2)/4 + sqrt(2)*log(2 - sqrt(2))/4

    References
    ==========

    .. [1] J. Hadamard, "Lectures on Cauchy's Problem in Linear Partial
           Differential Equations", Yale University Press, 1923.
    .. [2] https://en.wikipedia.org/wiki/Hadamard_regularization
    """
    x = sympify(x)
    if not isinstance(x, Symbol):
        raise ValueError('the integration variable must be a Symbol')
    a = sympify(a)
    b = sympify(b)
    if not (a.is_rational and b.is_rational):
        raise ValueError('the endpoints must be rational numbers')
    if not (a < b):
        raise ValueError('expected a < b')

    f = sympify(f)
    if f.free_symbols - {x}:
        raise ValueError('f must be a function of %s only' % x)
    f = cancel(f)
    num, den = f.as_numer_denom()
    try:
        pnum = Poly(num, x)
        pden = Poly(den, x)
    except PolynomialError:
        raise ValueError('f must be a rational function of %s' % x)
    for poly in (pnum, pden):
        if not (poly.get_domain().is_ZZ or poly.get_domain().is_QQ):
            raise ValueError(
                'the coefficients of f must be rational numbers')

    # Locate the real poles of f in [a, b] exactly.
    interior = {}
    endpoints = {}
    for r in real_roots(den, x) if pden.degree() > 0 else []:
        if r == a or r == b:
            where = endpoints
        else:
            gt = a < r
            lt = r < b
            if gt is S.true and lt is S.true:
                where = interior
            elif gt is S.false or lt is S.false:
                continue
            else:
                raise UnsupportedSingularityError(
                    r, 'cannot locate the pole at x = %s relative to '
                    'the interval' % r)
        where[r] = where.get(r, 0) + 1

    if endpoints and not allow_endpoint_poles:
        r = next(iter(endpoints))
        raise UnsupportedSingularityError(
            r, 'pole at the endpoint x = %s; pass '
            'allow_endpoint_poles=True to use the one-sided finite '
            'part' % r)

    result = S.Zero
    g = f

    # Endpoint poles: subtract their (rational) singular parts and add
    # the one-sided finite part of each singular term.
    for r, m in endpoints.items():
        gnum, gden = g.as_numer_denom()
        lc = _laurent_coefficients(gnum, gden, r, m, x)
        singular = S.Zero
        for k, ak in lc.items():
            singular += ak/(x - r)**k
            if r == a:
                if k == 1:
                    result += ak*log(b - a)
                else:
                    result += ak*(b - a)**(1 - k)/(1 - k)
            else:
                if k == 1:
                    result += -ak*log(b - a)
                else:
                    result += ak*(a - b)**(1 - k)/(k - 1)
        g = cancel(g - singular)

    if interior:
        # The finite part equals F(b) - F(a) + I*pi*sum(residues) for
        # any antiderivative F of g: the divergent terms cancel in the
        # symmetric limit and each interior pole contributes an
        # I*pi*A_1 jump of the antiderivative's logarithms.
        F = ratint(g, x)
        fp_g = F.subs(x, b) - F.subs(x, a)
        gnum, gden = g.as_numer_denom()
        for r, m in interior.items():
            fp_g += I*pi*_laurent_coefficients(gnum, gden, r, m, x)[1]
        result += fp_g.expand()
    else:
        result += integrate(g, (x, a, b))

    return result
