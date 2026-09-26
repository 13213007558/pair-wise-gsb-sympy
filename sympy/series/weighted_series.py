"""Multivariate truncated power series with weighted total degree.

This module provides :class:`WeightedSeries`, a truncated multivariate
power series where each generator carries a positive integer weight and
only terms whose weighted total degree is strictly less than the series
order are retained.  Unlike expanding variable by variable and clipping
afterwards, truncation is tracked directly in the weighted grading, and
every operation computes the largest order that can still be guaranteed
for the result.  In particular, differentiation lowers the known order
and substitution propagates the orders and leading degrees of the
substituted series instead of silently reusing the input order.
"""

from __future__ import annotations

from sympy.core import S, Rational, Symbol, sympify
from sympy.functions.elementary.exponential import exp as _exp, log as _log
from sympy.functions.elementary.integers import ceiling


class SeriesTruncationError(ValueError):
    """Raised when a coefficient beyond the guaranteed order is requested."""


def _check_gens_weights(gens, weights):
    gens = tuple(gens)
    weights = tuple(weights)
    if len(gens) != len(weights):
        raise ValueError("gens and weights must have the same length")
    if not gens:
        raise ValueError("at least one generator is required")
    for w in weights:
        w = sympify(w)
        if not (w.is_Integer and w.is_positive):
            raise ValueError("weights must be positive integers")
    return gens, tuple(int(w) for w in weights)


def _weighted_degree(weights, monom):
    return sum(w * m for w, m in zip(weights, monom))


class WeightedSeries:
    """A multivariate power series truncated by weighted total degree.

    Parameters
    ==========

    gens : tuple of symbols
        The generators of the series ring.
    weights : tuple of positive integers
        The weight of each generator.  The weighted degree of a monomial
        ``x1**m1 * ... * xk**mk`` is ``w1*m1 + ... + wk*mk``.
    terms : dict, optional
        Maps monomial exponent tuples to exact coefficients.
    order : nonnegative integer or oo
        Terms of weighted degree strictly less than ``order`` are known
        exactly; terms of degree ``>= order`` have not been computed.
        ``oo`` (the default) means the series is an exact polynomial.

    A coefficient that is absent from ``terms`` is exactly zero.  Use
    :meth:`coefficient` to tell a computed zero apart from a coefficient
    that was never computed: the former returns ``0`` while the latter
    raises :class:`SeriesTruncationError`.
    """

    def __init__(self, gens, weights, terms=None, order=S.Infinity):
        self.gens, self.weights = _check_gens_weights(gens, weights)
        order = sympify(order)
        if not (order is S.Infinity or (order.is_Integer and order >= 0)):
            raise ValueError("order must be a nonnegative integer or oo")
        self.order = order
        self._terms = {}
        if terms:
            for monom, coeff in dict(terms).items():
                monom = tuple(int(m) for m in monom)
                if len(monom) != len(self.gens):
                    raise ValueError("monomial has wrong length")
                coeff = sympify(coeff)
                if coeff == 0:
                    continue
                if (order is not S.Infinity
                        and _weighted_degree(self.weights, monom) >= order):
                    continue
                self._terms[monom] = self._terms.get(monom, S.Zero) + coeff
            self._terms = {m: c for m, c in self._terms.items() if c != 0}

    @classmethod
    def zero(cls, gens, weights, order=S.Infinity):
        """The zero series, known up to the given order."""
        return cls(gens, weights, {}, order)

    @classmethod
    def constant(cls, gens, weights, coeff):
        """An exact constant series."""
        coeff = sympify(coeff)
        terms = {} if coeff == 0 else {(0,) * len(tuple(gens)): coeff}
        return cls(gens, weights, terms)

    @classmethod
    def one(cls, gens, weights):
        """The exact multiplicative identity."""
        return cls.constant(gens, weights, 1)

    @classmethod
    def var(cls, gens, weights, i):
        """The ``i``-th generator as an exact series."""
        gens, weights = _check_gens_weights(gens, weights)
        monom = [0] * len(gens)
        monom[i] = 1
        return cls(gens, weights, {tuple(monom): 1})

    @classmethod
    def from_expr(cls, expr, gens, weights, order=S.Infinity):
        """Build a series from a polynomial expression, truncated at ``order``."""
        poly = sympify(expr).expand().as_poly(*gens)
        if poly is None:
            raise ValueError("expected a polynomial in the given generators")
        return cls(gens, weights, dict(poly.terms()), order)

    @property
    def terms(self):
        """A copy of the dict mapping monomial tuples to coefficients."""
        return dict(self._terms)

    @property
    def is_zero(self):
        """True if no term below the order is nonzero.

        The series is exactly zero only if the order is ``oo``; otherwise
        it is zero up to the truncation order.
        """
        return not self._terms

    def weighted_degree(self, monom):
        """The weighted degree of a monomial exponent tuple."""
        return _weighted_degree(self.weights, tuple(int(m) for m in monom))

    @property
    def leading_degree(self):
        """The smallest weighted degree of a known term (``oo`` if none)."""
        if not self._terms:
            return S.Infinity
        return min(self.weighted_degree(m) for m in self._terms)

    def coefficient(self, monom):
        """The exact coefficient of a monomial.

        Returns ``0`` if the coefficient was computed and is zero.
        Raises :class:`SeriesTruncationError` if the monomial's weighted
        degree is not below the series order, i.e. the coefficient was
        never computed.
        """
        monom = tuple(int(m) for m in monom)
        if len(monom) != len(self.gens):
            raise ValueError("monomial has wrong length")
        if (self.order is not S.Infinity
                and self.weighted_degree(monom) >= self.order):
            raise SeriesTruncationError(
                "coefficient of weighted degree %s was not computed "
                "(series order is %s)" % (self.weighted_degree(monom),
                                          self.order))
        return self._terms.get(monom, S.Zero)

    def truncate(self, order):
        """Return the series truncated to a lower (or equal) order."""
        order = sympify(order)
        if self.order is not S.Infinity and order > self.order:
            raise ValueError("cannot raise the order of a truncated series")
        return WeightedSeries(self.gens, self.weights, self._terms, order)

    def _check_compatible(self, other):
        if self.gens != other.gens or self.weights != other.weights:
            raise ValueError("series live in different rings")

    def _gen_index(self, gen):
        if isinstance(gen, int):
            return gen
        return self.gens.index(Symbol(gen) if isinstance(gen, str) else gen)

    def __add__(self, other):
        if isinstance(other, WeightedSeries):
            self._check_compatible(other)
            order = min(self.order, other.order)
            terms = dict(self._terms)
            for monom, coeff in other._terms.items():
                terms[monom] = terms.get(monom, S.Zero) + coeff
            return WeightedSeries(self.gens, self.weights, terms, order)
        other = sympify(other)
        terms = dict(self._terms)
        zero = (0,) * len(self.gens)
        terms[zero] = terms.get(zero, S.Zero) + other
        return WeightedSeries(self.gens, self.weights, terms, self.order)

    __radd__ = __add__

    def __neg__(self):
        return WeightedSeries(self.gens, self.weights,
                              {m: -c for m, c in self._terms.items()},
                              self.order)

    def __sub__(self, other):
        if isinstance(other, WeightedSeries):
            return self + (-other)
        return self + (-sympify(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, WeightedSeries):
            self._check_compatible(other)
            order = min(self.order, other.order)
            terms = {}
            for m1, c1 in self._terms.items():
                for m2, c2 in other._terms.items():
                    monom = tuple(a + b for a, b in zip(m1, m2))
                    if (order is not S.Infinity
                            and _weighted_degree(self.weights, monom) >= order):
                        continue
                    terms[monom] = terms.get(monom, S.Zero) + c1 * c2
            return WeightedSeries(self.gens, self.weights, terms, order)
        other = sympify(other)
        return WeightedSeries(self.gens, self.weights,
                              {m: c * other for m, c in self._terms.items()},
                              self.order)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, WeightedSeries):
            return self * other.inverse()
        return self * (1 / sympify(other))

    def __pow__(self, n):
        n = sympify(n)
        if not n.is_Integer:
            raise ValueError("only integer powers are supported")
        n = int(n)
        if n < 0:
            return self.inverse() ** (-n)
        result = WeightedSeries.one(self.gens, self.weights)
        base = self
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def _require_finite_order(self, func):
        if self.order is S.Infinity:
            raise ValueError("%s requires a series with finite order" % func)

    def inverse(self):
        """The multiplicative inverse, keeping the same order.

        The constant term must be nonzero.
        """
        self._require_finite_order("inverse")
        c = self._terms.get((0,) * len(self.gens), S.Zero)
        if c == 0:
            raise ValueError("only series with nonzero constant term "
                             "are invertible")
        u = (self - c) * (-1 / c)
        result = WeightedSeries.one(self.gens, self.weights)
        term = result
        while True:
            term = term * u
            if not term._terms:
                break
            result = result + term
        return result * (1 / c)

    def exp(self):
        """The exponential of the series, keeping the same order."""
        self._require_finite_order("exp")
        c = self._terms.get((0,) * len(self.gens), S.Zero)
        h = self - c
        result = WeightedSeries.one(self.gens, self.weights)
        term = result
        k = 0
        while True:
            k += 1
            term = term * h * Rational(1, k)
            if not term._terms:
                break
            result = result + term
        if c == 0:
            return result
        return result * _exp(c)

    def log(self):
        """The logarithm of the series, keeping the same order.

        The constant term must be nonzero.
        """
        self._require_finite_order("log")
        c = self._terms.get((0,) * len(self.gens), S.Zero)
        if c == 0:
            raise ValueError("log requires a nonzero constant term")
        h = (self - c) * (1 / c)
        result = WeightedSeries.zero(self.gens, self.weights, self.order)
        term = WeightedSeries.one(self.gens, self.weights)
        k = 0
        while True:
            k += 1
            term = term * h
            if not term._terms:
                break
            result = result + term * (Rational(-1 if k % 2 == 0 else 1, k))
        if c == 1:
            return result
        return result + _log(c)

    def diff(self, gen):
        """Differentiate with respect to a generator.

        The guaranteed order drops by the weight of the generator.
        """
        i = self._gen_index(gen)
        terms = {}
        for monom, coeff in self._terms.items():
            k = monom[i]
            if k:
                m = list(monom)
                m[i] -= 1
                terms[tuple(m)] = k * coeff
        if self.order is S.Infinity:
            order = S.Infinity
        else:
            order = max(0, self.order - self.weights[i])
        return WeightedSeries(self.gens, self.weights, terms, order)

    def compose(self, mapping):
        """Substitute series for generators.

        ``mapping`` maps each generator to a :class:`WeightedSeries` in
        the same ring (generators and weights) with zero constant term;
        generators missing from the mapping are kept.  The guaranteed
        order of the result is computed from this series' order, the
        orders of the substituted series, and their leading degrees: a
        substituted series whose leading degree differs from the weight
        of the generator it replaces changes the error bound, so the
        input order is never reused blindly.
        """
        subs = []
        for i, g in enumerate(self.gens):
            s = mapping.get(g)
            if s is None:
                s = WeightedSeries.var(self.gens, self.weights, i)
            elif not isinstance(s, WeightedSeries):
                s = WeightedSeries.constant(self.gens, self.weights, s)
            elif s.gens != self.gens or s.weights != self.weights:
                raise ValueError("substituted series must live in the "
                                 "same ring")
            subs.append(s)
        zero = (0,) * len(self.gens)
        for s in subs:
            if s._terms.get(zero, S.Zero) != 0:
                raise ValueError("can only substitute series with zero "
                                 "constant term")
        rho = None
        for s, w in zip(subs, self.weights):
            if s._terms:
                lead = s.leading_degree
            else:
                lead = s.order
            if lead is S.Infinity:
                continue
            r = Rational(lead, w)
            rho = r if rho is None else min(rho, r)
        orders = [s.order for s in subs]
        if self.order is not S.Infinity and rho is not None:
            orders.append(ceiling(rho * self.order))
        order = min(orders)
        result = WeightedSeries.zero(self.gens, self.weights, order)
        for monom, coeff in self._terms.items():
            term = WeightedSeries.constant(self.gens, self.weights, coeff)
            for s, k in zip(subs, monom):
                if k:
                    term = term * s ** k
            result = result + term
        return result

    def to_expr(self):
        """The known part of the series as a SymPy expression."""
        result = S.Zero
        for monom, coeff in self._terms.items():
            term = coeff
            for g, k in zip(self.gens, monom):
                if k:
                    term *= g ** k
            result += term
        return result

    def __eq__(self, other):
        return (isinstance(other, WeightedSeries)
                and self.gens == other.gens
                and self.weights == other.weights
                and self.order == other.order
                and self._terms == other._terms)

    def __repr__(self):
        return "WeightedSeries(%s, %s, %s, order=%s)" % (
            self.gens, self.weights, self._terms, self.order)
