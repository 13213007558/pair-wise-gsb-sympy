"""
Recurrences
"""
from __future__ import annotations

from sympy.core import S, sympify
from sympy.core.symbol import Symbol
from sympy.ntheory import isprime
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int


def linrec(coeffs, init, n):
    r"""
    Evaluation of univariate linear recurrences of homogeneous type
    having coefficients independent of the recurrence variable.

    Parameters
    ==========

    coeffs : iterable
        Coefficients of the recurrence
    init : iterable
        Initial values of the recurrence
    n : Integer
        Point of evaluation for the recurrence

    Notes
    =====

    Let `y(n)` be the recurrence of given type, ``c`` be the sequence
    of coefficients, ``b`` be the sequence of initial/base values of the
    recurrence and ``k`` (equal to ``len(c)``) be the order of recurrence.
    Then,

    .. math :: y(n) = \begin{cases} b_n & 0 \le n < k \\
        c_0 y(n-1) + c_1 y(n-2) + \cdots + c_{k-1} y(n-k) & n \ge k
        \end{cases}

    Let `x_0, x_1, \ldots, x_n` be a sequence and consider the transformation
    that maps each polynomial `f(x)` to `T(f(x))` where each power `x^i` is
    replaced by the corresponding value `x_i`. The sequence is then a solution
    of the recurrence if and only if `T(x^i p(x)) = 0` for each `i \ge 0` where
    `p(x) = x^k - c_0 x^(k-1) - \cdots - c_{k-1}` is the characteristic
    polynomial.

    Then `T(f(x)p(x)) = 0` for each polynomial `f(x)` (as it is a linear
    combination of powers `x^i`). Now, if `x^n` is congruent to
    `g(x) = a_0 x^0 + a_1 x^1 + \cdots + a_{k-1} x^{k-1}` modulo `p(x)`, then
    `T(x^n) = x_n` is equal to
    `T(g(x)) = a_0 x_0 + a_1 x_1 + \cdots + a_{k-1} x_{k-1}`.

    Computation of `x^n`,
    given `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`
    is performed using exponentiation by squaring (refer to [1_]) with
    an additional reduction step performed to retain only first `k` powers
    of `x` in the representation of `x^n`.

    Examples
    ========

    >>> from sympy.discrete.recurrences import linrec
    >>> from sympy.abc import x, y, z

    >>> linrec(coeffs=[1, 1], init=[0, 1], n=10)
    55

    >>> linrec(coeffs=[1, 1], init=[x, y], n=10)
    34*x + 55*y

    >>> linrec(coeffs=[x, y], init=[0, 1], n=5)
    x**2*y + x*(x**3 + 2*x*y) + y**2

    >>> linrec(coeffs=[1, 2, 3, 0, 0, 4], init=[x, y, z], n=16)
    13576*x + 5676*y + 2356*z

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    .. [2] https://en.wikipedia.org/w/index.php?title=Modular_exponentiation&section=6#Matrices

    See Also
    ========

    sympy.polys.agca.extensions.ExtensionElement.__pow__

    """

    if not coeffs:
        return S.Zero

    if not iterable(coeffs):
        raise TypeError("Expected a sequence of coefficients for"
                        " the recurrence")

    if not iterable(init):
        raise TypeError("Expected a sequence of values for the initialization"
                        " of the recurrence")

    n = as_int(n)
    if n < 0:
        raise ValueError("Point of evaluation of recurrence must be a "
                        "non-negative integer")

    c = [sympify(arg) for arg in coeffs]
    b = [sympify(arg) for arg in init]
    k = len(c)

    if len(b) > k:
        raise TypeError("Count of initial values should not exceed the "
                        "order of the recurrence")
    else:
        b += [S.Zero]*(k - len(b)) # remaining initial values default to zero

    if n < k:
        return b[n]
    terms = [u*v for u, v in zip(linrec_coeffs(c, n), b)]
    return sum(terms[:-1], terms[-1])


def linrec_coeffs(c, n):
    r"""
    Compute the coefficients of n'th term in linear recursion
    sequence defined by c.

    `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`.

    It computes the coefficients by using binary exponentiation.
    This function is used by `linrec` and `_eval_pow_by_cayley`.

    Parameters
    ==========

    c = coefficients of the divisor polynomial
    n = exponent of x, so dividend is x^n

    """

    k = len(c)

    def _square_and_reduce(u, offset):
        # squares `(u_0 + u_1 x + u_2 x^2 + \cdots + u_{k-1} x^k)` (and
        # multiplies by `x` if offset is 1) and reduces the above result of
        # length upto `2k` to `k` using the characteristic equation of the
        # recurrence given by, `x^k = c_0 x^{k-1} + c_1 x^{k-2} + \cdots + c_{k-1}`

        w = [S.Zero]*(2*len(u) - 1 + offset)
        for i, p in enumerate(u):
            for j, q in enumerate(u):
                w[offset + i + j] += p*q

        for j in range(len(w) - 1, k - 1, -1):
            for i in range(k):
                w[j - i - 1] += w[j]*c[i]

        return w[:k]

    def _final_coeffs(n):
        # computes the final coefficient list - `cf` corresponding to the
        # point at which recurrence is to be evalauted - `n`, such that,
        # `y(n) = cf_0 y(k-1) + cf_1 y(k-2) + \cdots + cf_{k-1} y(0)`

        if n < k:
            return [S.Zero]*n + [S.One] + [S.Zero]*(k - n - 1)
        else:
            return _square_and_reduce(_final_coeffs(n // 2), n % 2)

    return _final_coeffs(n)


def _berlekamp_massey(s, p):
    # Berlekamp-Massey algorithm over GF(p). Returns (L, C) where
    # C = [1, C[1], ..., C[L]] gives the connection polynomial
    # 1 + C[1] x + ... + C[L] x**L of minimal degree L annihilating s,
    # i.e. sum(C[i]*s[j - i] for i in range(L + 1)) == 0 (mod p) for
    # every j in range(L, len(s)).
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for n in range(len(s)):
        d = s[n]
        for i in range(1, L + 1):
            d = (d + C[i]*s[n - i]) % p
        if d == 0:
            m += 1
            continue
        T = C
        coef = d*pow(b, -1, p) % p
        C = list(C) + [0]*max(0, len(B) + m - len(C))
        for i in range(len(B)):
            C[i + m] = (C[i + m] - coef*B[i]) % p
        if 2*L <= n:
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1
    return L, C


class LinearRecurrence:
    r"""
    A linear recurrence with constant coefficients over the finite
    field `GF(p)`.

    Instances are usually created by `find_linear_recurrence`, which
    infers the shortest recurrence satisfied by a finite prefix of a
    sequence, or by `LinearRecurrence.from_characteristic_polynomial`.

    Attributes
    ==========

    modulus : int
        The prime `p` of the ground field `GF(p)`.
    coeffs : tuple
        The coefficients ``(c_0, ..., c_{k-1})`` of the recurrence
        ``y(n) = c_0 y(n-1) + ... + c_{k-1} y(n-k) (mod p)``. This is
        ``None`` when the training prefix was empty.
    order : int
        The order ``k`` of the recurrence. This is ``None`` when the
        training prefix was empty.
    start : int
        The first index from which the recurrence relation is known to
        hold. Terms with index ``>= start`` are predicted from the
        ``order`` preceding terms. This is ``None`` when the training
        prefix was empty.
    status : str
        One of:

        - ``'empty'``: the training prefix was empty, no recurrence
          could be inferred.
        - ``'zero'``: the training prefix was all zeros; the shortest
          recurrence has order 0 and predicts only zeros.
        - ``'unique'``: the prefix is long enough (``2*order <= len``)
          that the shortest recurrence is the unique recurrence of
          minimal order consistent with the prefix.
        - ``'underdetermined'``: the prefix is too short
          (``2*order > len``) to determine the future uniquely; other
          recurrences of the same minimal order fit the prefix but
          predict different continuations.
        - ``'assumed'``: the recurrence was not inferred from data but
          given directly, e.g. via ``from_characteristic_polynomial``.

    prefix_length : int
        The number of training terms the recurrence was inferred from.
    holdout_failure : int or None
        The absolute index of the first held-out term that does not
        satisfy the recurrence, or ``None`` if no holdout data was
        given or all held-out terms matched. The coefficients are
        never refitted to the holdout data.

    Notes
    =====

    A recurrence inferred from a finite prefix is only a hypothesis
    about the infinite sequence: any finite prefix is consistent with
    infinitely many recurrences, and even a ``'unique'`` status only
    means that no *shorter or alternative minimal* recurrence fits the
    same prefix. It is never a proof that the infinite sequence obeys
    the recurrence.

    Examples
    ========

    >>> from sympy.discrete.recurrences import find_linear_recurrence
    >>> fib = [0, 1]
    >>> for _ in range(18):
    ...     fib.append((fib[-1] + fib[-2]) % 7)
    >>> r = find_linear_recurrence(fib, 7)
    >>> r
    LinearRecurrence(coeffs=(1, 1), modulus=7, status='unique')
    >>> r.extend(fib, 3)
    [3, 5, 1]
    >>> r.characteristic_polynomial()
    Poly(x**2 - x - 1, x, modulus=7)

    See Also
    ========

    find_linear_recurrence, linrec

    """

    def __init__(self, coeffs, p, status='assumed', prefix_length=0):
        p = as_int(p)
        if not isprime(p):
            raise ValueError("the modulus of the ground field must be prime")
        self.modulus = p
        if coeffs is None:
            self.coeffs = None
            self.order = None
            self.start = None
        else:
            self.coeffs = tuple(int(c) % p for c in coeffs)
            self.order = len(self.coeffs)
            self.start = self.order
        self.status = status
        self.prefix_length = prefix_length
        self.holdout_failure = None

    def __repr__(self):
        return "%s(coeffs=%s, modulus=%s, status=%r)" % (
            type(self).__name__, self.coeffs, self.modulus, self.status)

    def __eq__(self, other):
        if not isinstance(other, LinearRecurrence):
            return NotImplemented
        return (self.modulus, self.coeffs) == (other.modulus, other.coeffs)

    def __hash__(self):
        return hash((self.modulus, self.coeffs))

    def _check_usable(self):
        if self.order is None:
            raise ValueError(
                "no recurrence was inferred from the empty prefix")

    def extend(self, values, count):
        r"""
        Extend a sequence by iterating the recurrence.

        Parameters
        ==========

        values : iterable
            The known terms preceding the terms to generate. Must
            contain at least ``order`` terms; only the last ``order``
            terms are used.
        count : int
            The number of new terms to generate.

        Returns
        =======

        list
            The ``count`` terms following ``values``, as elements of
            ``{0, ..., p-1}``.

        Examples
        ========

        >>> from sympy.discrete.recurrences import find_linear_recurrence
        >>> r = find_linear_recurrence([0, 1, 1, 2, 3, 5, 1, 6], 7)
        >>> r.extend([0, 1, 1, 2, 3, 5, 1, 6], 4)
        [0, 6, 6, 5]

        """
        self._check_usable()
        count = as_int(count)
        if count < 0:
            raise ValueError("count of terms to generate must be nonnegative")
        p = self.modulus
        s = [int(a) % p for a in values]
        k = self.order
        if len(s) < k:
            raise ValueError(
                "at least order (%d) known terms are required" % k)
        out = []
        for _ in range(count):
            nxt = (sum(c*a for c, a in zip(self.coeffs, reversed(s[-k:])))
                   % p if k else 0)
            out.append(nxt)
            s.append(nxt)
        return out

    def check(self, seq):
        r"""
        Find the first term of ``seq`` violating the recurrence.

        Parameters
        ==========

        seq : iterable
            A sequence of terms of the sequence, starting at index 0.

        Returns
        =======

        int or None
            The smallest index ``j >= start`` with
            ``seq[j] != sum(coeffs[i]*seq[j-1-i]) (mod p)``, or ``None``
            if every term from ``start`` on satisfies the recurrence.

        Examples
        ========

        >>> from sympy.discrete.recurrences import find_linear_recurrence
        >>> r = find_linear_recurrence([0, 1, 1, 2, 3, 5, 1, 6], 7)
        >>> r.check([0, 1, 1, 2, 3, 5, 1, 6, 0, 6]) is None
        True
        >>> r.check([0, 1, 1, 2, 3, 5, 1, 6, 1])
        8

        """
        self._check_usable()
        p = self.modulus
        s = [int(a) % p for a in seq]
        k = self.order
        for j in range(self.start, len(s)):
            if sum(self.coeffs[i]*s[j - 1 - i] for i in range(k)) % p != s[j]:
                return j
        return None

    def characteristic_polynomial(self, x=None):
        r"""
        The characteristic polynomial of the recurrence over `GF(p)`.

        For a recurrence of order ``k`` with coefficients
        ``(c_0, ..., c_{k-1})`` this is
        ``x**k - c_0 x**(k-1) - ... - c_{k-1}`` as a `Poly` with
        coefficients in `GF(p)`, i.e. SymPy's dense finite field
        polynomial representation.

        Parameters
        ==========

        x : Symbol, optional
            The generator of the polynomial. Defaults to ``x``.

        Examples
        ========

        >>> from sympy.discrete.recurrences import find_linear_recurrence
        >>> r = find_linear_recurrence([0, 1, 1, 2, 3, 5, 1, 6], 7)
        >>> r.characteristic_polynomial()
        Poly(x**2 - x - 1, x, modulus=7)

        """
        self._check_usable()
        if x is None:
            x = Symbol('x')
        k = self.order
        poly = x**k - sum(c*x**(k - 1 - i)
                          for i, c in enumerate(self.coeffs))
        return Poly(poly, x, modulus=self.modulus)

    @classmethod
    def from_characteristic_polynomial(cls, poly, x=None):
        r"""
        Build a recurrence from its characteristic polynomial.

        Parameters
        ==========

        poly : Poly or Expr
            A monic univariate polynomial
            ``x**k - c_0 x**(k-1) - ... - c_{k-1}`` with coefficients
            in a finite field `GF(p)`, e.g. created with
            ``Poly(..., modulus=p)``.
        x : Symbol, optional
            The generator, if ``poly`` is an expression.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy.polys import Poly
        >>> from sympy.discrete.recurrences import LinearRecurrence
        >>> r = LinearRecurrence.from_characteristic_polynomial(
        ...     Poly(x**2 + 6*x + 6, x, modulus=7))
        >>> r
        LinearRecurrence(coeffs=(1, 1), modulus=7, status='assumed')
        >>> r.extend([0, 1], 3)
        [1, 2, 3]

        """
        if not isinstance(poly, Poly):
            poly = Poly(poly, x)
        if len(poly.gens) != 1:
            raise ValueError("expected a univariate polynomial")
        dom = poly.get_domain()
        if not dom.is_FiniteField:
            raise ValueError(
                "expected a polynomial over a finite field GF(p)")
        if not poly.is_monic:
            raise ValueError("expected a monic characteristic polynomial")
        p = dom.characteristic()
        c = [(-int(a)) % p for a in poly.all_coeffs()[1:]]
        return cls(c, p)


def find_linear_recurrence(seq, p, holdout=None):
    r"""
    Infer the shortest linear recurrence over `GF(p)` satisfied by a
    finite prefix of a sequence.

    Parameters
    ==========

    seq : iterable
        The training prefix of the sequence. Elements are reduced
        modulo ``p``.
    p : int
        The prime modulus of the ground field `GF(p)`.
    holdout : iterable, optional
        Terms of the sequence immediately following the training
        prefix. They are only used to validate the inferred
        recurrence: the coefficients are computed from ``seq`` alone
        and are never adjusted to fit the holdout data. The absolute
        index of the first held-out term violating the recurrence is
        reported as ``holdout_failure`` of the result.

    Returns
    =======

    LinearRecurrence
        The shortest recurrence consistent with the training prefix,
        with its ``status`` attribute recording whether the prefix was
        empty, all zeros, long enough to determine the continuation
        uniquely, or too short to do so.

    Notes
    =====

    The recurrence is computed with the Berlekamp-Massey algorithm in
    `O(n^2)` field operations for a prefix of length `n`. If the
    minimal order is `L`, the recurrence is the unique recurrence of
    order `\le L` fitting the prefix exactly when `2L \le n`; for
    shorter prefixes the future terms are not determined by the data
    and the status is ``'underdetermined'``.

    The result describes the given finite prefix only. No finite
    prefix can prove that an infinite sequence satisfies a recurrence,
    so even a ``'unique'`` result is a hypothesis about the original
    sequence, not a theorem.

    Examples
    ========

    >>> from sympy.discrete.recurrences import find_linear_recurrence
    >>> fib = [0, 1]
    >>> for _ in range(18):
    ...     fib.append((fib[-1] + fib[-2]) % 7)
    >>> r = find_linear_recurrence(fib, 7)
    >>> r.order, r.coeffs, r.start, r.status
    (2, (1, 1), 2, 'unique')

    Holdout data validates the inferred recurrence without changing
    it; the first mismatching position is reported:

    >>> find_linear_recurrence(fib[:16], 7, holdout=[0, 1, 1, 2]).holdout_failure is None
    True
    >>> find_linear_recurrence(fib[:16], 7, holdout=[0, 1, 1, 3]).holdout_failure
    19

    Degenerate prefixes get explicit statuses:

    >>> find_linear_recurrence([], 7).status
    'empty'
    >>> find_linear_recurrence([0, 0, 0], 7).status
    'zero'
    >>> find_linear_recurrence([0, 0, 1], 7).status
    'underdetermined'

    See Also
    ========

    LinearRecurrence, linrec

    References
    ==========

    .. [1] Massey, J. L. "Shift-register synthesis and BCH decoding."
           IEEE transactions on Information Theory 15.1 (1969): 122-127.
    .. [2] https://en.wikipedia.org/wiki/Berlekamp-Massey_algorithm

    """
    p = as_int(p)
    if not isprime(p):
        raise ValueError("the modulus of the ground field must be prime")
    if not iterable(seq):
        raise TypeError("expected a sequence of terms for the prefix")
    s = [int(a) % p for a in seq]
    n = len(s)
    if n == 0:
        result = LinearRecurrence(None, p, status='empty')
    else:
        L, C = _berlekamp_massey(s, p)
        coeffs = [(-C[i]) % p for i in range(1, L + 1)]
        if L == 0:
            status = 'zero'
        elif 2*L <= n:
            status = 'unique'
        else:
            status = 'underdetermined'
        result = LinearRecurrence(coeffs, p,
                                  status=status, prefix_length=n)
    if holdout is not None:
        if result.order is None:
            raise ValueError(
                "cannot validate holdout data against an empty prefix")
        h = [int(a) % p for a in holdout]
        result.holdout_failure = result.check(s + h)
    return result
