"""
Recurrences
"""
from __future__ import annotations

from sympy.core import S, Dummy, sympify
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
    """
    Berlekamp-Massey algorithm over the finite field GF(p).

    Given a list s of residues modulo the prime p, return the connection
    polynomial C(x) = 1 + C_1 x + ... + C_L x^L (as a list
    [1, C_1, ..., C_L] of residues) of a shortest linear recurrence
    s_n + C_1 s_{n-1} + ... + C_L s_{n-L} = 0 (mod p) generating s,
    together with its order L.
    """
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
        coef = d*pow(b, -1, p) % p
        T = C[:]
        C = C + [0]*(len(B) + m - len(C))
        for i in range(len(B)):
            C[i + m] = (C[i + m] - coef*B[i]) % p
        if 2*L <= n:
            L = n + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1
    return C, L


class LinearRecurrence:
    """
    A shortest linear recurrence inferred from a finite prefix of a
    sequence over the finite field GF(p).

    Instances of this class are returned by
    :func:~.infer_linear_recurrence and should not be constructed
    directly.

    Attributes
    ==========

    modulus : int
        The prime p defining the field GF(p).
    prefix : tuple
        The training prefix (reduced modulo p) the recurrence was
        inferred from.
    order : int or None
        The minimal order L of the recurrence. None when the prefix is
        empty.
    coefficients : tuple
        The coefficients (c_1, ..., c_L) such that
        s_n = c_1 s_{n-1} + ... + c_L s_{n-L} (mod p) for n >= start.
    status : str
        One of the following:

        - 'empty': the prefix was empty, no recurrence exists.
        - 'zero': the prefix was all zeros; the zero recurrence
          (order 0) generates it and predicts zeros forever.
        - 'unique': the prefix is long enough (len >= 2*order) to
          determine a unique shortest recurrence.
        - 'underdetermined': the prefix is too short (len < 2*order)
          to determine the future uniquely; other shortest recurrences
          consistent with the prefix exist.
    holdout : tuple or None
        The held-out data passed to :func:~.infer_linear_recurrence,
        or None if none was given.
    holdout_mismatch : int or None
        The absolute index of the first position at which the holdout
        data disagrees with the recurrence, or None if the holdout is
        fully consistent (or no holdout was given). The coefficients
        are never refitted to the holdout data.

    Notes
    =====

    The recurrence is the shortest one consistent with the given finite
    prefix only. Even when the status is 'unique', this means unique
    among the recurrences generating the observed prefix; no finite
    prefix can prove that an infinite sequence follows a recurrence.
    """

    EMPTY = 'empty'
    ZERO = 'zero'
    UNIQUE = 'unique'
    UNDERDETERMINED = 'underdetermined'

    def __init__(self, modulus, prefix, order, coefficients, status,
                 holdout=None, holdout_mismatch=None):
        self.modulus = modulus
        self.prefix = prefix
        self.order = order
        self.coefficients = coefficients
        self.status = status
        self.holdout = holdout
        self.holdout_mismatch = holdout_mismatch

    @property
    def start(self):
        """
        The first index from which the recurrence applies.

        The inferred recurrence determines s_n for all n >= start
        (within and beyond the training prefix), where start equals
        the order of the recurrence. It is None for an empty prefix.
        """
        if self.status == self.EMPTY:
            return None
        return self.order

    def extend(self, count):
        """
        Extend the sequence by count further terms using the recurrence.

        Returns the list of the count new terms following the training
        prefix, each reduced modulo p. For an 'underdetermined' result
        this is one of several possible continuations.

        Examples
        ========

        >>> from sympy.discrete.recurrences import infer_linear_recurrence
        >>> r = infer_linear_recurrence([1, 1, 2, 3, 5, 1, 6, 0], 7)
        >>> r.extend(3)
        [6, 6, 5]
        """
        count = as_int(count)
        if count < 0:
            raise ValueError("Count of terms must be non-negative")
        if self.status == self.EMPTY:
            raise ValueError("Cannot extend from an empty prefix")
        p = self.modulus
        terms = list(self.prefix)
        new = []
        for _ in range(count):
            nxt = 0
            for i, c in enumerate(self.coefficients):
                nxt += c*terms[-1 - i]
            nxt %= p
            terms.append(nxt)
            new.append(nxt)
        return new

    def validate(self, holdout):
        """
        Check held-out data against the recurrence.

        Returns the absolute index (counted from the start of the
        training prefix) of the first position where holdout disagrees
        with the recurrence, or None if all given terms match. The
        recurrence itself is never modified by this check.
        """
        h = [as_int(a) % self.modulus for a in holdout]
        expected = self.extend(len(h))
        for i, (e, a) in enumerate(zip(expected, h)):
            if e != a:
                return len(self.prefix) + i
        return None

    def connection_polynomial(self, symbol=None):
        """
        The connection polynomial C(x) = 1 - c_1 x - ... - c_L x^L
        as a Poly over GF(p).

        The sequence satisfies sum of C_i s_{n-i} = 0 (mod p) for
        n >= start, where C_i are the coefficients of C(x).
        """
        if self.status == self.EMPTY:
            raise ValueError("An empty prefix defines no recurrence")
        if symbol is None:
            symbol = Dummy('x')
        p = self.modulus
        poly = 1 + sum((-c % p)*symbol**(i + 1)
                       for i, c in enumerate(self.coefficients))
        return Poly(poly, symbol, modulus=p)

    def characteristic_polynomial(self, symbol=None):
        """
        The characteristic polynomial
        x^L - c_1 x^(L-1) - ... - c_L as a Poly over GF(p).

        This is the polynomial p(x) of linrec with
        x^L = c_1 x^(L-1) + ... + c_L.
        """
        if self.status == self.EMPTY:
            raise ValueError("An empty prefix defines no recurrence")
        if symbol is None:
            symbol = Dummy('x')
        L = self.order
        poly = symbol**L - sum(c*symbol**(L - 1 - i)
                               for i, c in enumerate(self.coefficients))
        return Poly(poly, symbol, modulus=self.modulus)

    def __repr__(self):
        return ("LinearRecurrence(order=%s, coefficients=%s, modulus=%s, "
                "status=%r)" % (self.order, self.coefficients,
                                self.modulus, self.status))


def infer_linear_recurrence(prefix, p, holdout=None):
    r"""
    Infer a shortest linear recurrence from a finite prefix over GF(p).

    Given a prefix of a sequence modulo the prime p, find a recurrence
    of minimal order L

    .. math :: s_n = c_1 s_{n-1} + c_2 s_{n-2} + \cdots + c_L s_{n-L}
        pmod{p}

    generating the prefix, using the Berlekamp-Massey algorithm. Only
    the training prefix is used for the inference; the optional holdout
    data is checked against the inferred recurrence afterwards and never
    influences the coefficients.

    Parameters
    ==========

    prefix : iterable
        The training terms of the sequence; each term is reduced
        modulo p.
    p : int
        The prime modulus of the finite field GF(p).
    holdout : iterable, optional
        Additional terms immediately following the prefix, used only
        to validate the inferred recurrence.

    Returns
    =======

    LinearRecurrence
        The inference result; see the documentation of
        :class:~.LinearRecurrence for its attributes. In particular
        status is 'empty' for an empty prefix, 'zero' for an all-zero
        prefix, 'unique' when the prefix determines a unique shortest
        recurrence (len(prefix) >= 2*order) and 'underdetermined' when
        the prefix is too short to determine the future uniquely.
        holdout_mismatch gives the absolute index of the first term of
        the holdout data inconsistent with the recurrence (None if the
        holdout is fully consistent).

    Notes
    =====

    The result is the shortest recurrence consistent with the given
    finite prefix. Even a 'unique' result is unique only among the
    recurrences generating the observed prefix; a finite prefix can
    never prove that an infinite sequence follows a recurrence.

    Examples
    ========

    >>> from sympy.discrete.recurrences import infer_linear_recurrence

    The Fibonacci sequence modulo 7 has order 2:

    >>> r = infer_linear_recurrence([1, 1, 2, 3, 5, 1, 6, 0], 7)
    >>> r.status
    'unique'
    >>> r.order, r.coefficients, r.start
    (2, (1, 1), 2)
    >>> r.extend(3)
    [6, 6, 5]

    Holdout data is validated without changing the recurrence:

    >>> r = infer_linear_recurrence([1, 1, 2, 3, 5, 1, 6, 0], 7, holdout=[6, 0])
    >>> r.holdout_mismatch
    9
    >>> r.coefficients
    (1, 1)

    Degenerate and ambiguous prefixes get explicit statuses:

    >>> infer_linear_recurrence([], 7).status
    'empty'
    >>> infer_linear_recurrence([0, 0, 0], 7).status
    'zero'
    >>> infer_linear_recurrence([1, 2, 3], 7).status
    'underdetermined'

    See Also
    ========

    linrec
    """
    p = as_int(p)
    if not isprime(p):
        raise ValueError("Modulus must be a prime number")
    values = tuple(as_int(a) % p for a in prefix)
    n = len(values)
    if holdout is not None:
        holdout = tuple(as_int(a) % p for a in holdout)
    if n == 0:
        result = LinearRecurrence(p, values, None, (),
                                  LinearRecurrence.EMPTY, holdout)
    elif not any(values):
        result = LinearRecurrence(p, values, 0, (),
                                  LinearRecurrence.ZERO, holdout)
    else:
        conn, L = _berlekamp_massey(list(values), p)
        conn = (conn + [0]*(L + 1 - len(conn)))[:L + 1]
        coeffs = tuple((-c) % p for c in conn[1:])
        if 2*L <= n:
            status = LinearRecurrence.UNIQUE
        else:
            status = LinearRecurrence.UNDERDETERMINED
        result = LinearRecurrence(p, values, L, coeffs, status, holdout)
    if holdout is not None and result.status != LinearRecurrence.EMPTY:
        result.holdout_mismatch = result.validate(holdout)
    return result
