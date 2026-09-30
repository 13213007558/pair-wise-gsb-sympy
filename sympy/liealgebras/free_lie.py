from __future__ import annotations

from fractions import Fraction
from itertools import product

from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.physics.quantum.commutator import Commutator
from sympy.utilities.misc import as_int

MAX_BCH_ORDER = 8


def _is_lyndon(word):
    return all(word < word[i:] + word[:i] for i in range(1, len(word)))


def _lyndon_split(word):
    for i in range(len(word) - 1, 0, -1):
        left = word[:i]
        if _is_lyndon(left) and _is_lyndon(word[i:]):
            return left, word[i:]
    raise ValueError("Lyndon words of length one do not have a factorization.")


def _lyndon_words(length):
    return [word for word in product((0, 1), repeat=length)
            if _is_lyndon(word)]


def _all_words(length):
    return list(product((0, 1), repeat=length))


def _polynomial_multiply(left, right, order):
    result = {}
    for left_word, left_coefficient in left.items():
        for right_word, right_coefficient in right.items():
            word = left_word + right_word
            if len(word) <= order:
                result[word] = (result.get(word, Fraction(0)) +
                                left_coefficient * right_coefficient)
    return result


def _commutator_polynomial(left, right, order):
    result = _polynomial_multiply(left, right, order)
    for word, coefficient in _polynomial_multiply(right, left, order).items():
        result[word] = result.get(word, Fraction(0)) - coefficient
    return result


def _exponential_series(term, order):
    result = [{} for _ in range(order + 1)]
    result[0] = {(): Fraction(1)}
    power = {(): Fraction(1)}
    factorial = 1
    for degree in range(1, order + 1):
        factorial *= degree
        power = _polynomial_multiply(power, term, order)
        result[degree] = {word: coefficient / factorial
                          for word, coefficient in power.items()}
    return result


def _series_multiply(left, right, order):
    result = [{} for _ in range(order + 1)]
    for degree in range(order + 1):
        terms = {}
        for left_degree in range(degree + 1):
            product_terms = _polynomial_multiply(
                left[left_degree], right[degree - left_degree], order)
            for word, coefficient in product_terms.items():
                terms[word] = terms.get(word, Fraction(0)) + coefficient
        result[degree] = terms
    return result


def _logarithm_series(series, order):
    result = [{} for _ in range(order + 1)]
    nonconstant = [{}] + list(series[1:])
    power = [{} for _ in range(order + 1)]
    power[0] = {(): Fraction(1)}
    for exponent in range(1, order + 1):
        power = _series_multiply(power, nonconstant, order)
        coefficient = Fraction((-1) ** (exponent + 1), exponent)
        for degree in range(1, order + 1):
            for word, value in power[degree].items():
                result[degree][word] = (
                    result[degree].get(word, Fraction(0)) +
                    coefficient * value)
    return result


def _baker_campbell_hausdorff_series(order):
    left = _exponential_series({(0,): Fraction(1)}, order)
    right = _exponential_series({(1,): Fraction(1)}, order)
    return _logarithm_series(_series_multiply(left, right, order), order)


class _LyndonBasis:
    def __init__(self, generators):
        self.generators = generators
        self.expressions = {}
        self.signs = {}
        self.polynomials = {}

    def element(self, word):
        if word in self.expressions:
            return self.expressions[word], self.signs[word], self.polynomials[word]

        if len(word) == 1:
            expression = self.generators[word[0]]
            sign = 1
            polynomial = {word: Fraction(1)}
        else:
            left_word, right_word = _lyndon_split(word)
            left_expression, left_sign, left_polynomial = self.element(left_word)
            right_expression, right_sign, right_polynomial = self.element(right_word)
            expression = Commutator(left_expression, right_expression)
            sign = left_sign * right_sign
            polynomial = _commutator_polynomial(
                left_polynomial, right_polynomial, len(word))

            if isinstance(expression, Mul) and expression.args[0] is S.NegativeOne:
                sign = -sign
                expression = expression.args[1]

        self.expressions[word] = expression
        self.signs[word] = sign
        self.polynomials[word] = polynomial
        return expression, sign, polynomial


def _project_to_lyndon_basis(words, basis_words, target, basis):
    matrix = []
    for word in basis_words:
        _, sign, polynomial = basis.element(word)
        matrix.append([sign * polynomial.get(word_key, Fraction(0))
                       for word_key in words])

    pivot_word_indices = []
    rank = 0
    reduced = [list(row) for row in matrix]
    for word_column in range(len(words)):
        pivot = next((row for row in range(rank, len(basis_words))
                      if reduced[row][word_column]), None)
        if pivot is None:
            continue
        reduced[rank], reduced[pivot] = reduced[pivot], reduced[rank]
        pivot_value = reduced[rank][word_column]
        for row in range(rank + 1, len(basis_words)):
            if not reduced[row][word_column]:
                continue
            factor = reduced[row][word_column] / pivot_value
            for index in range(word_column, len(words)):
                reduced[row][index] -= factor * reduced[rank][index]
        pivot_word_indices.append(word_column)
        rank += 1

    selected_words = [words[index] for index in pivot_word_indices]
    transposed = [[matrix[basis_index][word_index]
                   for basis_index in range(len(basis_words))]
                  for word_index in pivot_word_indices]
    augmented = [transposed[row] + [target.get(selected_words[row], Fraction(0))]
                 for row in range(len(basis_words))]

    for column in range(len(basis_words)):
        pivot = next((row for row in range(column, len(augmented))
                      if augmented[row][column]), None)
        if pivot is None:
            raise RuntimeError("Lyndon basis matrix is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for row in range(len(augmented)):
            if row == column or not augmented[row][column]:
                continue
            factor = augmented[row][column] / pivot_value
            for index in range(column, len(basis_words) + 1):
                augmented[row][index] -= factor * augmented[column][index]

    solution = [augmented[row][-1] / augmented[row][row]
                for row in range(len(basis_words))]
    return solution


def baker_campbell_hausdorff(x, y, order=4, max_order=MAX_BCH_ORDER):
    r"""
    BCH truncated expansion of ``log(exp(x)*exp(y))``.

    Explanation
    ===========

    The result contains all terms through degree ``order`` in ``x`` and
    ``y``.  Every non-linear term is represented as a nested commutator.
    The basis is the Chen-Fox-Lyndon basis of the free Lie algebra, using
    lexicographic ordering with ``x < y``.  For a Lyndon word ``w``, write
    ``w = uv`` using its longest Lyndon prefix ``u`` and define the nested
    bracket recursively as ``b(w) = [b(u), b(v)]``, with ``b(x) = x`` and
    ``b(y) = y``.

    :class:`~sympy.physics.quantum.commutator.Commutator` reverses some
    nested brackets during its usual antisymmetric canonicalization.  The
    corresponding Lyndon coefficient then carries a sign.

    Internally, ``log(exp(x) exp(y))`` is first formed as a noncommutative
    word series and each homogeneous component is projected onto its Lyndon
    basis.  Consequently, commutator identities, including antisymmetry and
    the Jacobi identity, are used through the equality of their word
    expansions; no table of BCH coefficients is hard-coded.

    Parameters
    ==========

    x, y : Expr
        Noncommutative generators or expressions.
    order : int, optional
        Highest commutator degree included in the result.  The default is 4.
    max_order : int, optional
        Workload limit.  Orders larger than this raise ``ValueError``.  The
        default limit is 8; set this higher only for small one-off calls.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.liealgebras import baker_campbell_hausdorff
    >>> x, y = symbols("x y", commutative=False)
    >>> baker_campbell_hausdorff(x, y, order=2)
    x + y + [x,y]/2

    Commuting inputs reduce to the ordinary commutative logarithm:

    >>> a, b = symbols("a b")
    >>> baker_campbell_hausdorff(a, b, order=6)
    a + b

    """
    if x.is_commutative or y.is_commutative:
        return x + y
    if x == y:
        return 2*x

    try:
        order = as_int(order)
        max_order = as_int(max_order)
    except (TypeError, ValueError):
        raise ValueError(
            "order and max_order must be positive integers.") from None
    if order < 1 or max_order < 1:
        raise ValueError("order and max_order must be positive integers.")
    if order > max_order:
        raise ValueError(
            "order %d exceeds the BCH workload limit max_order=%d."
            % (order, max_order))

    series = _baker_campbell_hausdorff_series(order)
    result = x + y
    basis_factory = _LyndonBasis((x, y))

    for degree in range(2, order + 1):
        words = _all_words(degree)
        basis_words = _lyndon_words(degree)
        coefficients = _project_to_lyndon_basis(
            words, basis_words, series[degree], basis_factory)
        for word, coefficient in zip(basis_words, coefficients):
            if coefficient:
                expression, _, _ = basis_factory.element(word)
                result += Rational(coefficient.numerator,
                                   coefficient.denominator) * expression

    return result
