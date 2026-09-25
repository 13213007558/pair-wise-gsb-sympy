'''Functions returning normal forms of matrices'''
from __future__ import annotations

from typing import NamedTuple

from sympy.core.numbers import Integer
from sympy.matrices.dense import Matrix
from sympy.matrices.exceptions import ShapeError
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import (
        smith_normal_form as _snf,
        is_smith_normal_form as _is_snf,
        smith_normal_decomp as _snd,
        invariant_factors as _invf,
        hermite_normal_form as _hnf,
    )


def _to_domain(m, domain=None):
    """Convert Matrix to DomainMatrix"""
    # XXX: deprecated support for RawMatrix:
    ring = getattr(m, "ring", None)
    m = m.applyfunc(lambda e: e.as_expr() if isinstance(e, Poly) else e)

    dM = DomainMatrix.from_Matrix(m)

    domain = domain or ring
    if domain is not None:
        dM = dM.convert_to(domain)
    return dM


def smith_normal_form(m, domain=None):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(smith_normal_form(m, domain=ZZ))
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])

    '''
    dM = _to_domain(m, domain)
    return _snf(dM).to_Matrix()


def is_smith_normal_form(m, domain=None):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    dM = _to_domain(m, domain)
    return _is_snf(dM)


def smith_normal_decomp(m, domain=None):
    '''
    Return the Smith Normal Decomposition of a matrix `m` over the ring
    `domain`. This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_decomp
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> a, s, t = smith_normal_decomp(m, domain=ZZ)
    >>> assert a == s * m * t
    '''
    dM = _to_domain(m, domain)
    a, s, t = _snd(dM)
    return a.to_Matrix(), s.to_Matrix(), t.to_Matrix()


def invariant_factors(m, domain=None):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    .. [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    dM = _to_domain(m, domain)
    factors = _invf(dM)
    factors = tuple(dM.domain.to_sympy(f) for f in factors)
    # XXX: deprecated.
    if hasattr(m, "ring"):
        if m.ring.is_PolynomialRing:
            K = m.ring
            to_poly = lambda f: Poly(f, K.symbols, domain=K.domain)
            factors = tuple(to_poly(f) for f in factors)
    return factors


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of a Matrix *A* of integers.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.normalforms import hermite_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(hermite_normal_form(m))
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``Matrix`` of integers.

    D : int, optional
        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*
        being any multiple of $\det(W)$ may be provided. In this case, if *A*
        also has rank $m$, then we may use an alternative algorithm that works
        mod *D* in order to prevent coefficient explosion.

    check_rank : boolean, optional (default=False)
        The basic assumption is that, if you pass a value for *D*, then
        you already believe that *A* has rank $m$, so we do not waste time
        checking it for you. If you do want this to be checked (and the
        ordinary, non-modulo *D* algorithm to be used if the check fails), then
        set *check_rank* to ``True``.

    Returns
    =======

    ``Matrix``
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    DMShapeError
        If the mod *D* algorithm is used but the matrix has more rows than
        columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithms 2.4.5 and 2.4.8.)

    """
    # Accept any of Python int, SymPy Integer, and ZZ itself:
    if D is not None and not ZZ.of_type(D):
        D = ZZ(int(D))
    return _hnf(A._rep, D=D, check_rank=check_rank).to_Matrix()


class IntegerSolution(NamedTuple):
    '''The complete integer solution of a linear system ``A*x = b``.

    Attributes
    ==========

    particular : Matrix
        A particular integer solution ``x0`` with ``A*x0 = b``.
    nullbasis : Matrix
        A matrix ``B`` whose columns form a basis of the integer
        nullspace of ``A`` (so ``A*B`` is the zero matrix). The complete
        set of integer solutions is ``x0 + B*z`` as ``z`` ranges over
        all integer vectors of the appropriate length.
    '''
    particular: Matrix
    nullbasis: Matrix


class DivisibilityContradiction(NamedTuple):
    '''Certificate that a system ``A*x = b`` has no integer solution.

    Let ``D = S*A*T`` be the Smith normal decomposition of ``A`` (see
    :func:`smith_normal_decomp`) and let ``c = S * b`` be the
    transformed right-hand side. The system has an integer solution if
    and only if the diagonal entry ``D[i, i]`` divides ``c[i]`` for
    every row ``i`` (in particular ``c[i]`` must be zero for rows beyond
    the rank of ``A``). This certificate records a row for which that
    divisibility condition fails, so infeasibility can be checked
    directly with exact integer arithmetic.

    Attributes
    ==========

    column : int
        Index of the column of ``b`` for which the system is infeasible.
    row : int
        Index of the row of the Smith normal form where the obstruction
        occurs.
    divisor : Integer
        The diagonal entry ``D[row, row]`` of the Smith normal form of
        ``A`` (zero for rows beyond the rank of ``A``).
    residual : Integer
        The entry ``c[row]`` of the transformed right-hand side. It is
        not divisible by ``divisor`` (it is nonzero while ``divisor`` is
        zero for rows beyond the rank).
    transform : Matrix
        The unimodular integer matrix ``S`` mapping ``b`` to the
        transformed right-hand side ``c = transform * b``.
    '''
    column: int
    row: int
    divisor: Integer
    residual: Integer
    transform: Matrix


def integer_linsolve(A, b):
    '''
    Solve the linear system ``A*x = b`` over the integers.

    Given an integer matrix ``A`` of shape ``(m, n)`` and an integer
    matrix ``b`` of shape ``(m, k)``, describe all integer vectors ``x``
    satisfying ``A*x = b``. Unlike ``linsolve``, which solves over the
    rationals, this returns the complete solution lattice (or a proof
    that no integer solution exists), and ``A`` may be rectangular or
    rank deficient.

    Explanation
    ===========

    The system is solved through the Smith normal decomposition
    ``D = S*A*T`` with unimodular integer matrices ``S`` and ``T`` and
    diagonal ``D`` (see :func:`smith_normal_decomp`). Writing
    ``c = S * b`` and ``y = T**-1 * x``, the system is equivalent to the
    diagonal system ``D*y = c``, which has an integer solution exactly
    when ``D[i, i]`` divides ``c[i]`` for every row. All computations
    use exact integer arithmetic.

    Returns
    =======

    If ``b`` is a single column, a single result; if ``b`` has ``k > 1``
    columns, a list of ``k`` results, one per column (the Smith normal
    decomposition of ``A`` is computed only once and shared). Each
    result is either:

    * an :class:`IntegerSolution` ``(x0, B)`` such that the complete set
      of integer solutions for that column is ``x = x0 + B*z`` with
      ``z`` an arbitrary integer vector, or
    * a :class:`DivisibilityContradiction` certifying that the column
      has no integer solution.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.normalforms import integer_linsolve
    >>> A = Matrix([[2, 4], [3, 6]])
    >>> b = Matrix([2, 3])
    >>> x0, B = integer_linsolve(A, b)
    >>> x0
    Matrix([
    [1],
    [0]])
    >>> B
    Matrix([
    [-2],
    [ 1]])
    >>> A*x0 == b and (A*B).is_zero_matrix
    True

    A system may have a rational solution but no integer solution; the
    returned certificate exposes the failed divisibility:

    >>> cert = integer_linsolve(Matrix([[2]]), Matrix([1]))
    >>> cert.divisor, cert.residual
    (2, 1)
    >>> cert.residual % cert.divisor != 0
    True

    '''
    A = Matrix(A)
    b = Matrix(b)
    if b.rows != A.rows:
        raise ShapeError(
            "`A` and `b` must have the same number of rows.")
    for M in (A, b):
        if any(e.is_integer is not True for e in M):
            raise ValueError(
                "integer_linsolve requires matrices with integer entries")

    m, n = A.shape
    D, S, T = smith_normal_decomp(A, domain=ZZ)
    c = S * b
    r = sum(1 for i in range(min(m, n)) if D[i, i] != 0)
    nullbasis = T[:, r:]

    results = []
    for j in range(b.cols):
        y0 = Matrix.zeros(n, 1)
        contradiction = None
        for i in range(m):
            ci = c[i, j]
            if i < r:
                di = D[i, i]
                q, rem = divmod(ci, di)
                if rem:
                    contradiction = DivisibilityContradiction(
                        j, i, di, ci, S)
                    break
                y0[i] = q
            elif ci:
                contradiction = DivisibilityContradiction(
                    j, i, D[i, i] if i < n else Integer(0), ci, S)
                break
        if contradiction is None:
            results.append(IntegerSolution(T * y0, nullbasis))
        else:
            results.append(contradiction)

    if b.cols == 1:
        return results[0]
    return results
