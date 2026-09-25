"""Solving systems of linear equations over the integers."""
from __future__ import annotations

from sympy.core.numbers import Integer
from sympy.matrices.dense import Matrix
from sympy.matrices.exceptions import ShapeError
from sympy.matrices.normalforms import smith_normal_decomp
from sympy.polys.domains import ZZ


def _divides(d, c):
    """Exact divisibility in ZZ, where 0 divides only 0."""
    if d == 0:
        return c == 0
    return c % d == 0


class DivisibilityContradiction:
    r"""
    Certificate that a system $A x = b$ has no solution over the integers.

    The certificate is an integer row vector ``u`` together with an
    integer ``d`` such that ``d`` divides every entry of ``u*A``
    but ``d`` does not divide ``c = (u*b)[0]``. Any integer vector
    ``x`` with ``A*x = b`` would satisfy ``(u*A)*x = c``, whose
    left-hand side is divisible by ``d`` while the right-hand side is
    not, so no such ``x`` can exist. The case ``d = 0`` means
    ``u*A`` is the zero row while ``u*b`` is nonzero.

    Use :meth:`verify` to check the certificate against ``A`` and
    ``b``.
    """

    def __init__(self, u, d, c):
        self.u = u
        self.d = Integer(d)
        self.c = Integer(c)

    def verify(self, A, b):
        """
        Check that this certificate really proves ``A*x = b`` has no
        integer solution. ``b`` must be a single column.
        """
        A = Matrix(A)
        b = Matrix(b)
        if b.cols != 1:
            raise ShapeError("b must be a single column")
        uA = self.u * A
        if self.d == 0:
            if any(e != 0 for e in uA):
                return False
        elif any(not _divides(self.d, e) for e in uA):
            return False
        return not _divides(self.d, (self.u * b)[0])

    def __repr__(self):
        return "DivisibilityContradiction(u=%s, d=%s, c=%s)" % (
            self.u, self.d, self.c)

    def __eq__(self, other):
        if not isinstance(other, DivisibilityContradiction):
            return NotImplemented
        return (self.u, self.d, self.c) == (other.u, other.d, other.c)

    __hash__ = None


def _check_integer_matrix(M, name):
    for e in M:
        if not isinstance(e, Integer):
            raise ValueError(
                "%s must have integer entries, got %r" % (name, e))


def ilinsolve(A, b):
    r"""
    Solve the system $A x = b$ for $x$ over the integers.

    Explanation
    ===========

    Unlike ``linsolve``, which solves over the rationals,
    ``ilinsolve`` describes the complete lattice of integer solutions.
    The system is reduced with the Smith normal decomposition
    $S A T = D$ (computed once with exact integer arithmetic and shared
    by all right-hand sides), where $S$ and $T$ are unimodular and $D$
    is diagonal. With $x = T y$ the system becomes $D y = S b$, which is
    solved by exact division.

    Parameters
    ==========

    A : Matrix, $m \times n$ with integer entries
    b : Matrix, $m \times k$ with integer entries. Each column is one
        right-hand side; columns are solved independently but share the
        single Smith normal decomposition of ``A``.

    Returns
    =======

    (solutions, nullspace_basis)

    solutions : list of length $k$. Entry ``j`` is either an
        $n \times 1$ Matrix ``x0`` with ``A*x0 == b[:, j]``, or a
        :class:`DivisibilityContradiction` certifying that column
        ``j`` has no integer solution.
    nullspace_basis : Matrix, $n \times r$. Its columns are a basis
        (over ``ZZ``) of $\{x \in \mathbb{Z}^n : A x = 0\}$, so
        ``A*nullspace_basis`` is the zero matrix.

    For every solvable column ``j`` the complete set of integer
    solutions is ``x = x0 + nullspace_basis*z`` with ``z`` ranging
    over all integer vectors of length $r$.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.ilinsolve import ilinsolve
    >>> A = Matrix([[2, 1, -1], [8, 4, -4]])
    >>> b = Matrix([[3], [12]])
    >>> (x0,), B = ilinsolve(A, b)
    >>> x0
    Matrix([
    [0],
    [3],
    [0]])
    >>> B
    Matrix([
    [0,  1],
    [1, -2],
    [1,  0]])
    >>> A*x0 == b, A*B == A.zeros(2, 2)
    (True, True)

    A system may have rational solutions but no integer solution; then a
    verifiable :class:`DivisibilityContradiction` is returned instead:

    >>> (bad,), B = ilinsolve(Matrix([[2]]), Matrix([[1]]))
    >>> bad
    DivisibilityContradiction(u=Matrix([[1]]), d=2, c=1)
    >>> bad.verify(Matrix([[2]]), Matrix([[1]]))
    True

    Several right-hand sides share one factorization, and each column
    reports its own solvability:

    >>> sols, B = ilinsolve(Matrix([[2, 4]]), Matrix([[6, 3]]))
    >>> sols[0]
    Matrix([
    [3],
    [0]])
    >>> sols[1]
    DivisibilityContradiction(u=Matrix([[1]]), d=2, c=3)
    """
    A = Matrix(A)
    b = Matrix(b)
    if A.rows != b.rows:
        raise ShapeError(
            "A and b must have the same number of rows, got %d and %d"
            % (A.rows, b.rows))
    _check_integer_matrix(A, "A")
    _check_integer_matrix(b, "b")

    m, n = A.shape
    D, S, T = smith_normal_decomp(A, domain=ZZ)
    diags = [D[i, i] for i in range(min(m, n))]
    nullspace_basis = T[:, len([d for d in diags if d != 0]):]

    Sb = S * b
    solutions = []
    for j in range(b.cols):
        particular = None
        for i in range(m):
            d = diags[i] if i < len(diags) else Integer(0)
            c = Sb[i, j]
            if not _divides(d, c):
                particular = DivisibilityContradiction(S[i, :], d, c)
                break
        if particular is None:
            y = [Sb[i, j] // diags[i] if i < len(diags) and diags[i] != 0
                 else Integer(0) for i in range(n)]
            particular = T * Matrix(y)
        solutions.append(particular)
    return solutions, nullspace_basis
