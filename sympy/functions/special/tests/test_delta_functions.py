from __future__ import annotations
from sympy.core.numbers import (I, nan, oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.function import diff
from sympy.functions.elementary.complexes import (adjoint, conjugate, sign, transpose)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.delta_functions import (
    DiracDelta, Heaviside, distributional_diff)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.simplify.simplify import signsimp, simplify


from sympy.testing.pytest import raises

from sympy.core.expr import unchanged

from sympy.core.function import ArgumentIndexError


x, y = symbols('x y')
i = symbols('t', nonzero=True)
j = symbols('j', positive=True)
k = symbols('k', negative=True)

def test_DiracDelta():
    assert DiracDelta(1) == 0
    assert DiracDelta(5.1) == 0
    assert DiracDelta(-pi) == 0
    assert DiracDelta(5, 7) == 0
    assert DiracDelta(x, 0) == DiracDelta(x)
    assert DiracDelta(i) == 0
    assert DiracDelta(j) == 0
    assert DiracDelta(k) == 0
    assert DiracDelta(nan) is nan
    assert DiracDelta(0).func is DiracDelta
    assert DiracDelta(x).func is DiracDelta
    # FIXME: this is generally undefined @ x=0
    #         But then limit(Delta(c)*Heaviside(x),x,-oo)
    #         need's to be implemented.
    # assert 0*DiracDelta(x) == 0

    assert adjoint(DiracDelta(x)) == DiracDelta(x)
    assert adjoint(DiracDelta(x - y)) == DiracDelta(x - y)
    assert conjugate(DiracDelta(x)) == DiracDelta(x)
    assert conjugate(DiracDelta(x - y)) == DiracDelta(x - y)
    assert transpose(DiracDelta(x)) == DiracDelta(x)
    assert transpose(DiracDelta(x - y)) == DiracDelta(x - y)

    assert DiracDelta(x).diff(x) == DiracDelta(x, 1)
    assert DiracDelta(x, 1).diff(x) == DiracDelta(x, 2)

    assert DiracDelta(x).is_simple(x) is True
    assert DiracDelta(3*x).is_simple(x) is True
    assert DiracDelta(x**2).is_simple(x) is False
    assert DiracDelta(sqrt(x)).is_simple(x) is False
    assert DiracDelta(x).is_simple(y) is False

    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True, wrt=x) == (
        DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    assert DiracDelta(2*x) != DiracDelta(x)  # scaling property
    assert DiracDelta(x) == DiracDelta(-x)  # even function
    assert DiracDelta(-x, 2) == DiracDelta(x, 2)
    assert DiracDelta(-x, 1) == -DiracDelta(x, 1)  # odd deriv is odd
    assert DiracDelta(-oo*x) == DiracDelta(oo*x)
    assert DiracDelta(x - y) != DiracDelta(y - x)
    assert signsimp(DiracDelta(x - y) - DiracDelta(y - x)) == 0

    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True) == (
            DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    raises(ArgumentIndexError, lambda: DiracDelta(x).fdiff(2))
    raises(ValueError, lambda: DiracDelta(x, -1))
    raises(ValueError, lambda: DiracDelta(I))
    raises(ValueError, lambda: DiracDelta(2 + 3*I))


def test_heaviside():
    assert Heaviside(-5) == 0
    assert Heaviside(1) == 1
    assert Heaviside(0) == S.Half

    assert Heaviside(0, x) == x
    assert unchanged(Heaviside,x, nan)
    assert Heaviside(0, nan) == nan

    h0  = Heaviside(x, 0)
    h12 = Heaviside(x, S.Half)
    h1  = Heaviside(x, 1)

    assert h0.args == h0.pargs == (x, 0)
    assert h1.args == h1.pargs == (x, 1)
    assert h12.args == (x, S.Half)
    assert h12.pargs == (x,) # default 1/2 suppressed

    assert adjoint(Heaviside(x)) == Heaviside(x)
    assert adjoint(Heaviside(x - y)) == Heaviside(x - y)
    assert conjugate(Heaviside(x)) == Heaviside(x)
    assert conjugate(Heaviside(x - y)) == Heaviside(x - y)
    assert transpose(Heaviside(x)) == Heaviside(x)
    assert transpose(Heaviside(x - y)) == Heaviside(x - y)

    assert Heaviside(x).diff(x) == DiracDelta(x)
    assert Heaviside(x + I).is_Function is True
    assert Heaviside(I*x).is_Function is True

    raises(ArgumentIndexError, lambda: Heaviside(x).fdiff(2))
    raises(ValueError, lambda: Heaviside(I))
    raises(ValueError, lambda: Heaviside(2 + 3*I))


def test_rewrite():
    x, y = Symbol('x', real=True), Symbol('y')
    assert Heaviside(x).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (Heaviside(0), Eq(x, 0)), (1, True)))
    assert Heaviside(y).rewrite(Piecewise) == (
        Piecewise((0, y < 0), (Heaviside(0), Eq(y, 0)), (1, True)))
    assert Heaviside(x, y).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (y, Eq(x, 0)), (1, True)))
    assert Heaviside(x, 0).rewrite(Piecewise) == (
        Piecewise((0, x <= 0), (1, True)))
    assert Heaviside(x, 1).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (1, True)))
    assert Heaviside(x, nan).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True)))

    assert Heaviside(x).rewrite(sign) == \
        Heaviside(x, H0=Heaviside(0)).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(Heaviside(0), S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (Heaviside(0), True)), True)
        )

    assert Heaviside(y).rewrite(sign) == Heaviside(y)
    assert Heaviside(x, S.Half).rewrite(sign) == (sign(x)+1)/2
    assert Heaviside(x, y).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(y, S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (y, True)), True)
        )

    assert DiracDelta(y).rewrite(Piecewise) == Piecewise((DiracDelta(0), Eq(y, 0)), (0, True))
    assert DiracDelta(y, 1).rewrite(Piecewise) == DiracDelta(y, 1)
    assert DiracDelta(x - 5).rewrite(Piecewise) == (
        Piecewise((DiracDelta(0), Eq(x - 5, 0)), (0, True)))

    assert (x*DiracDelta(x - 10)).rewrite(SingularityFunction) == x*SingularityFunction(x, 10, -1)
    assert 5*x*y*DiracDelta(y, 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, 0, -2)
    assert DiracDelta(0).rewrite(SingularityFunction) == SingularityFunction(0, 0, -1)
    assert DiracDelta(0, 1).rewrite(SingularityFunction) == SingularityFunction(0, 0, -2)

    assert Heaviside(x).rewrite(SingularityFunction) == SingularityFunction(x, 0, 0)
    assert 5*x*y*Heaviside(y + 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, -1, 0)
    assert ((x - 3)**3*Heaviside(x - 3)).rewrite(SingularityFunction) == (x - 3)**3*SingularityFunction(x, 3, 0)
    assert Heaviside(0).rewrite(SingularityFunction) == S.Half


def test_distributional_diff():
    # a jump discontinuity contributes a DiracDelta term whose
    # coefficient is the right limit minus the left limit
    assert distributional_diff(Piecewise((0, x < 0), (1, True)), x) == DiracDelta(x)
    assert distributional_diff(Piecewise((0, x < 1), (2, True)), x) == 2*DiracDelta(x - 1)
    # the function value at the break point does not affect the result
    assert distributional_diff(
        Piecewise((0, x < 0), (5, Eq(x, 0)), (2, True)), x) == 2*DiracDelta(x)
    # several break points
    ramp = Piecewise((0, x < 0), (x, x < 1), (1, True))
    assert distributional_diff(ramp, x) == Piecewise(
        (0, x < 0), (1, And(0 < x, x < 1)), (0, True))
    assert distributional_diff(ramp, x, 2) == DiracDelta(x) - DiracDelta(x - 1)
    # a continuous function whose derivative jumps: no DiracDelta in the
    # first derivative, but differentiating again exposes the jump
    g = Piecewise((-x, x < 0), (x, True))
    assert distributional_diff(g, x) == Piecewise((-1, x < 0), (1, True))
    assert distributional_diff(g, x, 2) == 2*DiracDelta(x)
    assert distributional_diff(g, x, 3) == 2*DiracDelta(x, 1)
    # repeated application composes with the n argument
    assert distributional_diff(distributional_diff(g, x), x) == 2*DiracDelta(x)
    # rational function pieces
    assert distributional_diff(Piecewise((0, x < 1), (1/x, True)), x) == \
        Piecewise((0, x < 1), (-1/x**2, True)) + DiracDelta(x - 1)
    # non-piecewise input is differentiated in the ordinary sense
    assert distributional_diff(x**3 + 2*x, x) == 3*x**2 + 2
    assert distributional_diff(x**2, x, 2) == 2
    # n = 0 returns the input unchanged
    assert distributional_diff(ramp, x, 0) == ramp
    # DiracDelta terms with x-dependent coefficients are reduced using
    # x*DiracDelta(x - a) = a*DiracDelta(x - a)
    assert distributional_diff(x*DiracDelta(x - 1), x) == DiracDelta(x - 1, 1)
    assert distributional_diff(x*DiracDelta(x - 1, 1), x) == \
        DiracDelta(x - 1, 2) - DiracDelta(x - 1, 1)


def test_distributional_diff_errors():
    a = symbols('a')
    # a pole inside a piece is not a jump discontinuity
    raises(ValueError, lambda: distributional_diff(
        Piecewise((0, x < 0), (1/(x - 2), x < 3), (0, True)), x))
    # a pole at a break point gives non-finite one-sided limits
    raises(ValueError, lambda: distributional_diff(
        Piecewise((0, x < 0), (1/x, True)), x))
    # pieces must be polynomials or rational functions
    raises(ValueError, lambda: distributional_diff(
        Piecewise((0, x < 0), (sin(x), True)), x))
    # the pieces must cover the whole real line
    raises(ValueError, lambda: distributional_diff(Piecewise((1, x < 0)), x))
    # break points must be comparable real numbers
    raises(ValueError, lambda: distributional_diff(
        Piecewise((0, x < a), (1, True)), x))
    raises(ValueError, lambda: distributional_diff(
        Piecewise((0, x < 0), (1, True)), x, -1))


def test_distributional_diff_integration_by_parts():
    # the defining identity of the distributional derivative:
    # integrate(T'*phi) = -integrate(T*phi') for any compactly
    # supported smooth test function phi
    phi = Piecewise(((1 - x**2)**2, And(x > -1, x < 1)), (0, True))
    f = Piecewise((1, x < 0), (x**2, x < 2), (1/x, True))
    df = distributional_diff(f, x)
    assert simplify(integrate(df*phi, (x, -oo, oo))
        + integrate(f*diff(phi, x), (x, -oo, oo))) == 0
    # second derivative, checked against a C^2 test function:
    # integrate(T''*phi) = integrate(T*phi'')
    phi2 = Piecewise(((1 - x**2)**3, And(x > -1, x < 1)), (0, True))
    d2f = distributional_diff(f, x, 2)
    assert simplify(integrate(d2f*phi2, (x, -oo, oo))
        - integrate(f*diff(phi2, x, 2), (x, -oo, oo))) == 0
