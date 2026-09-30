"""Test ideals.py code."""
from __future__ import annotations

from sympy.polys import QQ, ilex
from sympy.abc import x, y, z
from sympy.testing.pytest import raises


def test_ideal_operations():
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x)
    J = R.ideal(y)
    S = R.ideal(x*y)
    T = R.ideal(x, y)

    assert not (I == J)
    assert I == I

    assert I.union(J) == T
    assert I + J == T
    assert I + T == T

    assert not I.subset(T)
    assert T.subset(I)

    assert I.product(J) == S
    assert I*J == S
    assert x*J == S
    assert I*y == S
    assert R.convert(x)*J == S
    assert I*R.convert(y) == S

    assert not I.is_zero()
    assert not J.is_whole_ring()

    assert R.ideal(x**2 + 1, x).is_whole_ring()
    assert R.ideal() == R.ideal(0)
    assert R.ideal().is_zero()

    assert T.contains(x*y)
    assert T.subset([x, y])

    assert T.in_terms_of_generators(x) == [R(1), R(0)]

    assert T**0 == R.ideal(1)
    assert T**1 == T
    assert T**2 == R.ideal(x**2, y**2, x*y)
    assert I**5 == R.ideal(x**5)


def test_exceptions():
    I = QQ.old_poly_ring(x).ideal(x)
    J = QQ.old_poly_ring(y).ideal(1)
    raises(ValueError, lambda: I.union(x))
    raises(ValueError, lambda: I + J)
    raises(ValueError, lambda: I * J)
    raises(ValueError, lambda: I.union(J))
    assert (I == J) is False
    assert I != J


def test_zero_generators_do_not_change_ideal():
    R = QQ.old_poly_ring(x)

    assert R.ideal(0, x) == R.ideal(x)
    assert R.ideal(0, 0) == R.ideal()
    assert R.ideal(0, x, x) == R.ideal(x)


def test_nontriv_global():
    R = QQ.old_poly_ring(x, y, z)

    def contains(I, f):
        return R.ideal(*I).contains(f)

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**3)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y**2)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x**4 + y**3 + 2*z*y*x)
    assert contains([x + y + z, x*y + x*z + y*z, x*y*z], x*y*z)
    assert contains([x, 1 + x + y, 5 - 7*y], 1)
    assert contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**3)
    assert not contains(
        [x**3 + y**3, y**3 + z**3, z**3 + x**3, x**2*y + x**2*z + y**2*z],
        x**2 + y**2)

    # compare local order
    assert not contains([x*(1 + x + y), y*(1 + z)], x)
    assert not contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_nontriv_local():
    R = QQ.old_poly_ring(x, y, z, order=ilex)

    def contains(I, f):
        return R.ideal(*I).contains(f)

    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x**2 + y, x**2 + x], x - y)
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


def test_intersection():
    R = QQ.old_poly_ring(x, y, z)
    # SCA, example 1.8.11
    assert R.ideal(x, y).intersect(R.ideal(y**2, z)) == R.ideal(y**2, y*z, x*z)

    assert R.ideal(x, y).intersect(R.ideal()).is_zero()

    R = QQ.old_poly_ring(x, y, z, order="ilex")
    assert R.ideal(x, y).intersect(R.ideal(y**2 + y**2*z, z + z*x**3*y)) == \
        R.ideal(y**2, y*z, x*z)


def test_quotient():
    # SCA, example 1.8.13
    R = QQ.old_poly_ring(x, y, z)
    assert R.ideal(x, y).quotient(R.ideal(y**2, z)) == R.ideal(x, y)


def _saturation_by_elimination(Igens, Jgens, gens):
    # Compute I : J**oo independently of Ideal.saturate, using the
    # Rabinowitsch trick: I : J**oo is the intersection over the
    # generators g of J of (I, 1 - t*g) intersected with QQ[gens].
    from sympy import Dummy, groebner
    t = Dummy('t')
    R = QQ.old_poly_ring(*gens)
    sat = R.ideal(1)
    for g in Jgens:
        G = groebner(list(Igens) + [1 - t*g], t, *gens, domain=QQ)
        sat = sat.intersect(R.ideal(*[e for e in G.exprs if not e.has(t)]))
    return sat


def test_saturate():
    R = QQ.old_poly_ring(x, y)

    I = R.ideal(x*y)
    assert I.saturate(R.ideal(x)) == R.ideal(y)
    assert I.saturate(R.ideal(x), exponent=True) == (R.ideal(y), 1)

    # Saturating by each generator of J separately and combining the
    # results does not give the saturation by J.
    assert I.saturate(R.ideal(x)).union(I.saturate(R.ideal(y))) == R.ideal(x, y)
    assert I.saturate(R.ideal(x, y)) == I
    assert I.saturate(R.ideal(x, y), exponent=True) == (I, 0)

    assert R.ideal(x**2, y**2).saturate(R.ideal(x, y), exponent=True) == (
        R.ideal(1), 3)

    # zero and unit ideals
    assert R.ideal().saturate(R.ideal(x), exponent=True) == (R.ideal(), 0)
    assert R.ideal(x).saturate(R.ideal(), exponent=True) == (R.ideal(1), 1)
    assert R.ideal(x).saturate(R.ideal(1), exponent=True) == (R.ideal(x), 0)
    assert R.ideal(1).saturate(R.ideal(x), exponent=True) == (R.ideal(1), 0)

    # the same ideals written with different generators
    assert R.ideal(x*y, x**2).saturate(R.ideal(x)) == \
        R.ideal(x*y + x**2, x**2).saturate(R.ideal(x, x**2))

    # ideals of different rings are rejected
    raises(ValueError, lambda: I.saturate(QQ.old_poly_ring(y).ideal(y)))
    raises(ValueError, lambda: I.saturate(x))

    # computation budget: report incomplete if the quotient sequence
    # has not stabilized
    J = R.ideal(x, y)
    raises(RuntimeError, lambda: R.ideal(x**2, y**2).saturate(J, max_iterations=2))
    assert R.ideal(x**2, y**2).saturate(J, max_iterations=3).is_whole_ring()


def test_saturate_against_elimination():
    R = QQ.old_poly_ring(x, y)
    for Igens, Jgens in [
            ([x*y], [x]),
            ([x*y], [x, y]),
            ([x**2, x*y], [x]),
            ([x**2, y**2], [x, y]),
            ([x**2*y, x*y**2], [x + y]),
            ([], [x]),
            ([1], [x, y]),
            ([x], [1])]:
        I = R.ideal(*Igens)
        assert I.saturate(R.ideal(*Jgens)) == \
            _saturation_by_elimination(Igens, Jgens, (x, y))


def test_saturate_coefficients():
    R = QQ.old_poly_ring(x, y)

    def check(Igens, Jgens):
        I = R.ideal(*Igens)
        J = R.ideal(*Jgens)
        sat, n, evidence = I.saturate(J, exponent=True, coefficients=True)
        Igens_ring = list(I.gens)
        Jn_gens = list((J**n).gens)
        sat_gens = list(sat.gens)
        assert len(evidence) == len(sat_gens)
        for h, rels in zip(sat_gens, evidence):
            assert len(rels) == len(Jn_gens)
            for (p, c), q in zip(rels, Jn_gens):
                assert p == q
                assert len(c) == len(Igens_ring)
                assert h*p == sum(ci*gi for ci, gi in zip(c, Igens_ring))
        return sat, n

    assert check([x*y, x**2], [x]) == (R.ideal(1), 2)
    assert check([x*y], [x, y]) == (R.ideal(x*y), 0)
    assert check([x**2, y**2], [x, y])[0].is_whole_ring()
    assert check([x], []) == (R.ideal(1), 1)

    # coefficients alone also returns a tuple
    sat, evidence = R.ideal(x*y).saturate(R.ideal(x), coefficients=True)
    assert sat == R.ideal(y)
    assert len(evidence) == len(list(sat.gens))


def test_reduction():
    from sympy.polys.distributedmodules import sdm_nf_buchberger_reduced
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x**5, y)
    e = R.convert(x**3 + y**2)
    assert I.reduce_element(e) == e
    assert I.reduce_element(e, NF=sdm_nf_buchberger_reduced) == R.convert(x**3)
