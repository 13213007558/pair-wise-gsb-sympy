"""Test ideals.py code."""
from __future__ import annotations

from sympy import groebner, symbols
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


def _saturate_by_elimination(ring, I, J):
    generators = [ring.to_sympy(f) for f in J.gens if f != ring.zero]
    if not generators:
        return ring.ideal(1)

    auxiliaries = symbols('_saturation_auxiliary:%d' % len(generators))
    generic_generator = sum(
        (auxiliary*generator for auxiliary, generator in zip(
            auxiliaries, generators)), 0)
    t = symbols('_saturation_auxiliary_t')
    basis = groebner(
        [ring.to_sympy(f) for f in I.gens] + [t*generic_generator - 1],
        t, *(auxiliaries + ring.symbols), order='lex')
    if 1 in basis:
        return ring.ideal(1)
    return ring.ideal(*(
        f for f in basis
        if not f.has(t) and not any(f.has(u) for u in auxiliaries)))


def _check_saturation_relations(R, I, J, result):
    ideal_generators = tuple(I.gens)
    power_generators = tuple((J**result.exponent).gens)
    assert len(result.relations) == len(tuple(result.ideal.gens))
    for generator, generator_relations in zip(result.ideal.gens, result.relations):
        assert len(generator_relations) == len(power_generators)
        for (power_generator, coefficients), expected in zip(
                generator_relations, power_generators):
            assert power_generator == expected
            assert len(coefficients) == len(ideal_generators)
            assert sum(
                (R.convert(coefficient)*ideal_generator
                 for coefficient, ideal_generator in zip(
                         coefficients, ideal_generators)),
                R.zero) == generator*power_generator


def test_saturate():
    R = QQ.old_poly_ring(x, y, z)
    cases = [
        (R.ideal(), R.ideal(x, y)),
        (R.ideal(1), R.ideal(x, y)),
        (R.ideal(x), R.ideal(0)),
        (R.ideal(x**2), R.ideal(x)),
        (R.ideal(x*y, x*z), R.ideal(y, z)),
        (R.ideal(x*y, z), R.ideal(x, y)),
        (R.ideal(x**2, y**2), R.ideal(x, y)),
        (R.ideal(x, y), R.ideal(x + y)),
    ]

    for I, J in cases:
        result = I.saturate(J, exponent=True, relations=True)
        assert result.complete is True
        expected = _saturate_by_elimination(R, I, J)
        assert result.ideal == expected
        assert result.ideal == I.quotient(J**result.exponent)
        _check_saturation_relations(R, I, J, result)


def test_saturate_equivalent_generating_sets():
    R = QQ.old_poly_ring(x, y)
    I1 = R.ideal(x**2, x*y)
    I2 = R.ideal(x**2, x*y, x**3)
    J1 = R.ideal(x)
    J2 = R.ideal(x, x + x**2)

    assert I1.saturate(J1) == I2.saturate(J2)
    assert I1.saturate(J1) == _saturate_by_elimination(R, I2, J2)
    assert I2.saturate(J2) == _saturate_by_elimination(R, I1, J1)

    result = I1.saturate(J2, exponent=True)
    assert result.ideal == I1.quotient(J2**result.exponent)


def test_saturate_multiple_generators():
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x*y)
    J = R.ideal(x, y)

    assert I.saturate(J) == R.ideal(x*y)
    assert I.saturate(R.ideal(x)).union(
        I.saturate(R.ideal(y))) == R.ideal(x, y)
    assert I.saturate(R.ideal(x)).intersect(
        I.saturate(R.ideal(y))) == R.ideal(x*y)


def test_saturate_budget():
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x**2, y**2)
    J = R.ideal(x, y)

    partial = I.quotient(J)
    result = I.saturate(
        J, exponent=True, relations=True, max_iterations=1, incomplete=True)
    assert result.complete is False
    assert result.ideal == partial
    assert result.exponent is None
    assert result.relations is None

    raises(NotImplementedError, lambda: I.saturate(J, max_iterations=1))
    raises(ValueError, lambda: I.saturate(
        J, max_iterations=-1, incomplete=True))

    result = I.saturate(
        J, exponent=True, relations=True, max_iterations=0, incomplete=True)
    assert result.ideal == I
    assert result.exponent is None
    assert result.relations is None
    assert result.complete is False


def test_saturate_cross_ring():
    R = QQ.old_poly_ring(x)
    S = QQ.old_poly_ring(y)

    raises(ValueError, lambda: R.ideal(x).saturate(S.ideal(y)))


def test_reduction():
    from sympy.polys.distributedmodules import sdm_nf_buchberger_reduced
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x**5, y)
    e = R.convert(x**3 + y**2)
    assert I.reduce_element(e) == e
    assert I.reduce_element(e, NF=sdm_nf_buchberger_reduced) == R.convert(x**3)
