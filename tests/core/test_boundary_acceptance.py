"""Pure component-wise scaled boundary acceptance contracts."""

from __future__ import annotations

import numpy as np
import pytest

from bvp_core import BVPValidationError
from bvp_core.tolerances import evaluate_boundary_acceptance


@pytest.mark.parametrize(
    ("residual", "expected"),
    [([5e-9], True), ([2e-8], False)],
)
def test_pure_absolute_component_acceptance(residual, expected) -> None:
    check = evaluate_boundary_acceptance(
        residual, boundary_atol=1e-8, boundary_rtol=0.0, boundary_scales=[1.0]
    )

    assert check.success is expected
    np.testing.assert_allclose(check.thresholds, [1e-8])


def test_relative_scale_builds_explicit_threshold_and_ratio() -> None:
    check = evaluate_boundary_acceptance(
        [0.5], boundary_atol=1e-8, boundary_rtol=1e-3, boundary_scales=[1000.0]
    )

    np.testing.assert_allclose(check.thresholds, [1.00000001])
    assert check.component_success.tolist() == [True]
    assert check.max_scaled_ratio == pytest.approx(0.5 / 1.00000001)


def test_one_failed_component_rejects_the_whole_boundary() -> None:
    check = evaluate_boundary_acceptance(
        [5e-9, 2e-8],
        boundary_atol=1e-8,
        boundary_rtol=0.0,
        boundary_scales=[1.0, 1.0],
    )

    assert check.component_success.tolist() == [True, False]
    assert check.success is False
    assert check.max_scaled_ratio == pytest.approx(2.0)


@pytest.mark.parametrize("scales", [[1.0], [0.0, 1.0], [np.inf, 1.0]])
def test_invalid_scale_shape_or_values_are_rejected(scales) -> None:
    with pytest.raises(BVPValidationError, match="boundary_scales"):
        evaluate_boundary_acceptance(
            [0.0, 0.0],
            boundary_atol=1e-8,
            boundary_rtol=1e-3,
            boundary_scales=scales,
        )


@pytest.mark.parametrize("residual", [[np.nan], [np.inf]])
def test_non_finite_residual_is_never_accepted(residual) -> None:
    check = evaluate_boundary_acceptance(
        residual,
        boundary_atol=1e-8,
        boundary_rtol=0.0,
        boundary_scales=[1.0],
    )

    assert check.finite is False
    assert check.success is False
    assert check.component_success.tolist() == [False]
    assert np.isinf(check.scaled_ratios).all()


def test_boundary_diagnostic_arrays_are_read_only() -> None:
    check = evaluate_boundary_acceptance(
        [0.0], boundary_atol=1e-8, boundary_rtol=0.0, boundary_scales=[1.0]
    )

    for array in (
        check.scales,
        check.thresholds,
        check.component_success,
        check.scaled_ratios,
    ):
        assert array.flags.writeable is False
