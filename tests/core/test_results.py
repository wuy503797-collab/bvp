"""Contract tests for the structured public result model."""

from __future__ import annotations

import numpy as np
import pytest

from bvp_core.results import BVPResult


def _legacy_result(*, success: bool) -> dict:
    return {
        "success": success,
        "status": "success" if success else "boundary_residual_too_large",
        "message": "accepted" if success else "boundary residual too large",
        "method": "shooting",
        "p_opt": np.array([1.0]),
        "t": np.array([0.0, 1.0]),
        "y": np.array([[1.0, np.e]]),
        "sol": object(),
        "ivp_success": True,
        "ivp_status": 0,
        "ivp_message": "completed",
        "ivp_t_final": 1.0,
        "optimizer_success": True,
        "algorithm_success": True,
        "finite_success": True,
        "boundary_success": success,
        "boundary_residual": np.array([0.0 if success else 1.0]),
        "boundary_residual_norm": 0.0 if success else 1.0,
        "boundary_atol": 1e-8,
        "boundary_rtol": 0.0,
        "boundary_acceptance": "l2_norm <= boundary_atol",
        "iterations": 6,
        "solver_metadata": {
            "optimizer": "root/hybr",
            "root": {"success": True, "nfev": np.int64(6)},
        },
    }


@pytest.mark.parametrize("success", [True, False])
def test_result_preserves_success_and_failure_contract(success: bool) -> None:
    legacy = _legacy_result(success=success)
    result = BVPResult.from_legacy_dict(legacy)
    compatible = result.to_dict()

    assert result.success is success
    assert compatible["status"] == legacy["status"]
    assert compatible["message"] == legacy["message"]
    assert compatible["ivp_status"] == 0
    assert compatible["boundary_success"] is success
    assert compatible["residual_norm"] == legacy["boundary_residual_norm"]
    assert compatible["solver_metadata"]["root"]["nfev"] == 6
    assert compatible["sol"] is legacy["sol"]
    np.testing.assert_array_equal(compatible["p_opt"], legacy["p_opt"])
    np.testing.assert_array_equal(compatible["y"], legacy["y"])


def test_result_arrays_are_read_only_and_to_dict_returns_copies() -> None:
    result = BVPResult.from_legacy_dict(_legacy_result(success=True))

    assert result.p_opt.flags.writeable is False
    assert result.t.flags.writeable is False
    assert result.y.flags.writeable is False
    assert result.boundary_residual.flags.writeable is False
    with pytest.raises(ValueError):
        result.p_opt[0] = 2.0

    first = result.to_dict()
    second = result.to_dict()
    first["p_opt"][0] = 9.0
    first["y"][0, 0] = 9.0

    assert result.p_opt[0] == 1.0
    assert result.y[0, 0] == 1.0
    assert second["p_opt"][0] == 1.0
    assert second["y"][0, 0] == 1.0


def test_result_rejects_unexplainable_solver_metadata() -> None:
    legacy = _legacy_result(success=True)
    legacy["solver_metadata"] = {"opaque": object()}

    with pytest.raises(TypeError, match="plain diagnostic data"):
        BVPResult.from_legacy_dict(legacy)
