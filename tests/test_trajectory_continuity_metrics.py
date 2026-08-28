import numpy as np
import torch

from scripts_maniskill.eval_trajectory_continuity import (
    ContinuityPolicyProxy,
    compute_continuity_summary,
)


def test_continuity_summary_detects_cbf_jump():
    q_current = np.zeros((1, 2), dtype=np.float32)
    q_no_cbf = np.array([[[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]], dtype=np.float32)
    q_cbf = q_no_cbf.copy()
    q_cbf[0, 1] += np.array([0.4, -0.4], dtype=np.float32)

    summary = compute_continuity_summary(
        {
            "q_current": q_current,
            "q_cbf": q_cbf,
            "q_no_cbf": q_no_cbf,
            "qp_active": np.array([[False, True, False]]),
        }
    )

    assert summary["num_chunks"] == 1
    assert summary["horizon"] == 3
    assert summary["arm_dof"] == 2
    assert summary["max_abs_joint_step_cbf_rad"] > summary["max_abs_joint_step_no_cbf_rad"]
    assert summary["max_second_difference_ratio_cbf_over_no_cbf"] > 1.0
    assert summary["max_cbf_correction_variation_rad"] > 0.0
    assert summary["worst_chunk_index"] == 0


def test_continuity_summary_is_identity_when_cbf_does_nothing():
    q_current = np.array([[0.1, -0.2]], dtype=np.float32)
    q = np.array([[[0.2, -0.1], [0.3, 0.0], [0.4, 0.1]]], dtype=np.float32)

    summary = compute_continuity_summary(
        {
            "q_current": q_current,
            "q_cbf": q,
            "q_no_cbf": q.copy(),
            "qp_active": np.zeros((1, 3), dtype=bool),
        }
    )

    assert np.isclose(
        summary["max_abs_joint_step_cbf_rad"],
        summary["max_abs_joint_step_no_cbf_rad"],
    )
    assert np.isclose(summary["max_cbf_correction_variation_rad"], 0.0)
    assert np.isclose(summary["qp_active_fraction_final_denoise"], 0.0)


class _FakePolicy:
    def __init__(self):
        self.guidance_method = "cbf"
        self.arm_dof = 2
        self.guidance_grad_clip = 0.1
        self._last_joint_traj = None
        self._cbf_qp_fn = self._solve_qp

    @staticmethod
    def _solve_qp(jac, grad_h, h_value, constraint_scale):
        del jac, grad_h, constraint_scale
        rhs = torch.clamp(0.07 - h_value, min=0.0)
        feasible = torch.ones_like(rhs, dtype=torch.bool)
        denom = torch.ones_like(rhs)
        dq = torch.stack([rhs, -rhs], dim=-1)
        return dq, rhs, denom, feasible

    def predict_action(self, *args, **kwargs):
        del args, kwargs
        base = torch.randn((1, 3, 2), dtype=torch.float32)
        if self.guidance_method == "cbf":
            h = torch.tensor([[0.08, 0.05, 0.08]], dtype=torch.float32)
            jac = torch.zeros((1, 3, 6, 2), dtype=torch.float32)
            grad = torch.zeros((1, 3, 2), dtype=torch.float32)
            dq, _, _, _ = self._cbf_qp_fn(jac, grad, h, torch.tensor(1.0))
            base = base + dq
        grip = torch.zeros((1, 3, 1), dtype=torch.float32)
        self._last_joint_traj = torch.cat([base, grip], dim=-1)
        return {"joint_action_pred": self._last_joint_traj.clone()}


def test_policy_proxy_restores_rng_and_main_output():
    torch.manual_seed(123)
    _ = torch.randn((1, 3, 2))
    expected_next_random = torch.randn((1,))

    torch.manual_seed(123)
    policy = _FakePolicy()
    proxy = ContinuityPolicyProxy(policy, torch)
    result = proxy.predict_action(current_joint_angles=torch.zeros((1, 2)))
    actual_next_random = torch.randn((1,))

    assert torch.allclose(actual_next_random, expected_next_random)
    assert policy.guidance_method == "cbf"
    assert torch.allclose(policy._last_joint_traj, result["joint_action_pred"])
    assert len(proxy.records) == 1
    assert proxy.records[0]["qp_active"].tolist() == [False, True, False]
    assert not np.allclose(proxy.records[0]["q_cbf"], proxy.records[0]["q_no_cbf"])
