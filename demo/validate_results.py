"""Numerical checks for the saved joint and decoupled funnel solutions."""

import json
import os

import numpy as np

from lipschitz import lipschitz_estimator
from plotting_utils_new import _simulate_traj
from unicycle_main import RESULTS_FILE, load_results_npz
from util import config

mode = config.current_mode
T = mode["T"]
dt = mode["Tf"] / (T - 1)
nx, nw, nwy, nq = mode["nx"], mode["nw"], mode["nwy"], mode["nq"]
Cq, Dq, Gq = mode["Cq"], mode["Dq"], mode["Gq"]
E = dt * mode["E"]
alpha, beta = mode["alpha"], mode["beta"]
tau_x, tau_y = mode["tau_x"], mode["tau_y"]
vx, vy = mode["vx"], mode["vy"]


def _sym_min_eig(matrix):
    matrix = 0.5 * (matrix + matrix.T)
    return float(np.linalg.eigvalsh(matrix)[0])


def exact_lmi_margins(all_trajs, gamma):
    """Evaluate the unlinearized PSD forms of the paper's control/observer LMIs."""
    A, B, G, C, D = [np.asarray(all_trajs[i]) for i in range(2, 7)]
    Q, P, K, L, sigma = [np.asarray(all_trajs[i]) for i in range(7, 12)]
    ctrl_margins, obs_margins = [], []

    for t in range(T - 1):
        z33 = np.zeros((nx, nx))
        z32 = np.zeros((nx, nw))
        z3q = np.zeros((nx, nq))
        z23 = np.zeros((nw, nx))
        z22 = np.zeros((nw, nw))
        z2q = np.zeros((nw, nq))
        zq3 = np.zeros((nq, nx))
        zq2 = np.zeros((nq, nw))
        zqq = np.zeros((nq, nq))

        acl_q = (A[t] + B[t] @ K[t]) @ Q[t]
        bcl_p = -B[t] @ K[t] @ P[t]
        h1 = (Cq + Dq @ K[t]) @ Q[t]
        h2 = -Dq @ K[t] @ P[t]
        base_ctrl = np.block([
            [(alpha - tau_x) * Q[t], z33, z32, z3q, acl_q.T],
            [z33, sigma[t] * P[t], z32, z3q, bcl_p.T],
            [z23, z23, tau_x * np.eye(nw), z2q, G[t].T],
            [zq3, zq3, zq2, zqq, E.T],
            [acl_q, bcl_p, G[t], E, Q[t + 1]],
        ])
        qc_ctrl = np.block([
            [-gamma[t] ** 2 * h1.T @ h1, -gamma[t] ** 2 * h1.T @ h2,
             -gamma[t] ** 2 * h1.T @ Gq, z3q, z33],
            [-gamma[t] ** 2 * h2.T @ h1, -gamma[t] ** 2 * h2.T @ h2,
             -gamma[t] ** 2 * h2.T @ Gq, z3q, z33],
            [-gamma[t] ** 2 * Gq.T @ h1, -gamma[t] ** 2 * Gq.T @ h2,
             -gamma[t] ** 2 * Gq.T @ Gq, z2q, z23],
            [zq3, zq3, zq2, np.eye(nq) / vx ** 2, zq3],
            [z33, z33, z32, z3q, z33],
        ])
        ctrl_margins.append(_sym_min_eig(base_ctrl + vx * qc_ctrl))

        aobs_p = (A[t] - L[t] @ C[t]) @ P[t]
        ld = -L[t] @ D[t]
        hobs = Cq @ P[t]
        z3wy = np.zeros((nx, nwy))
        z2wy = np.zeros((nw, nwy))
        zwy3 = np.zeros((nwy, nx))
        zwy2 = np.zeros((nwy, nw))
        zwyq = np.zeros((nwy, nq))
        base_obs = np.block([
            [(beta - tau_x - tau_y) * P[t], z32, z3wy, z3q, aobs_p.T],
            [z23, tau_x * np.eye(nw), z2wy, z2q, G[t].T],
            [zwy3, zwy2, tau_y * np.eye(nwy), zwyq, ld.T],
            [zq3, zq2, np.zeros((nq, nwy)), zqq, E.T],
            [aobs_p, G[t], ld, E, P[t + 1]],
        ])
        qc_obs = np.block([
            [-gamma[t] ** 2 * hobs.T @ hobs, -gamma[t] ** 2 * hobs.T @ Gq,
             z3wy, z3q, z33],
            [-gamma[t] ** 2 * Gq.T @ hobs, -gamma[t] ** 2 * Gq.T @ Gq,
             z2wy, z2q, z23],
            [zwy3, zwy2, np.zeros((nwy, nwy)), zwyq, zwy3],
            [zq3, zq2, np.zeros((nq, nwy)), np.eye(nq) / vy ** 2, zq3],
            [z33, z32, z3wy, z3q, z33],
        ])
        obs_margins.append(_sym_min_eig(base_obs + vy * qc_obs))

    return np.asarray(ctrl_margins), np.asarray(obs_margins)


def _max_membership(vectors, shapes):
    result = 0.0
    for t, shape in enumerate(shapes):
        values = np.einsum(
            "bi,ij,bj->b", vectors[:, t], np.linalg.pinv(shape), vectors[:, t]
        )
        result = max(result, float(np.max(values)))
    return result


def validate_method(label, all_trajs):
    gamma = lipschitz_estimator(
        all_trajs[2], all_trajs[3], all_trajs[4], all_trajs[9],
        all_trajs[7], all_trajs[8], all_trajs[0], all_trajs[1],
        sample_count=320, safety_factor=1.20,
    )
    ctrl_margin, obs_margin = exact_lmi_margins(all_trajs, gamma)
    report = {
        "method": label,
        "terminal_error_norm": float(np.linalg.norm(all_trajs[0][-1] - mode["x_des"])),
        "max_nominal_dynamics_defect": float(np.max(np.linalg.norm(
            np.asarray(all_trajs[12]) - np.asarray(all_trajs[0][1:]), axis=1
        ))),
        "gamma_min": float(np.min(gamma)),
        "gamma_max": float(np.max(gamma)),
        "exact_control_lmi_min_eigenvalue": float(np.min(ctrl_margin)),
        "exact_observer_lmi_min_eigenvalue": float(np.min(obs_margin)),
    }
    maximum_membership = 0.0
    minimum_clearance = np.inf

    for simulation_mode in ("controller", "observer", "combined_1", "combined_2"):
        state, _, estimate = _simulate_traj(simulation_mode, all_trajs)
        if simulation_mode == "observer":
            eta = np.zeros_like(estimate[1:])
            error = np.asarray(all_trajs[0])[None, :, :] - estimate[1:]
        else:
            eta = state[1:] - np.asarray(all_trajs[0])[None, :, :]
            error = state[1:] - estimate[1:]
        if simulation_mode != "observer":
            state_membership = _max_membership(
                eta, all_trajs[7]
            )
            report[f"{simulation_mode}_max_state_membership"] = state_membership
            maximum_membership = max(maximum_membership, state_membership)
            distances = np.linalg.norm(
                state[1:, :, None, 0:2] - mode["obs"][None, None, :, :], axis=-1
            ) - mode["obs_r"]
            minimum_clearance = min(minimum_clearance, float(np.min(distances)))
        if simulation_mode != "controller":
            observer_membership = _max_membership(
                error, all_trajs[8]
            )
            report[f"{simulation_mode}_max_observer_membership"] = observer_membership
            maximum_membership = max(maximum_membership, observer_membership)
    report["maximum_tested_membership"] = maximum_membership
    report["minimum_tested_obstacle_clearance"] = minimum_clearance
    report["empirical_invariance_pass"] = bool(maximum_membership <= 1.0 + 1e-6)
    return report


def main():
    joint, separate = load_results_npz(RESULTS_FILE)
    reports = [validate_method("joint", joint), validate_method("decoupled", separate)]
    output_path = os.path.join(os.path.dirname(__file__), "validation_report.json")
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(reports, stream, indent=2)
    print(json.dumps(reports, indent=2))
    print(f"Saved validation report to: {output_path}")


if __name__ == "__main__":
    main()
