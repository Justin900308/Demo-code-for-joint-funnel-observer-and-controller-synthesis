import numpy as np
from numpy import linalg as LA
import jax
import cvxpy as cp
import os
from solver_utils import solve_or_skip

jax.config.update('jax_enable_x64', True)
from util import linearization as lr
from util import Integrator as it
from util import config

mode = config.current_mode
T = mode["T"]
Tf = mode["Tf"]
dt = Tf / (T - 1)
x_des = mode["x_des"]
## system dimensions
nx = mode["nx"]
nu = mode["nu"]
ny = mode["ny"]
nw = mode["nw"]
nw_y = mode["nwy"]
nq = mode["nq"]
## system channels
Dq = mode["Dq"]
Cq = mode["Cq"]
Gq = mode["Gq"]
E = mode["E"] * dt
vx = mode["vx"]
vy = mode["vy"]
# E = np.zeros([nx, nq])  ## linear test case
# vx = 0
# vy = 0
N = mode["N"]
num_obs = mode["num_obs"]
obs = mode["obs"]
obs_r = mode["obs_r"]
rng = np.random.default_rng(987654321)
W_traj_s = rng.uniform(low=-1.0, high=1.0, size=(N, T - 1, nw))
Wy_traj_s = rng.uniform(low=-1.0, high=1.0, size=(N, T - 1, nw_y))
# Wy_traj_s = np.zeros([N, T - 1, nw_y])
# W_traj_s = np.zeros([N, T - 1, nw])

alpha = mode.get("alpha", 0.98)
beta = mode.get("beta", 0.95)
tau_x = mode.get("tau_x", 0.1)
tau_y = mode.get("tau_y", 0.1)


def check_nonlinear_matrix_inequalities(
    A_traj, B_traj, G_traj, C_traj, D_traj,
    Q_traj, P_traj, K_traj, L_traj, sigma_traj, gamma_traj,
    label="Post-solve",
):
    """Evaluate the original (unlinearized) nonlinear PSD matrix inequalities.

    The optimization uses first-order models of products such as ``K @ Q``,
    ``sigma * P``, and ``H.T @ H``.  This diagnostic substitutes the returned
    numerical iterate directly into those products, excludes feasibility
    slacks, and reports the worst minimum eigenvalue over all transitions.
    For the PSD convention used in this file, ``lambda_min < 0`` is a
    violation and ``max(0, -lambda_min)`` is its magnitude.
    """
    controller_min_eigs = np.empty(T - 1)
    observer_min_eigs = np.empty(T - 1)

    for t in range(T - 1):
        A_t = np.asarray(A_traj[t], dtype=float)
        B_t = np.asarray(B_traj[t], dtype=float)
        G_t = np.asarray(G_traj[t], dtype=float)
        C_t = np.asarray(C_traj[t], dtype=float)
        D_t = np.asarray(D_traj[t], dtype=float)
        Q_t = np.asarray(Q_traj[t], dtype=float)
        Q_tp1 = np.asarray(Q_traj[t + 1], dtype=float)
        P_t = np.asarray(P_traj[t], dtype=float)
        P_tp1 = np.asarray(P_traj[t + 1], dtype=float)
        K_t = np.asarray(K_traj[t], dtype=float)
        L_t = np.asarray(L_traj[t], dtype=float)
        sigma_t = float(sigma_traj[t])
        gamma_sq = float(gamma_traj[t]) ** 2

        # Exact controller products (no SCP first-order approximation).
        AQ = (A_t + B_t @ K_t) @ Q_t
        BP = -B_t @ K_t @ P_t
        H1 = (Cq + Dq @ K_t) @ Q_t
        H2 = -Dq @ K_t @ P_t
        ctrl_base = np.block([
            [(alpha - tau_x) * Q_t, np.zeros((nx, nx)), np.zeros((nx, nw)), np.zeros((nx, nq)), AQ.T],
            [np.zeros((nx, nx)), sigma_t * P_t, np.zeros((nx, nw)), np.zeros((nx, nq)), BP.T],
            [np.zeros((nw, nx)), np.zeros((nw, nx)), tau_x * np.eye(nw), np.zeros((nw, nq)), G_t.T],
            [np.zeros((nq, nx)), np.zeros((nq, nx)), np.zeros((nq, nw)), np.zeros((nq, nq)), E.T],
            [AQ, BP, G_t, E, Q_tp1],
        ])
        ctrl_qc = np.block([
            [-gamma_sq * H1.T @ H1, -gamma_sq * H1.T @ H2, -gamma_sq * H1.T @ Gq, np.zeros((nx, nq)), np.zeros((nx, nx))],
            [-gamma_sq * H2.T @ H1, -gamma_sq * H2.T @ H2, -gamma_sq * H2.T @ Gq, np.zeros((nx, nq)), np.zeros((nx, nx))],
            [-gamma_sq * Gq.T @ H1, -gamma_sq * Gq.T @ H2, -gamma_sq * Gq.T @ Gq, np.zeros((nw, nq)), np.zeros((nw, nx))],
            [np.zeros((nq, nx)), np.zeros((nq, nx)), np.zeros((nq, nw)), np.eye(nq) / (vx ** 2), np.zeros((nq, nx))],
            [np.zeros((nx, nx)), np.zeros((nx, nx)), np.zeros((nx, nw)), np.zeros((nx, nq)), np.zeros((nx, nx))],
        ])
        ctrl_mi = ctrl_base + vx * ctrl_qc
        ctrl_mi = 0.5 * (ctrl_mi + ctrl_mi.T)
        controller_min_eigs[t] = np.linalg.eigvalsh(ctrl_mi)[0]

        # Exact observer products (no SCP first-order approximation).
        AP = (A_t - L_t @ C_t) @ P_t
        LD = -L_t @ D_t
        H = Cq @ P_t
        obs_base = np.block([
            [(beta - tau_y - tau_x) * P_t, np.zeros((nx, nw)), np.zeros((nx, nw_y)), np.zeros((nx, nq)), AP.T],
            [np.zeros((nw, nx)), tau_x * np.eye(nw), np.zeros((nw, nw_y)), np.zeros((nw, nq)), G_t.T],
            [np.zeros((nw_y, nx)), np.zeros((nw_y, nw)), tau_y * np.eye(nw_y), np.zeros((nw_y, nq)), LD.T],
            [np.zeros((nq, nx)), np.zeros((nq, nw)), np.zeros((nq, nw_y)), np.zeros((nq, nq)), E.T],
            [AP, G_t, LD, E, P_tp1],
        ])
        obs_qc = np.block([
            [-gamma_sq * H.T @ H, -gamma_sq * H.T @ Gq, np.zeros((nx, nw_y)), np.zeros((nx, nq)), np.zeros((nx, nx))],
            [-gamma_sq * Gq.T @ H, -gamma_sq * Gq.T @ Gq, np.zeros((nw, nw_y)), np.zeros((nw, nq)), np.zeros((nw, nx))],
            [np.zeros((nw_y, nx)), np.zeros((nw_y, nw)), np.zeros((nw_y, nw_y)), np.zeros((nw_y, nq)), np.zeros((nw_y, nx))],
            [np.zeros((nq, nx)), np.zeros((nq, nw)), np.zeros((nq, nw_y)), np.eye(nq) / (vy ** 2), np.zeros((nq, nx))],
            [np.zeros((nx, nx)), np.zeros((nx, nw)), np.zeros((nx, nw_y)), np.zeros((nx, nq)), np.zeros((nx, nx))],
        ])
        obs_mi = obs_base + vy * obs_qc
        obs_mi = 0.5 * (obs_mi + obs_mi.T)
        observer_min_eigs[t] = np.linalg.eigvalsh(obs_mi)[0]

    ctrl_t = int(np.argmin(controller_min_eigs))
    obs_t = int(np.argmin(observer_min_eigs))
    ctrl_min = float(controller_min_eigs[ctrl_t])
    obs_min = float(observer_min_eigs[obs_t])
    print(f"{label} exact nonlinear MI check (PSD form, no relaxation):")
    print(
        f"  controller: worst lambda_min = {ctrl_min:.6e} at t={ctrl_t}; "
        f"maximum violation = {max(0.0, -ctrl_min):.6e}"
    )
    print(
        f"  observer:   worst lambda_min = {obs_min:.6e} at t={obs_t}; "
        f"maximum violation = {max(0.0, -obs_min):.6e}"
    )
    return {
        "controller_min_eigenvalues": controller_min_eigs,
        "observer_min_eigenvalues": observer_min_eigs,
        "controller_worst_index": ctrl_t,
        "observer_worst_index": obs_t,
        "controller_worst_min_eigenvalue": ctrl_min,
        "observer_worst_min_eigenvalue": obs_min,
        "controller_max_violation": max(0.0, -ctrl_min),
        "observer_max_violation": max(0.0, -obs_min),
    }


def obj_funnel(Q_traj, P_traj, K_traj, L_traj, Q, P, K, L, sQ, sP, s_LMI, s_LMI_control, s_LMI_observer_linear,
               s_LMI_observer, sigma_traj,
               sigma):
    f_funnel = 0
    ## Final constraints
    for t in range(T):
        ## minimize funnels
        f_funnel += sQ[t] + sP[t]
        if t < T - 1:
            ## constraints violation
            f_funnel += 1e4 * s_LMI_control[t]
            f_funnel += 1e4 * s_LMI_observer[t]
            f_funnel += sigma[t]

    ## regularization terms
    f_all = f_funnel
    lamb = 0.1
    for t in range(T):
        f_all += lamb * cp.sum_squares(Q[t] - Q_traj[t])
        f_all += lamb * cp.sum_squares(P[t] - P_traj[t])
        if t < T - 1:
            f_all += lamb * cp.sum_squares(K[t] - K_traj[t])
            f_all += lamb * cp.sum_squares(L[t] - L_traj[t])
            f_all += lamb * cp.sum_squares(sigma_traj[t] - sigma[t])
    return f_funnel, f_all


def obj_traj(x_traj, u_traj, dx, du):
    f_traj = 0
    ## Final constraints
    # f_traj += 1000 * cp.norm(x_traj[T - 1] + dx[T - 1] - x_des, 1)
    for t in range(T):
        # ## maximize the funnel
        # f_funnel += 10 * s[t]
        if t < T - 1:
            ## for regularization
            f_traj += 10 * cp.sum_squares(du[t])
            f_traj += cp.sum_squares(u_traj[t] + du[t])
            # f_traj += cp.norm(v[t], 1)

    return f_traj


def joint_joint_funnel_problem(All_trajs, gamma_traj, iter):
    x_traj = All_trajs[0]
    u_traj = All_trajs[1]
    A_traj = All_trajs[2]
    B_traj = All_trajs[3]
    G_traj = All_trajs[4]
    C_traj = All_trajs[5]
    D_traj = All_trajs[6]
    Q_traj = All_trajs[7]
    P_traj = All_trajs[8]
    K_traj = All_trajs[9]
    L_traj = All_trajs[10]
    sigma_traj = All_trajs[11]
    f_traj = All_trajs[12]
    Q_reference = np.asarray(Q_traj)
    P_reference = np.asarray(P_traj)
    K_reference = np.asarray(K_traj)
    L_reference = np.asarray(L_traj)
    sigma_reference = np.asarray(sigma_traj)
    dx = cp.Variable([T, nx])
    v = cp.Variable([T, nx])
    du = cp.Variable([T - 1, nu])
    Q = [cp.Variable((nx, nx), PSD=True) for _ in range(T)]
    P = [cp.Variable((nx, nx), PSD=True) for _ in range(T)]
    K = [cp.Variable((nu, nx)) for _ in range(T - 1)]
    L = [cp.Variable((nx, ny)) for _ in range(T - 1)]
    sQ = cp.Variable(T, nonneg=True)
    sP = cp.Variable(T, nonneg=True)
    sigma = cp.Variable(T - 1, nonneg=True)
    # Nonnegative feasibility relaxations; zero means the paper LMI is met.
    s_LMI = cp.Variable(T - 1, nonneg=True)
    s_LMI_control = cp.Variable(T - 1, nonneg=True)
    s_LMI_observer = cp.Variable(T - 1, nonneg=True)
    s_LMI_observer_linear = cp.Variable(T - 1, nonneg=True)
    constraints = []
    constraints.append(x_traj[T - 1] + dx[T - 1] == x_des)
    constraints.append(dx[0] == np.zeros(nx))
    constraints.append(cp.norm(dx, "inf") <= mode.get("trust_x", 0.6))
    constraints.append(cp.norm(du, "inf") <= mode.get("trust_u", 1.5))
    ## initial funnel constraints
    constraints.append(Q[0] >> 0.3 * np.eye(nx))
    constraints.append(P[0] >> 0.3 * np.eye(nx))
    constraints.append(Q[0] << 0.5 * np.eye(nx))
    constraints.append(P[0] << 0.5 * np.eye(nx))

    constraints.append(Q[T - 1] << 0.5 * np.eye(nx))
    constraints.append(P[T - 1] << 0.5 * np.eye(nx))
    for t in range(T - 1):
        ## for trajectory
        x_t = x_traj[t]
        x_tp1 = x_traj[t + 1]
        u_t = u_traj[t]
        dx_t = dx[t]
        dx_tp1 = dx[t + 1]
        du_t = du[t]
        Ad = A_traj[t]
        Bd = B_traj[t]
        ## for funnel
        Q_t = Q[t]
        Q_tp1 = Q[t + 1]
        P_t = P[t]
        P_tp1 = P[t + 1]
        K_t = K[t]
        L_t = L[t]
        sQ_t = sQ[t]
        sP_t = sP[t]
        sigma_t = sigma[t]
        A_t = A_traj[t]
        B_t = B_traj[t]
        G_t = G_traj[t]
        C_t = C_traj[t]
        D_t = D_traj[t]
        gamma_t = gamma_traj[t]

        ############################ dynamics constraints
        f_t = f_traj[t]
        ## dynamics
        constraints.append(dx_tp1 + x_tp1 == f_t + Ad @ dx_t + Bd @ du_t + 0 * v[t])

        ########################### for funnels
        # Intermediate funnels are optimized, not artificially capped at the
        # terminal-set size.  The old 0.5 I cap made the robust LMIs infeasible
        # under persistent disturbances and forced positive relaxations.
        constraints.append(Q_t >> 1e-6 * np.eye(nx))
        constraints.append(P_t >> 1e-6 * np.eye(nx))
        constraints.append(cp.norm(Q_t - Q_traj[t], "fro") <= mode.get("trust_Q", 0.2))
        constraints.append(cp.norm(P_t - P_traj[t], "fro") <= mode.get("trust_P", 0.2))
        constraints.append(cp.norm(K_t - K_traj[t], "fro") <= mode.get("trust_K", 0.75))
        constraints.append(cp.norm(L_t - L_traj[t], "fro") <= mode.get("trust_L", 0.75))
        constraints.append(cp.abs(sigma_t - sigma_traj[t]) <= mode.get("trust_sigma", 0.3))
        ## funnel size objective
        I = np.eye(nx)
        constraints.append(Q_t << sQ_t * I)
        constraints.append(P_t << sP_t * I)

        ## construct the constant terms from old iteration
        BA_cl_traj_t = (A_t + B_t @ K_traj[t]) @ Q_traj[t]  ## bold A_cl (previous iteration)
        BB_cl_traj_t = -B_t @ K_traj[t] @ P_traj[t]  ## previous iteration
        BA_ob_traj_t = (A_t - L_traj[t] @ C_t) @ P_traj[t]  ## previous iteration

        ## construct the current linear version of BA,BB (exact version)
        BA_cl_t = BA_cl_traj_t + B_t @ (K_t - K_traj[t]) @ Q_traj[t] + (A_t + B_t @ K_traj[t]) @ (Q_t - Q_traj[t])
        BA_ob_t = BA_ob_traj_t - (L_t - L_traj[t]) @ C_t @ P_traj[t] + (A_t - L_traj[t] @ C_t) @ (P_t - P_traj[t])
        BB_cl_t = BB_cl_traj_t - B_t @ (K_t - K_traj[t]) @ P_traj[t] - B_t @ K_traj[t] @ (P_t - P_traj[t])

        # =========================
        # Nonlinear control funnel (baseline + vx * QC)
        # =========================
        LD_t = -L_t @ D_t  # linear in L
        I_nw = np.eye(nw)
        I_nq = np.eye(nq)

        Z_nx_nx = np.zeros((nx, nx))
        Z_nx_nw = np.zeros((nx, nw))
        Z_nx_nq = np.zeros((nx, nq))

        Z_nw_nx = np.zeros((nw, nx))
        Z_nw_nq = np.zeros((nw, nq))

        Z_nq_nx = np.zeros((nq, nx))
        Z_nq_nw = np.zeros((nq, nw))

        # ---- SCP linearization pieces for QC (only used in QC block)
        Hbar_delta = (Cq + Dq @ K_traj[t]) @ Q_traj[t]  # (nq x nx) constant
        Hbar_delta2 = -Dq @ K_traj[t] @ P_traj[t]  # for the control, est coupling

        dH_delta = (
                Dq @ (K_t - K_traj[t]) @ Q_traj[t]
                + (Cq + Dq @ K_traj[t]) @ (Q_t - Q_traj[t])
        )  # (nq x nx) affine

        dH_delta2 = (
                - Dq @ ((K_t - K_traj[t]) @ P_traj[t] + K_traj[t] @ (P_t - P_traj[t]))
        )  # (nq x nx) affine

        HtH_delta_lin = (
                Hbar_delta.T @ Hbar_delta
                + Hbar_delta.T @ dH_delta
                + dH_delta.T @ Hbar_delta
        )  # (nx x nx) affine

        HtH_delta2_lin = (
                Hbar_delta2.T @ Hbar_delta2
                + Hbar_delta2.T @ dH_delta2
                + dH_delta2.T @ Hbar_delta2
        )  # (nx x nx) affine

        # First-order model of H_delta,2^T H_delta,1.  This block is not
        # symmetric, so its two differential terms must not be duplicated.
        HtH_delta21_lin = (
                Hbar_delta2.T @ Hbar_delta
                + dH_delta2.T @ Hbar_delta
                + Hbar_delta2.T @ dH_delta
        )  # (nx x nx) affine

        HtG_delta_lin = (Hbar_delta.T @ Gq) + (dH_delta.T @ Gq)  # (nx x nw) affine

        HtG_delta2_lin = (Hbar_delta2.T @ Gq) + (dH_delta2.T @ Gq)  # (nx x nw) affine

        GtGq = Gq.T @ Gq  # (nw x nw) constant

        # -------------------------
        # Baseline block (no QC)
        # Order: [eta, e, w_x, delta_p, eta_next]
        # -------------------------
        # Equation (23) multiplied by -1, hence the PSD convention below.
        B11 = (alpha - tau_x) * Q_t
        # First-order expansion of sigma_t P_t about the previous iterate.
        B22 = (
                sigma_t * P_traj[t]
                + sigma_traj[t] * P_t
                - sigma_traj[t] * P_traj[t]
        )
        B33 = tau_x * I_nw
        B44 = np.zeros((nq, nq))  # baseline has 0 here (QC will add +I)

        row1 = cp.hstack((B11, Z_nx_nx, Z_nx_nw, Z_nx_nq, BA_cl_t.T))
        row2 = cp.hstack((Z_nx_nx, B22, Z_nx_nw, Z_nx_nq, BB_cl_t.T))
        row3 = cp.hstack((Z_nw_nx, Z_nw_nx, B33, Z_nw_nq, G_t.T))
        row4 = cp.hstack((Z_nq_nx, Z_nq_nx, Z_nq_nw, B44, E.T))
        row5 = cp.hstack((BA_cl_t, BB_cl_t, G_t, E, Q_tp1))

        LMI_base_ctrl_nl = cp.vstack((row1, row2, row3, row4, row5))

        # -------------------------
        # QC block (multiplied by vx)
        # QC contribution in PSD-form:
        #   -gamma^2||delta q||^2 + ||delta p||^2
        # delta q = H_delta * eta + Gq * w_x
        # -------------------------
        Q11 = -(gamma_t ** 2) * HtH_delta_lin
        Q13 = -(gamma_t ** 2) * HtG_delta_lin
        Q21 = -(gamma_t ** 2) * HtH_delta21_lin
        Q22 = - (gamma_t ** 2) * HtH_delta2_lin
        Q23 = - (gamma_t ** 2) * HtG_delta2_lin
        Q33 = -(gamma_t ** 2) * GtGq
        # vx multiplies this whole QC block.  1/vx**2 therefore produces the
        # +I/vx block obtained by negating Eq. (23).
        Q44 = I_nq / (vx ** 2)

        row1 = cp.hstack((Q11, Q21.T, Q13, Z_nx_nq, Z_nx_nx))
        row2 = cp.hstack((Q21, Q22, Q23, Z_nx_nq, Z_nx_nx))
        row3 = cp.hstack((Q13.T, Q23.T, Q33, Z_nw_nq, Z_nw_nx))
        row4 = cp.hstack((Z_nq_nx, Z_nq_nx, Z_nq_nw, Q44, Z_nq_nx))
        row5 = cp.hstack((Z_nx_nx, Z_nx_nx, Z_nx_nw, Z_nx_nq, Z_nx_nx))

        LMI_qc_ctrl_nl = cp.vstack((row1, row2, row3, row4, row5))

        # Final nonlinear control LMI: baseline + vx * QC
        LMI_nl_ctrl = LMI_base_ctrl_nl + vx * LMI_qc_ctrl_nl
        LMI_nl_ctrl = 0.5 * (LMI_nl_ctrl + LMI_nl_ctrl.T)
        constraints.append(
            LMI_nl_ctrl + s_LMI_control[t] * np.eye(3 * nx + nw + nq) >> 0
        )

        # =========================
        # Nonlinear observer funnel (baseline + vy * QC)
        # =========================

        I_nw = np.eye(nw)
        I_nwy = np.eye(nw_y)
        I_nq = np.eye(nq)

        Z_nx_nw = np.zeros((nx, nw))
        Z_nx_nwy = np.zeros((nx, nw_y))
        Z_nx_nq = np.zeros((nx, nq))

        Z_nw_nx = np.zeros((nw, nx))
        Z_nw_nwy = np.zeros((nw, nw_y))
        Z_nw_nq = np.zeros((nw, nq))

        Z_nwy_nx = np.zeros((nw_y, nx))
        Z_nwy_nw = np.zeros((nw_y, nw))
        Z_nwy_nq = np.zeros((nw_y, nq))

        Z_nq_nx = np.zeros((nq, nx))
        Z_nq_nw = np.zeros((nq, nw))
        Z_nq_nwy = np.zeros((nq, nw_y))

        Z_nx_nx = np.zeros((nx, nx))

        # ---- SCP linearization for H_Delta^T H_Delta (only needed in QC block)
        Hbar_Delta = Cq @ P_traj[t]  # (nq x nx) constant
        dH_Delta = Cq @ (P_t - P_traj[t])  # (nq x nx) affine

        HtH_Delta_lin = (
                Hbar_Delta.T @ Hbar_Delta
                + Hbar_Delta.T @ dH_Delta
                + dH_Delta.T @ Hbar_Delta
        )  # (nx x nx) affine

        # Cross term H^T Gq is already affine in P_t (as you noted)
        H_Delta = Cq @ P_t
        HtG_Delta = H_Delta.T @ Gq
        GtGq = Gq.T @ Gq

        # -------------------------
        # Baseline block (no QC)
        # Order: [e, w_x, w_y, Delta_p, e_next]
        # -------------------------
        B11 = (beta - tau_y - tau_x) * P_t
        B22 = tau_x * I_nw
        B33 = tau_y * I_nwy
        B44 = np.zeros((nq, nq))

        row1 = cp.hstack((B11, Z_nx_nw, Z_nx_nwy, Z_nx_nq, BA_ob_t.T))
        row2 = cp.hstack((Z_nw_nx, B22, Z_nw_nwy, Z_nw_nq, G_t.T))
        row3 = cp.hstack((Z_nwy_nx, Z_nwy_nw, B33, Z_nwy_nq, LD_t.T))
        row4 = cp.hstack((Z_nq_nx, Z_nq_nw, Z_nq_nwy, B44, E.T))
        row5 = cp.hstack((BA_ob_t, G_t, LD_t, E, P_tp1))

        LMI_base_obs_nl = cp.vstack((row1, row2, row3, row4, row5))

        # -------------------------
        # QC block (multiplied by vy)
        # PSD-form contribution:
        #   -gamma^2||Delta q||^2 + ||Delta p||^2
        # Delta q = (Cq P) e + Gq w_x
        # -------------------------
        Q11 = -(gamma_t ** 2) * HtH_Delta_lin
        Q12 = -(gamma_t ** 2) * HtG_Delta
        Q22 = -(gamma_t ** 2) * GtGq
        Q44 = I_nq / (vy ** 2)

        row1 = cp.hstack((Q11, Q12, Z_nx_nwy, Z_nx_nq, Z_nx_nx))
        row2 = cp.hstack((Q12.T, Q22, Z_nw_nwy, Z_nw_nq, Z_nw_nx))
        row3 = cp.hstack((Z_nwy_nx, Z_nwy_nw, np.zeros((nw_y, nw_y)), Z_nwy_nq, Z_nwy_nx))
        row4 = cp.hstack((Z_nq_nx, Z_nq_nw, Z_nq_nwy, Q44, Z_nq_nx))
        row5 = cp.hstack((Z_nx_nx, Z_nx_nw, Z_nx_nwy, Z_nx_nq, Z_nx_nx))

        LMI_qc_obs_nl = cp.vstack((row1, row2, row3, row4, row5))

        # Final nonlinear observer LMI: baseline + vy * QC
        LMI_nl_obs = LMI_base_obs_nl + vy * LMI_qc_obs_nl
        LMI_nl_obs = 0.5 * (LMI_nl_obs + LMI_nl_obs.T)
        constraints.append(
            LMI_nl_obs + s_LMI_observer[t] * np.eye(2 * nx + nw + nw_y + nq) >> 0
        )

        ## state constraints 1 (obs constraints)
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a_t = - 2 * (x_t[0:2] - obs_j)
            b_t = a_t @ x_t[0:2] - h_j
            Q2 = Q_t[0:2, 0:2]
            a_row = cp.Constant(a_t).reshape((1, 2), order="C")  # 1×2
            x2 = x_t[0:2].reshape((2, 1))  # 2×1 numpy → 2×1
            phi = (b_t - a_row @ x2) ** 2
            dphi = -2 * (b_t - a_row @ x2) * a_t

            B11 = phi + dphi.T @ dx_t[0:2]  # 1×1
            B12 = a_row @ Q2.T  # 1×2
            B_row1 = cp.hstack((B11, B12))  # 1×3
            B_row2 = cp.hstack((Q2 @ a_row.T, Q2))  # (2×1) hstack (2×2) → 2×3
            B_matrix = cp.vstack((B_row1, B_row2))  # 3×3 PSD
            constraints.append(B_matrix >> 0)
            constraints.append(b_t - a_row @ (x2 + cp.reshape(dx_t[0:2], (2, 1), order="C")) >= 0)

    f_traj = obj_traj(x_traj, u_traj, dx, du)
    f_funnel, f_funnel_all = obj_funnel(Q_traj, P_traj, K_traj, L_traj, Q, P, K, L, sQ, sP, s_LMI, s_LMI_control,
                                        s_LMI_observer_linear,
                                        s_LMI_observer, sigma_traj, sigma)
    f_all = f_traj + f_funnel_all
    problem = cp.Problem(cp.Minimize(f_all), constraints)  ## minimize the regularized obj
    if not solve_or_skip(problem, context="joint trajectory/funnel iteration"):
        return None, np.inf, np.inf
    output_traj = []
    Q_traj = np.zeros([T, nx, nx])
    P_traj = np.zeros([T, nx, nx])
    K_traj = np.zeros([T - 1, nu, nx])
    L_traj = np.zeros([T - 1, nx, ny])
    for t in range(T):
        if t < T - 1:
            K_traj[t] = K[t].value
            L_traj[t] = L[t].value
        Q_traj[t] = Q[t].value
        P_traj[t] = P[t].value
    step = mode.get("funnel_step", 0.5)
    Q_traj = (1.0 - step) * Q_reference + step * Q_traj
    P_traj = (1.0 - step) * P_reference + step * P_traj
    K_traj = (1.0 - step) * K_reference + step * K_traj
    L_traj = (1.0 - step) * L_reference + step * L_traj
    sigma_value = (1.0 - step) * sigma_reference + step * sigma.value
    dx_val = dx.value
    du_val = du.value
    output_traj.append(x_traj + dx_val)
    output_traj.append(u_traj + du_val)
    output_traj.append(Q_traj)
    output_traj.append(P_traj)
    output_traj.append(K_traj)
    output_traj.append(L_traj)
    output_traj.append(sigma_value)
    s_Lmi_val = s_LMI.value
    print(
        "LMI relaxations (control, observer):",
        float(np.sum(s_LMI_control.value)),
        float(np.sum(s_LMI_observer.value)),
    )
    check_nonlinear_matrix_inequalities(
        A_traj, B_traj, G_traj, C_traj, D_traj,
        Q_traj, P_traj, K_traj, L_traj, sigma_value, gamma_traj,
        label="Joint post-solve",
    )
    return output_traj, f_all.value, f_funnel.value


def funnel_gen(
    x_traj, u_traj, Q_traj, P_traj, K_traj, L_traj, gamma_traj,
    sigma_traj=None,
):
    W_traj = np.ones([T - 1, nw])
    Wy_traj = np.ones([T - 1, nw_y])

    # traj_preview_plt(x_traj, u_traj)

    max_funnel_iter = int(os.environ.get("FUNNEL_JOINT_INNER_ITER", "12"))
    true_cost_old = 0
    objective_cost_old = 0
    if sigma_traj is None:
        sigma_traj = np.ones(T - 1) * 0.2
    for iter in range(max_funnel_iter):
        f_traj, _ = jax.vmap(
            lambda x, u, w, wy: it.STEP_JIT_NEW(dt, x, u, np.zeros(nw), np.zeros(nw_y)),
            in_axes=(0, 0, 0, 0)
        )(x_traj[0:T - 1, :], u_traj, W_traj, Wy_traj)
        A_traj, B_traj, G_traj, C_traj, D_traj = lr.linearize_new(x_traj, u_traj, W_traj, Wy_traj)
        All_trajs = []
        All_trajs.append(x_traj)
        All_trajs.append(u_traj)
        All_trajs.append(A_traj)
        All_trajs.append(B_traj)
        All_trajs.append(G_traj)
        All_trajs.append(C_traj)
        All_trajs.append(D_traj)
        All_trajs.append(Q_traj)
        All_trajs.append(P_traj)
        All_trajs.append(K_traj)
        All_trajs.append(L_traj)
        All_trajs.append(sigma_traj)
        All_trajs.append(f_traj)
        output_trajs, true_cost, objective_cost = joint_joint_funnel_problem(All_trajs, gamma_traj, 0)
        if output_trajs is None:
            # Retain the last valid x/u/Q/P/K/L/sigma values and advance the
            # loop.  No CVXPY variable values are read after a failed solve.
            continue
        print("Current funnel iter:", iter + 1, "true_cost:", true_cost)
        ## update the funnel parameters
        ## get current nonlinear states
        x_traj = output_trajs[0]
        u_traj = output_trajs[1]
        Q_traj = output_trajs[2]
        P_traj = output_trajs[3]
        K_traj = output_trajs[4]
        L_traj = output_trajs[5]
        # traj_preview_plt(x_traj, u_traj)
        sigma_traj = output_trajs[6]
        if iter > 2 and np.abs(true_cost - true_cost_old) < 0.1:
            break
        true_cost_old = true_cost  ## update the cost

    # Return the final updated iterate rather than the pre-solve linearization
    # from the last inner iteration.
    f_traj, _ = jax.vmap(
        lambda x, u, w, wy: it.STEP_JIT_NEW(dt, x, u, np.zeros(nw), np.zeros(nw_y)),
        in_axes=(0, 0, 0, 0)
    )(x_traj[0:T - 1, :], u_traj, W_traj, Wy_traj)
    A_traj, B_traj, G_traj, C_traj, D_traj = lr.linearize_new(x_traj, u_traj, W_traj, Wy_traj)
    check_nonlinear_matrix_inequalities(
        A_traj, B_traj, G_traj, C_traj, D_traj,
        Q_traj, P_traj, K_traj, L_traj, sigma_traj, gamma_traj,
        label="Joint final (relinearized trajectory)",
    )
    return [x_traj, u_traj, A_traj, B_traj, G_traj, C_traj, D_traj,
            Q_traj, P_traj, K_traj, L_traj, sigma_traj, f_traj]
