import numpy as np
from numpy import linalg as LA
import cvxpy as cp

from solver_utils import solve_or_skip

from util import config

mode = config.current_mode
T = mode["T"]
Tf = mode["Tf"]
dt = Tf / (T - 1)
x_des = mode["x_des"]

nx = mode["nx"]
nu = mode["nu"]
ny = mode["ny"]
nw = mode["nw"]
nw_y = mode["nwy"]
nq = mode["nq"]

Dq = mode["Dq"]
Cq = mode["Cq"]
Gq = mode["Gq"]
E = mode["E"] * dt
vx = mode["vx"]
vy = mode["vy"]

num_obs = mode["num_obs"]
obs = mode["obs"]
obs_r = mode["obs_r"]
alpha = mode.get("alpha", 0.99)
beta = mode.get("beta", 0.99)
tau_x = mode.get("tau_x", 0.05)
tau_y = mode.get("tau_y", 0.05)


def trajectory_update(All_trajs):
    """Update only the nominal trajectory/control while the funnel is fixed."""
    x_traj = All_trajs[0]
    u_traj = All_trajs[1]
    A_traj = All_trajs[2]
    B_traj = All_trajs[3]
    Q_traj = All_trajs[7]
    f_traj = All_trajs[12]

    dx = cp.Variable((T, nx))
    v = cp.Variable((T, nx))
    du = cp.Variable((T - 1, nu))

    constraints = [
        x_traj[T - 1] + dx[T - 1] == x_des,
        dx[0] == np.zeros(nx),
    ]
    constraints.append(cp.norm(dx, "inf") <= mode.get("trust_x", 0.6))
    constraints.append(cp.norm(du, "inf") <= mode.get("trust_u", 1.5))

    for t in range(T - 1):
        x_t = x_traj[t]
        x_tp1 = x_traj[t + 1]
        dx_t = dx[t]
        dx_tp1 = dx[t + 1]
        du_t = du[t]

        # Same linearized dynamics as the joint problem.
        constraints.append(
            dx_tp1 + x_tp1
            == f_traj[t] + A_traj[t] @ dx_t + B_traj[t] @ du_t + 0 * v[t]
        )

        # Same coupled obstacle/funnel LMI, now with Q fixed.
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a_t = -2 * (x_t[0:2] - obs_j)
            b_t = a_t @ x_t[0:2] - h_j
            Q2 = Q_traj[t, 0:2, 0:2]

            a_row = cp.Constant(a_t).reshape((1, 2), order="C")
            x2 = x_t[0:2].reshape((2, 1))
            phi = (b_t - a_row @ x2) ** 2
            dphi = -2 * (b_t - a_row @ x2) * a_t

            B11 = phi + dphi.T @ dx_t[0:2]
            B12 = a_row @ Q2.T
            B_row1 = cp.hstack((B11, B12))
            B_row2 = cp.hstack((Q2 @ a_row.T, Q2))
            B_matrix = cp.vstack((B_row1, B_row2))
            constraints.append(B_matrix >> 0)
            constraints.append(
                b_t - a_row @ (x2 + cp.reshape(dx_t[0:2], (2, 1), order="C")) >= 0
            )

    # Same trajectory objective used in the joint problem.
    f_traj_obj = 0
    for t in range(T - 1):
        f_traj_obj += 10 * cp.sum_squares(du[t])
        f_traj_obj += cp.sum_squares(u_traj[t] + du[t])

    problem = cp.Problem(cp.Minimize(f_traj_obj), constraints)
    if not solve_or_skip(problem, context="separated trajectory iteration"):
        return x_traj.copy(), u_traj.copy(), np.inf

    return x_traj + dx.value, u_traj + du.value, problem.value


def funnel_update(All_trajs, gamma_traj):
    """Update only Q/P/K/L/sigma while the nominal trajectory is fixed."""
    x_traj = All_trajs[0]
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

    Q = [cp.Variable((nx, nx), PSD=True) for _ in range(T)]
    P = [cp.Variable((nx, nx), PSD=True) for _ in range(T)]
    K = [cp.Variable((nu, nx)) for _ in range(T - 1)]
    L = [cp.Variable((nx, ny)) for _ in range(T - 1)]
    sQ = cp.Variable(T, nonneg=True)
    sP = cp.Variable(T, nonneg=True)
    sigma = cp.Variable(T - 1, nonneg=True)
    s_LMI = cp.Variable(T - 1, nonneg=True)
    s_LMI_control = cp.Variable(T - 1, nonneg=True)
    s_LMI_observer = cp.Variable(T - 1, nonneg=True)
    s_LMI_observer_linear = cp.Variable(T - 1, nonneg=True)

    constraints = [
        Q[0] >> 0.3 * np.eye(nx),
        P[0] >> 0.3 * np.eye(nx),
        Q[0] << 0.5 * np.eye(nx),
        P[0] << 0.5 * np.eye(nx),
        Q[T - 1] << 0.5 * np.eye(nx),
        P[T - 1] << 0.5 * np.eye(nx),
    ]

    for t in range(T - 1):
        x_t = x_traj[t]
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

        constraints.append(Q_t >> 1e-6 * np.eye(nx))
        constraints.append(P_t >> 1e-6 * np.eye(nx))
        constraints.append(cp.norm(Q_t - Q_traj[t], "fro") <= mode.get("trust_Q", 0.2))
        constraints.append(cp.norm(P_t - P_traj[t], "fro") <= mode.get("trust_P", 0.2))
        constraints.append(cp.norm(K_t - K_traj[t], "fro") <= mode.get("trust_K", 0.75))
        constraints.append(cp.norm(L_t - L_traj[t], "fro") <= mode.get("trust_L", 0.75))
        constraints.append(cp.abs(sigma_t - sigma_traj[t]) <= mode.get("trust_sigma", 0.3))
        constraints.append(Q_t << sQ_t * np.eye(nx))
        constraints.append(P_t << sP_t * np.eye(nx))

        # Same SCP product linearizations as the joint problem.
        BA_cl_traj_t = (A_t + B_t @ K_traj[t]) @ Q_traj[t]
        BB_cl_traj_t = -B_t @ K_traj[t] @ P_traj[t]
        BA_ob_traj_t = (A_t - L_traj[t] @ C_t) @ P_traj[t]

        BA_cl_t = (
            BA_cl_traj_t
            + B_t @ (K_t - K_traj[t]) @ Q_traj[t]
            + (A_t + B_t @ K_traj[t]) @ (Q_t - Q_traj[t])
        )
        BA_ob_t = (
            BA_ob_traj_t
            - (L_t - L_traj[t]) @ C_t @ P_traj[t]
            + (A_t - L_traj[t] @ C_t) @ (P_t - P_traj[t])
        )
        BB_cl_t = (
            BB_cl_traj_t
            - B_t @ (K_t - K_traj[t]) @ P_traj[t]
            - B_t @ K_traj[t] @ (P_t - P_traj[t])
        )

        # =========================
        # Nonlinear control funnel
        # =========================
        LD_t = -L_t @ D_t
        I_nw = np.eye(nw)
        I_nq = np.eye(nq)

        Z_nx_nx = np.zeros((nx, nx))
        Z_nx_nw = np.zeros((nx, nw))
        Z_nx_nq = np.zeros((nx, nq))
        Z_nw_nx = np.zeros((nw, nx))
        Z_nw_nq = np.zeros((nw, nq))
        Z_nq_nx = np.zeros((nq, nx))
        Z_nq_nw = np.zeros((nq, nw))

        Hbar_delta = (Cq + Dq @ K_traj[t]) @ Q_traj[t]
        Hbar_delta2 = -Dq @ K_traj[t] @ P_traj[t]

        dH_delta = (
            Dq @ (K_t - K_traj[t]) @ Q_traj[t]
            + (Cq + Dq @ K_traj[t]) @ (Q_t - Q_traj[t])
        )
        dH_delta2 = -Dq @ (
            (K_t - K_traj[t]) @ P_traj[t]
            + K_traj[t] @ (P_t - P_traj[t])
        )

        HtH_delta_lin = (
            Hbar_delta.T @ Hbar_delta
            + Hbar_delta.T @ dH_delta
            + dH_delta.T @ Hbar_delta
        )
        HtH_delta2_lin = (
            Hbar_delta2.T @ Hbar_delta2
            + Hbar_delta2.T @ dH_delta2
            + dH_delta2.T @ Hbar_delta2
        )
        HtH_delta21_lin = (
            Hbar_delta2.T @ Hbar_delta
            + dH_delta2.T @ Hbar_delta
            + Hbar_delta2.T @ dH_delta
        )
        HtG_delta_lin = Hbar_delta.T @ Gq + dH_delta.T @ Gq
        HtG_delta2_lin = Hbar_delta2.T @ Gq + dH_delta2.T @ Gq
        GtGq = Gq.T @ Gq

        B11 = (alpha - tau_x) * Q_t
        B22 = (
            sigma_t * P_traj[t]
            + sigma_traj[t] * P_t
            - sigma_traj[t] * P_traj[t]
        )
        B33 = tau_x * I_nw
        B44 = np.zeros((nq, nq))

        row1 = cp.hstack((B11, Z_nx_nx, Z_nx_nw, Z_nx_nq, BA_cl_t.T))
        row2 = cp.hstack((Z_nx_nx, B22, Z_nx_nw, Z_nx_nq, BB_cl_t.T))
        row3 = cp.hstack((Z_nw_nx, Z_nw_nx, B33, Z_nw_nq, G_t.T))
        row4 = cp.hstack((Z_nq_nx, Z_nq_nx, Z_nq_nw, B44, E.T))
        row5 = cp.hstack((BA_cl_t, BB_cl_t, G_t, E, Q_tp1))
        LMI_base_ctrl_nl = cp.vstack((row1, row2, row3, row4, row5))

        Q11 = -(gamma_t ** 2) * HtH_delta_lin
        Q13 = -(gamma_t ** 2) * HtG_delta_lin
        Q21 = -(gamma_t ** 2) * HtH_delta21_lin
        Q22 = -(gamma_t ** 2) * HtH_delta2_lin
        Q23 = -(gamma_t ** 2) * HtG_delta2_lin
        Q33 = -(gamma_t ** 2) * GtGq
        Q44 = I_nq / (vx ** 2)

        row1 = cp.hstack((Q11, Q21.T, Q13, Z_nx_nq, Z_nx_nx))
        row2 = cp.hstack((Q21, Q22, Q23, Z_nx_nq, Z_nx_nx))
        row3 = cp.hstack((Q13.T, Q23.T, Q33, Z_nw_nq, Z_nw_nx))
        row4 = cp.hstack((Z_nq_nx, Z_nq_nx, Z_nq_nw, Q44, Z_nq_nx))
        row5 = cp.hstack((Z_nx_nx, Z_nx_nx, Z_nx_nw, Z_nx_nq, Z_nx_nx))
        LMI_qc_ctrl_nl = cp.vstack((row1, row2, row3, row4, row5))

        LMI_nl_ctrl = LMI_base_ctrl_nl + vx * LMI_qc_ctrl_nl
        LMI_nl_ctrl = 0.5 * (LMI_nl_ctrl + LMI_nl_ctrl.T)
        constraints.append(
            LMI_nl_ctrl + s_LMI_control[t] * np.eye(3 * nx + nw + nq) >> 0
        )

        # =========================
        # Nonlinear observer funnel
        # =========================
        I_nwy = np.eye(nw_y)
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

        Hbar_Delta = Cq @ P_traj[t]
        dH_Delta = Cq @ (P_t - P_traj[t])
        HtH_Delta_lin = (
            Hbar_Delta.T @ Hbar_Delta
            + Hbar_Delta.T @ dH_Delta
            + dH_Delta.T @ Hbar_Delta
        )
        H_Delta = Cq @ P_t
        HtG_Delta = H_Delta.T @ Gq
        GtGq = Gq.T @ Gq

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

        LMI_nl_obs = LMI_base_obs_nl + vy * LMI_qc_obs_nl
        LMI_nl_obs = 0.5 * (LMI_nl_obs + LMI_nl_obs.T)
        constraints.append(
            LMI_nl_obs + s_LMI_observer[t] * np.eye(2 * nx + nw + nw_y + nq) >> 0
        )

        # Same obstacle/funnel LMI with the trajectory fixed (dx = 0).
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a_t = -2 * (x_t[0:2] - obs_j)
            b_t = a_t @ x_t[0:2] - h_j
            Q2 = Q_t[0:2, 0:2]

            a_row = cp.Constant(a_t).reshape((1, 2), order="C")
            x2 = x_t[0:2].reshape((2, 1))
            phi = (b_t - a_row @ x2) ** 2
            B11 = phi
            B12 = a_row @ Q2.T
            B_row1 = cp.hstack((B11, B12))
            B_row2 = cp.hstack((Q2 @ a_row.T, Q2))
            B_matrix = cp.vstack((B_row1, B_row2))
            constraints.append(B_matrix >> 0)
            constraints.append(b_t - a_row @ x2 >= 0)

    # Same funnel objective and regularization as the joint problem.
    f_funnel = 0
    for t in range(T):
        f_funnel += sQ[t] + sP[t]
        if t < T - 1:
            f_funnel += 1e4 * s_LMI_control[t]
            f_funnel += 1e4 * s_LMI_observer[t]
            f_funnel += sigma[t]

    f_all = f_funnel
    lamb = 0.1
    for t in range(T):
        f_all += lamb * cp.sum_squares(Q[t] - Q_traj[t])
        f_all += lamb * cp.sum_squares(P[t] - P_traj[t])
        if t < T - 1:
            f_all += lamb * cp.sum_squares(K[t] - K_traj[t])
            f_all += lamb * cp.sum_squares(L[t] - L_traj[t])
            f_all += lamb * cp.sum_squares(sigma_traj[t] - sigma[t])

    problem = cp.Problem(cp.Minimize(f_all), constraints)
    if not solve_or_skip(problem, context="separated funnel iteration"):
        return (
            Q_traj.copy(), P_traj.copy(), K_traj.copy(), L_traj.copy(),
            sigma_traj.copy(), np.inf, np.inf,
        )

    Q_new = np.array([Q[t].value for t in range(T)])
    P_new = np.array([P[t].value for t in range(T)])
    K_new = np.array([K[t].value for t in range(T - 1)])
    L_new = np.array([L[t].value for t in range(T - 1)])
    step = mode.get("funnel_step", 0.5)
    Q_new = (1.0 - step) * Q_traj + step * Q_new
    P_new = (1.0 - step) * P_traj + step * P_new
    K_new = (1.0 - step) * K_traj + step * K_new
    L_new = (1.0 - step) * L_traj + step * L_new
    sigma_new = (1.0 - step) * sigma_traj + step * sigma.value

    print(
        "Separated LMI relaxations (control, observer):",
        float(np.sum(s_LMI_control.value)),
        float(np.sum(s_LMI_observer.value)),
    )
    return Q_new, P_new, K_new, L_new, sigma_new, f_all.value, f_funnel.value
