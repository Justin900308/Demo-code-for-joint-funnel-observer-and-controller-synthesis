import os
import numpy as np
import scipy.linalg as la
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# =========================
# Plot font-size controls
# Change only these values to resize text in every figure.
# =========================
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 16
LABEL_FONT_SIZE = 20
TITLE_FONT_SIZE = 20

plt.rcParams.update({
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "pdf.fonttype": 42,  # Uses TrueType fonts (Type 42) instead of Type 3
    "ps.fonttype": 42,  # Same for PostScript outputs
})

from util import Integrator as it
from util import config

folder_name = "figures"
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, folder_name)
os.makedirs(results_dir, exist_ok=True)

mode = config.current_mode
T = mode["T"]
Tf = mode["Tf"]
dt = Tf / (T - 1)
nx = mode["nx"]
nu = mode["nu"]
nw = mode["nw"]
nwy = mode["nwy"]
N = mode["N"]
num_obs = mode["num_obs"]
obs = mode["obs"]
obs_r = mode["obs_r"]
time_grid = np.linspace(0.0, Tf, T)


def plot_ellip(ellip_t, x_t, color, alpha, Label):
    vals, vecs = LA.eigh(ellip_t)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vmax = vecs[:, 0]
    angle_deg = np.degrees(np.arctan2(vmax[1], vmax[0]))
    return Ellipse(
        xy=(x_t[0], x_t[1]),
        width=2 * np.sqrt(vals[0]),
        height=2 * np.sqrt(vals[1]),
        angle=angle_deg,
        fill=True,
        alpha=alpha,
        facecolor=color,
        edgecolor="k",
        linewidth=1,
        label=Label,
    )


def _shared_legend(fig, axes):
    """Place one deduplicated legend inside the upper-right/right subplot."""
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 0:
        axes_array = axes_array.reshape(1, 1)
    elif axes_array.ndim == 1:
        axes_array = axes_array.reshape(1, -1)

    handles, labels = [], []
    for ax in axes_array.ravel():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)

    if handles:
        # For a multi-row state plot this is the upper-right subplot; for the
        # spatial 1x2 plot this is simply the right-hand subplot.
        right_ax = axes_array[0, -1]
        right_ax.legend(handles, labels, loc="best", fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout()


def _mode_data(modes, x_traj_sim, x_est_sim, Q_traj, P_traj):
    if modes == "controller":
        return x_traj_sim, None, Q_traj, None
    if modes == "observer":
        return x_est_sim, None, P_traj, None
    if modes in ("combined_1", "combined_2"):
        return x_traj_sim, x_est_sim, Q_traj, P_traj
    raise ValueError(f"Unknown plotting mode: {modes}")


def plotting_fcn(method_results, modes):
    """Plot all existing outputs with one method in each column."""
    n_methods = len(method_results)

    # State bounds: same bound calculation, now nx rows x two method columns.
    fig, axes = plt.subplots(
        nx, n_methods, figsize=(6.2 * n_methods, 3.0 * nx), squeeze=False
    )
    for col, result in enumerate(method_results):
        x_traj_sim = result["x_traj_sim"]
        x_est_sim = result["x_est_sim"]
        Q_traj = result["Q_traj"]
        P_traj = result["P_traj"]
        _, _, ellip_traj, _ = _mode_data(modes, x_traj_sim, x_est_sim, Q_traj, P_traj)

        x_Mm = np.zeros((T, nx, 2))
        for t in range(T):
            vals, _ = LA.eigh(ellip_traj[t])
            vals = vals[vals.argsort()[::-1]]
            for i in range(nx):
                # Projection of {z: z^T Q^-1 z <= 1} onto coordinate i.
                radius = np.sqrt(max(ellip_traj[t, i, i], 0.0))
                x_Mm[t, i, 0] = x_traj_sim[0, t, i] + radius
                x_Mm[t, i, 1] = x_traj_sim[0, t, i] - radius

        for i in range(nx):
            ax = axes[i, col]
            for j in range(N + 1):
                ax.plot(
                    time_grid,
                    x_traj_sim[j, :, i],
                    "b-",
                    linewidth=0.8,
                    label="Samples" if j == 0 else None,
                )
            ax.plot(time_grid, x_Mm[:, i, 0], "r", label="State bounds")
            ax.plot(time_grid, x_Mm[:, i, 1], "r")
            if i == nx - 1:
                ax.set_xlabel("time (sec)")
            ax.set_ylabel(f"$x_{i + 1}(m)$" if i < nx - 1 else f"$x_{i + 1}(rad)$")
            ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
            ax.grid(True)
            if i == 0:
                ax.set_title(result["label"])

    _shared_legend(fig, axes)
    fig.savefig(os.path.join(results_dir, f"{modes}_state_bounds_comparison.pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # Bound violation magnitude: same nx-row x method-column arrangement as
    # state_bounds_comparison. Violation is zero inside the state bound and is
    # the amount by which a simulated sample exceeds the upper/lower bound.
    fig, axes = plt.subplots(
        nx, n_methods, figsize=(6.2 * n_methods, 3.0 * nx), squeeze=False
    )

    for col, result in enumerate(method_results):
        x_traj_sim = result["x_traj_sim"]
        x_est_sim = result["x_est_sim"]
        Q_traj = result["Q_traj"]
        P_traj = result["P_traj"]
        _, _, ellip_traj, _ = _mode_data(
            modes, x_traj_sim, x_est_sim, Q_traj, P_traj
        )

        x_Mm = np.zeros((T, nx, 2))
        for t in range(T):
            vals, _ = LA.eigh(ellip_traj[t])
            vals = vals[vals.argsort()[::-1]]
            for i in range(nx):
                radius = np.sqrt(max(ellip_traj[t, i, i], 0.0))
                x_Mm[t, i, 0] = x_traj_sim[0, t, i] + radius
                x_Mm[t, i, 1] = x_traj_sim[0, t, i] - radius

        for i in range(nx):
            ax = axes[i, col]

            # One violation curve for each disturbed sample.
            # nominal trajectory (index 0) is excluded from the violation test.
            violation_all = np.zeros((N, T))
            upper = x_Mm[:, i, 0]
            lower = x_Mm[:, i, 1]

            for j in range(N):
                sample = x_traj_sim[j + 1, :, i]
                upper_violation = np.maximum(sample - upper, 0.0)
                lower_violation = np.maximum(lower - sample, 0.0)
                violation_all[j] = upper_violation + lower_violation

                ax.plot(
                    time_grid,
                    violation_all[j],
                    linewidth=0.8,
                    alpha=0.7,
                    label="Sample violation" if j == 0 else None,
                )

            # Highlight the worst violation across all samples at each time.
            max_violation = np.max(violation_all, axis=0)
            ax.plot(
                time_grid,
                max_violation,
                "k-",
                linewidth=2.0,
                label="Maximum violation",
            )
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)

            if i == nx - 1:
                ax.set_xlabel("time (sec)")
            ax.set_ylabel(
                f"$x_{{{i + 1}}}$ violation (m)"
                if i < nx - 1
                else f"$x_{{{i + 1}}}$ violation (rad)"
            )
            ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
            ax.grid(True)
            if i == 0:
                ax.set_title(result["label"])

    _shared_legend(fig, axes)
    fig.savefig(
        os.path.join(results_dir, f"{modes}_bound_violation_comparison.pdf"),
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    # Spatial funnel plot: one row, two columns, one legend on the right.
    fig, axes = plt.subplots(1, n_methods, figsize=(7.0 * n_methods, 6.0), squeeze=False)
    axes = axes[0]
    theta_grid = np.linspace(0, 2 * np.pi, 101)

    for col, result in enumerate(method_results):
        ax = axes[col]
        x_traj_sim = result["x_traj_sim"]
        x_est_sim = result["x_est_sim"]
        Q_traj = result["Q_traj"]
        P_traj = result["P_traj"]
        x_trajs_plot, x_ests_plot, ellip_traj, ellip_traj_2 = _mode_data(
            modes, x_traj_sim, x_est_sim, Q_traj, P_traj
        )

        for t in range(T):
            theta = x_trajs_plot[0, t, 2]
            leng = 0.4
            heading_vector = np.array([
                [x_trajs_plot[0, t, 0], x_trajs_plot[0, t, 1]],
                [
                    x_trajs_plot[0, t, 0] + leng * np.cos(theta),
                    x_trajs_plot[0, t, 1] + leng * np.sin(theta),
                ],
            ])
            ax.plot(heading_vector[:, 0], heading_vector[:, 1], "g")
            ax.plot(x_trajs_plot[0, t, 0], x_trajs_plot[0, t, 1], "g.")

            for j in range(num_obs):
                x_theta_j = obs[j, 0] + np.cos(theta_grid) * obs_r
                y_theta_j = obs[j, 1] + np.sin(theta_grid) * obs_r
                ax.plot(
                    x_theta_j,
                    y_theta_j,
                    "green",
                    label="Obstacle" if (j == num_obs - 1 and t == T - 1) else None,
                )

            Q_t = ellip_traj[t, 0:2, 0:2]
            x_traj_t = x_trajs_plot[0, t]
            if t == 0:
                ellp1 = plot_ellip(Q_t, x_traj_t, "green", 0.2, "Initial state funnel")
            elif t == T - 1:
                ellp1 = plot_ellip(Q_t, x_traj_t, "green", 0.2, "Final state funnel")
            else:
                ellp1 = plot_ellip(Q_t, x_traj_t, "gray", 0.2, None)
            ax.add_patch(ellp1)

            if modes in ("combined_1", "combined_2"):
                P_t = ellip_traj_2[t, 0:2, 0:2]
                for i in range(N - 1):
                    ax.add_patch(plot_ellip(P_t, x_trajs_plot[i + 1, t], "blue", 0.05, None))

        for i in range(N):
            x_sim_i = x_trajs_plot[i + 1]
            if modes == "observer":
                ax.plot(
                    x_sim_i[:, 0], x_sim_i[:, 1], "r-",
                    label="Est trajs" if i == N - 1 else None,
                )
            else:
                ax.plot(
                    x_sim_i[:, 0], x_sim_i[:, 1], "r-",
                    label="System trajs" if i == N - 1 else None,
                )
            if modes in ("combined_1", "combined_2"):
                ax.plot(
                    x_ests_plot[i + 1, :, 0], x_ests_plot[i + 1, :, 1], "cyan",
                    label="Est trajs" if i == N - 1 else None,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        ax.set_title(result["label"])
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.grid(True)

    _shared_legend(fig, axes)
    fig.savefig(os.path.join(results_dir, f"{modes}_comparison.pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _simulate_traj(modes, All_trajs):
    x_traj = All_trajs[0]
    u_traj = All_trajs[1]
    C_traj = All_trajs[5]
    D_traj = All_trajs[6]
    Q_traj = All_trajs[7]
    P_traj = All_trajs[8]
    K_traj = All_trajs[9]
    L_traj = All_trajs[10]
    W_traj_s = All_trajs[13]
    Wy_traj_s = All_trajs[14]
    x_0 = mode["x_0"]

    x_traj_sim = np.zeros((N + 1, T, nx))
    u_traj_sim = np.zeros((N + 1, T - 1, nu))
    x_est_sim = np.zeros((N + 1, T, nx))

    x_traj_sim[0, 0] = x_0
    for t in range(T - 1):
        x_traj_sim[0, t + 1], _ = it.STEP_NEW(
            dt, x_traj_sim[0, t], u_traj[t], np.zeros(nw), np.zeros(nwy)
        )
    x_est_sim[0] = x_traj_sim[0].copy()

    # Legacy simulation ICs from the first uploaded code.  These intentionally
    # isolate position and heading effects differently in the four plot cases.
    Q_half = np.real_if_close(la.sqrtm(Q_traj[0, 0:2, 0:2]))
    P_half = np.real_if_close(la.sqrtm(P_traj[0, 0:2, 0:2]))
    vals, _ = LA.eigh(Q_traj[0])
    vals = vals[vals.argsort()[::-1]]
    headings = np.linspace(-np.sqrt(max(vals[-1], 0.0)) * 0.8,
                           np.sqrt(max(vals[-1], 0.0)) * 0.8, N)
    theta = np.linspace(0, 2 * np.pi, N)
    x_0_traj_s = np.zeros((N, nx))
    x_0_est_s = np.zeros((N, nx))

    for i in range(N):
        idx = i + 1
        W_traj_i = W_traj_s[i]
        Wy_traj_i = Wy_traj_s[i]

        if modes == "controller":
            x_0_traj_s[i, 0:2] = np.array(
                [np.cos(theta[i]), np.sin(theta[i])]
            ) * 0.85
            x_0_traj_s[i, 0:2] = Q_half @ x_0_traj_s[i, 0:2]
            x_0_traj_s[i] += x_traj_sim[0, 0]
        elif modes == "observer":
            x_0_est_s[i, 0:2] = np.array(
                [np.cos(theta[i]), np.sin(theta[i])]
            ) * 0.85
            x_0_est_s[i, 0:2] = P_half @ x_0_est_s[i, 0:2]
            x_0_est_s[i] += x_traj_sim[0, 0]
        elif modes == "combined_1":
            x_0_traj_s[i, 0:2] = np.array(
                [np.cos(theta[i]), np.sin(theta[i])]
            ) * 0.85
            x_0_traj_s[i, 0:2] = Q_half @ x_0_traj_s[i, 0:2]
            x_0_traj_s[i, 2] = headings[i]
            x_0_traj_s[i] += x_traj_sim[0, 0]
            x_0_est_s[i] += x_traj_sim[0, 0]
        elif modes == "combined_2":
            x_0_traj_s[i, 0:2] = np.array(
                [np.cos(theta[3]), np.sin(theta[3])]
            ) * 0.85
            x_0_traj_s[i, 0:2] = Q_half @ x_0_traj_s[i, 0:2]
            x_0_traj_s[i, 2] = headings[i]
            x_0_traj_s[i] += x_traj_sim[0, 0]
            x_0_est_s[i, 0:2] = np.array(
                [np.cos(theta[i]), np.sin(theta[i])]
            ) * 0.85
            x_0_est_s[i, 0:2] = P_half @ x_0_est_s[i, 0:2]
            x_0_est_s[i] += x_0_traj_s[i]
        else:
            raise ValueError(f"Unknown simulation mode: {modes}")

        for t in range(T - 1):
            W_t = W_traj_i[t]
            Wy_t = Wy_traj_i[t]
            C_t = C_traj[t]
            D_t = D_traj[t]
            if t == 0:
                x_traj_sim[idx, t] = x_0_traj_s[i]
                x_est_sim[idx, t] = x_0_est_s[i]

            if modes == "controller":
                u_t = u_traj[t] + K_traj[t] @ (x_traj_sim[idx, t] - x_traj[t])
                u_traj_sim[idx, t] = u_t
                x_traj_sim[idx, t + 1], _ = it.STEP_NEW(
                    dt, x_traj_sim[idx, t], u_t, W_t, np.zeros(nwy)
                )
            elif modes == "observer":
                y_hat_t = C_t @ x_est_sim[idx, t]
                y_t = C_t @ x_traj[t] + D_t @ Wy_t
                o_t = L_traj[t] @ (y_t - y_hat_t)
                x_est_sim[idx, t + 1], _ = it.STEP_NEW(
                    dt, x_est_sim[idx, t], u_traj[t], np.zeros(nw), np.zeros(nwy)
                )
                x_est_sim[idx, t + 1] += o_t
            else:
                y_hat_t = C_t @ x_est_sim[idx, t]
                y_t = C_t @ x_traj_sim[idx, t] + D_t @ Wy_t
                o_t = L_traj[t] @ (y_t - y_hat_t)
                u_t = u_traj[t] + K_traj[t] @ (x_est_sim[idx, t] - x_traj[t])
                u_traj_sim[idx, t] = u_t
                x_est_sim[idx, t + 1], _ = it.STEP_NEW(
                    dt, x_est_sim[idx, t], u_t, np.zeros(nw), np.zeros(nwy)
                )
                x_est_sim[idx, t + 1] += o_t
                x_traj_sim[idx, t + 1], _ = it.STEP_NEW(
                    dt, x_traj_sim[idx, t], u_t, W_t, Wy_t
                )

    return x_traj_sim, u_traj_sim, x_est_sim


def traj_sim_new(modes, All_trajs, is_test=False, is_multi=True, is_plotting=False):
    """Backward-compatible single-method simulation entry point."""
    x_traj_sim, u_traj_sim, x_est_sim = _simulate_traj(modes, All_trajs)
    if is_plotting:
        plotting_fcn([
            {
                "label": "Method",
                "x_traj_sim": x_traj_sim,
                "x_est_sim": x_est_sim,
                "Q_traj": All_trajs[7],
                "P_traj": All_trajs[8],
            }
        ], modes)
    return x_traj_sim, u_traj_sim


def compare_methods(modes, joint_trajs, separate_trajs, labels=("Joint", "Separated")):
    """Simulate both methods using the same disturbances and compare side by side."""
    joint_x, joint_u, joint_est = _simulate_traj(modes, joint_trajs)
    sep_x, sep_u, sep_est = _simulate_traj(modes, separate_trajs)

    plotting_fcn([
        {
            "label": labels[0],
            "x_traj_sim": joint_x,
            "x_est_sim": joint_est,
            "Q_traj": joint_trajs[7],
            "P_traj": joint_trajs[8],
        },
        {
            "label": labels[1],
            "x_traj_sim": sep_x,
            "x_est_sim": sep_est,
            "Q_traj": separate_trajs[7],
            "P_traj": separate_trajs[8],
        },
    ], modes)
    return (joint_x, joint_u), (sep_x, sep_u)


def traj_preview_plt(x_traj, u_traj):
    fig, ax = plt.subplots()
    theta_grid = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta_grid) * obs_r
    circle_y = np.sin(theta_grid) * obs_r
    x_traj_sim = np.zeros((T, nx))
    x_traj_sim[0] = x_traj[0]
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    for t in range(T - 1):
        x_traj_sim[t + 1], _ = it.STEP_NEW(
            dt, x_traj_sim[t], u_traj[t], np.zeros(nw), np.zeros(nwy)
        )
        ax.plot(x_traj_sim[0:t, 0], x_traj_sim[0:t, 1])
        ax.plot(x_traj[0:t, 0], x_traj[0:t, 1])
        for n in range(num_obs):
            obs_pos = obs[n]
            ax.plot(obs_pos[0] + circle_x, obs_pos[1] + circle_y)
        if t < T - 2:
            plt.pause(0.01)
            ax.clear()
        else:
            plt.show()
