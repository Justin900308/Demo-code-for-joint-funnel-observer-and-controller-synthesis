import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from util import config
from util import Integrator as it
from util import linearization as lr
from plotting_utils_new import compare_methods
from traj_ini import generate_initial_trajectory
from joint_joint_funnel_nonlinear import funnel_gen as joint_funnel_gen
from separate_funnel_nonlinear import trajectory_update, funnel_update
from lipschitz import lipschitz_estimator
# If False, an existing NPZ file is loaded and all optimization calculations are
# skipped. Set True only when you explicitly want to recompute the results.
FORCE_RECOMPUTE = os.environ.get("FUNNEL_FORCE_RECOMPUTE", "0") == "1"

# Plot the loaded/newly-computed results. Set False if you only want to compute
# and save the NPZ file without opening the figures.
PLOT_RESULTS = os.environ.get("FUNNEL_PLOT_RESULTS", "1") == "1"

# Cache file written next to this main script.
script_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(script_dir, "funnel_joint_vs_separate_results.npz")


mode = config.current_mode
T = mode["T"]
Tf = mode["Tf"]
dt = Tf / (T - 1)
print("dt", dt)

nx = mode["nx"]
nu = mode["nu"]
ny = mode["ny"]
nw = mode["nw"]
nw_y = mode["nwy"]
N = mode["N"]


def save_results_npz(file_path, joint_trajs, separate_trajs):
    """Save both complete plotting data lists without using pickle objects."""
    payload = {
        "joint_len": np.array(len(joint_trajs), dtype=int),
        "separate_len": np.array(len(separate_trajs), dtype=int),
    }
    for i, value in enumerate(joint_trajs):
        payload[f"joint_{i}"] = np.asarray(value)
    for i, value in enumerate(separate_trajs):
        payload[f"separate_{i}"] = np.asarray(value)

    np.savez_compressed(file_path, **payload)
    print(f"Saved results to: {file_path}")


def load_results_npz(file_path):
    """Load the two trajectory lists saved by save_results_npz()."""
    with np.load(file_path, allow_pickle=False) as data:
        joint_len = int(data["joint_len"])
        separate_len = int(data["separate_len"])
        joint_trajs = [np.array(data[f"joint_{i}"], copy=True) for i in range(joint_len)]
        separate_trajs = [np.array(data[f"separate_{i}"], copy=True) for i in range(separate_len)]
    return joint_trajs, separate_trajs


def compute_results():
    """Run the original joint and separated calculations and return data."""

    rng = np.random.default_rng(987654321)

    def sample_unit_ball(shape):
        vectors = rng.normal(size=shape)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
        radii = rng.random(shape[:-1] + (1,)) ** (1.0 / shape[-1])
        return vectors * radii

    # The S-procedure LMIs assume Euclidean unit-ball disturbances, not a box.
    W_traj_s = sample_unit_ball((N, T - 1, nw))
    Wy_traj_s = sample_unit_ball((N, T - 1, nw_y))

    Q_traj0 = np.repeat((0.6 * np.eye(nx))[None, :, :], T, axis=0)
    P_traj0 = np.repeat((0.6 * np.eye(nx))[None, :, :], T, axis=0)
    K_traj0 = np.zeros((T - 1, nu, nx))
    L_traj0 = np.zeros((T - 1, nx, ny))

    ini_traj = generate_initial_trajectory()
    # traj_ini returns [x, u, f, A, B, C, D, G]. Both methods use independent
    # copies of the same initialized trajectory.
    x_ini = np.array(ini_traj[0], copy=True)
    u_ini = np.array(ini_traj[1], copy=True)

    x_joint = x_ini.copy()
    u_joint = u_ini.copy()
    Q_joint = Q_traj0.copy()
    P_joint = P_traj0.copy()
    K_joint = K_traj0.copy()
    L_joint = L_traj0.copy()
    sigma_joint = np.ones(T - 1) * 0.2

    x_sep = x_ini.copy()
    u_sep = u_ini.copy()
    Q_sep = Q_traj0.copy()
    P_sep = P_traj0.copy()
    K_sep = K_traj0.copy()
    L_sep = L_traj0.copy()
    sigma_sep = np.ones(T - 1) * 0.2

    W_nom = np.zeros((T - 1, nw))
    Wy_nom = np.zeros((T - 1, nw_y))

    def refresh_linearization(x_traj, u_traj):
        f_traj, _ = jax.vmap(
            lambda x, u, w, wy: it.STEP_JIT_NEW(
                dt, x, u, np.zeros(nw), np.zeros(nw_y)
            ),
            in_axes=(0, 0, 0, 0),
        )(x_traj[0:T - 1], u_traj, W_nom, Wy_nom)
        A_traj, B_traj, G_traj, C_traj, D_traj = lr.linearize_new(
            x_traj, u_traj, W_nom, Wy_nom
        )
        return A_traj, B_traj, G_traj, C_traj, D_traj, f_traj

    def pack_all(x, u, A, B, G, C, D, Q, P, K, L, sigma, f):
        return [x, u, A, B, G, C, D, Q, P, K, L, sigma, f]

    max_iter = int(os.environ.get("FUNNEL_MAX_ITER", "3"))
    separate_inner_iter = int(os.environ.get("FUNNEL_SEPARATE_INNER_ITER", "6"))

    A_joint, B_joint, G_joint, C_joint, D_joint, f_joint = refresh_linearization(
        x_joint, u_joint
    )
    A_sep, B_sep, G_sep, C_sep, D_sep, f_sep = refresh_linearization(x_sep, u_sep)

    All_trajs_joint = None
    All_trajs_sep = None

    # One Lipschitz estimation per outer iteration
    for outer_iter in range(max_iter):
        print(f"\n===== Outer iteration {outer_iter + 1}/{max_iter}: JOINT =====")
        gamma_joint = lipschitz_estimator(
            A_joint, B_joint, G_joint, K_joint, Q_joint, P_joint,
            x_joint, u_joint
        )
        print("Joint gamma:", gamma_joint)

        All_trajs_joint = joint_funnel_gen(
            x_joint, u_joint, Q_joint, P_joint, K_joint, L_joint, gamma_joint,
            sigma_joint,
        )
        x_joint, u_joint = All_trajs_joint[0], All_trajs_joint[1]
        A_joint, B_joint, G_joint = (
            All_trajs_joint[2], All_trajs_joint[3], All_trajs_joint[4]
        )
        C_joint, D_joint = All_trajs_joint[5], All_trajs_joint[6]
        Q_joint, P_joint = All_trajs_joint[7], All_trajs_joint[8]
        K_joint, L_joint = All_trajs_joint[9], All_trajs_joint[10]
        sigma_joint = All_trajs_joint[11]
        f_joint = All_trajs_joint[12]

        print(f"\n===== Outer iteration {outer_iter + 1}/{max_iter}: SEPARATED =====")
        gamma_sep = lipschitz_estimator(
            A_sep, B_sep, G_sep, K_sep, Q_sep, P_sep,
            x_sep, u_sep
        )
        print("Separated gamma:", gamma_sep)

        # This inner iteration is for the Bilevel comparison
        for inner_iter in range(separate_inner_iter):
            print(
                f"Separated inner {inner_iter + 1}/{separate_inner_iter}: "
                "trajectory update"
            )
            All_trajs_sep = pack_all(
                x_sep, u_sep, A_sep, B_sep, G_sep, C_sep, D_sep,
                Q_sep, P_sep, K_sep, L_sep, sigma_sep, f_sep
            )
            # Trajectory step
            x_sep, u_sep, traj_cost = trajectory_update(All_trajs_sep)
            print("Separated trajectory cost:", traj_cost)
            if not np.isfinite(traj_cost):
                print(
                    "Separated trajectory solve failed; skipping the remainder "
                    "of this inner iteration."
                )
                continue

            A_sep, B_sep, G_sep, C_sep, D_sep, f_sep = refresh_linearization(
                x_sep, u_sep
            )
            All_trajs_sep = pack_all(
                x_sep, u_sep, A_sep, B_sep, G_sep, C_sep, D_sep,
                Q_sep, P_sep, K_sep, L_sep, sigma_sep, f_sep
            )

            print(
                f"Separated inner {inner_iter + 1}/{separate_inner_iter}: "
                "funnel update"
            )
            # Separate funnel update
            (
                Q_sep, P_sep, K_sep, L_sep, sigma_sep,
                funnel_cost, raw_funnel_cost
            ) = funnel_update(All_trajs_sep, gamma_sep)
            print(
                "Separated funnel cost:", funnel_cost,
                "raw funnel cost:", raw_funnel_cost
            )
            if not np.isfinite(funnel_cost):
                print(
                    "Separated funnel solve failed; retained the previous "
                    "funnel and continuing."
                )

        A_sep, B_sep, G_sep, C_sep, D_sep, f_sep = refresh_linearization(
            x_sep, u_sep
        )
        All_trajs_sep = pack_all(
            x_sep, u_sep, A_sep, B_sep, G_sep, C_sep, D_sep,
            Q_sep, P_sep, K_sep, L_sep, sigma_sep, f_sep
        )

    All_trajs_joint_plot = All_trajs_joint + [W_traj_s, Wy_traj_s]
    All_trajs_sep_plot = All_trajs_sep + [W_traj_s, Wy_traj_s]
    return All_trajs_joint_plot, All_trajs_sep_plot


def main():
    if os.path.isfile(RESULTS_FILE) and not FORCE_RECOMPUTE:
        print(f"Found cached NPZ: {RESULTS_FILE}")
        print("Skipping trajectory/funnel optimization calculations.")
        joint_plot, separate_plot = load_results_npz(RESULTS_FILE)
    else:
        if FORCE_RECOMPUTE and os.path.isfile(RESULTS_FILE):
            print("FORCE_RECOMPUTE=True: ignoring existing NPZ and recalculating.")
        else:
            print("No cached NPZ found. Running calculations.")

        joint_plot, separate_plot = compute_results()
        save_results_npz(RESULTS_FILE, joint_plot, separate_plot)

    if PLOT_RESULTS:
        for plotting_mode in ("combined_1", "combined_2", "observer", "controller"):
            compare_methods(
                plotting_mode,
                joint_plot,
                separate_plot,
                labels=("Joint", "Decoupled"),
            )
    else:
        print("PLOT_RESULTS=False: plotting skipped.")


if __name__ == "__main__":
    main()
