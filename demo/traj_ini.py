import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import jax
import cvxpy as cp

from solver_utils import solve_or_skip

jax.config.update('jax_enable_x64', True)
from util import linearization as lr
from util import Integrator as it
from scipy.linalg import sqrtm
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
N = mode["N"]
num_obs = mode["num_obs"]
obs = mode["obs"]
obs_r = mode["obs_r"]
rng = np.random.default_rng(987654321)
W_traj_s = rng.uniform(low=-1.0, high=1.0, size=(N, T - 1, nw))
Wy_traj_s = rng.uniform(low=-1.0, high=1.0, size=(N, T - 1, nw_y))
# Wy_traj_s = np.zeros([N, T - 1, nw_y])
# W_traj_s = np.zeros([N, T - 1, nw])

alpha = 0.99


def obj_traj(x_traj, u_traj, dx, du):
    f_traj = 0
    ## Final constraints
    f_traj += 1000 * cp.norm(x_traj[T - 1] + dx[T - 1] - x_des, 1)
    for t in range(T):
        # ## maximize the funnel
        # f_funnel += 10 * s[t]
        if t < T - 1:
            ## for regularization
            f_traj += 1 * cp.sum_squares(du[t])
            f_traj += cp.sum_squares(u_traj[t] + du[t])
            # f_traj += cp.norm(v[t], 1)

    return f_traj


def updt_traj(x_traj, u_traj, A_traj, B_traj, f_traj, iter):
    dx = cp.Variable([T, nx])
    v = cp.Variable([T, nx])
    du = cp.Variable([T - 1, nu])
    constraints = []
    constraints.append(dx[0] == np.zeros(nx))
    for t in range(T - 1):
        x_t = x_traj[t]
        x_tp1 = x_traj[t + 1]
        u_t = u_traj[t]
        dx_t = dx[t]
        dx_tp1 = dx[t + 1]
        du_t = du[t]
        f_t = f_traj[t]
        ## dynamics
        constraints.append(dx_tp1 + x_tp1 == f_t + A_traj[t] @ dx_t + B_traj[t] @ du_t + 0 * np.eye(nx) @ v[t])
        ## obs constraints
        for j in range(num_obs):
            obs_j = obs[j]
            h_j = obs_r ** 2 - LA.norm(x_t[0:2] - obs_j, 2) ** 2
            a = - 2 * (x_t[0:2] - obs_j)
            LHS = h_j + a @ dx_t[0:2]  ## obs constraints
            Q_t = np.eye(3) * 0.6
            Q_t_root = sqrtm(Q_t[0:2, 0:2])
            LHS += LA.norm(Q_t_root @ a, 2)
            # constraints.append(LHS <= s_t[j])
            # constraints.append(LHS <= 0)
            if iter > 1:
                constraints.append(LHS <= 0)

    f_traj = obj_traj(x_traj, u_traj, dx, du)
    problem = cp.Problem(cp.Minimize(f_traj), constraints)
    if not solve_or_skip(problem, context=f"initial trajectory iteration {iter + 1}"):
        return None, None
    print("Initial trajectory cost:", problem.value)
    dx_val = dx.value
    du_val = du.value

    return dx_val, du_val


def traj_gen(x_traj, u_traj, max_state_iter=25):
    W_traj = np.ones([T - 1, nw])
    Wy_traj = np.ones([T - 1, nw_y])

    ## to generate the initial trajectory
    for iter in range(max_state_iter):
        ## get current nonlinear states
        f_traj, _ = jax.vmap(
            lambda x, u, w, wy: it.STEP_JIT_NEW(dt, x, u, np.zeros(nw), np.zeros(nw_y)),
            in_axes=(0, 0, 0, 0)
        )(x_traj[0:T - 1, :], u_traj, W_traj, Wy_traj)
        ## get system matrices
        A_traj, B_traj, G_traj, C_traj, D_traj = lr.linearize_new(x_traj, u_traj, W_traj, Wy_traj)
        dx_val, du_val = updt_traj(x_traj, u_traj, A_traj, B_traj, f_traj, iter)
        if dx_val is None:
            # Keep the last valid nominal trajectory and try the next SCP
            # iteration instead of terminating the whole simulation.
            continue
        x_traj += dx_val
        u_traj += du_val
        if max(LA.norm(dx_val, ord=np.inf), LA.norm(du_val, ord=np.inf)) < 1e-5:
            break

    # Refresh all quantities at the returned iterate; the original code
    # returned the pre-update values from the final SCP iteration.
    f_traj, _ = jax.vmap(
        lambda x, u, w, wy: it.STEP_JIT_NEW(
            dt, x, u, np.zeros(nw), np.zeros(nw_y)
        ),
        in_axes=(0, 0, 0, 0),
    )(x_traj[0:T - 1, :], u_traj, W_traj, Wy_traj)
    A_traj, B_traj, G_traj, C_traj, D_traj = lr.linearize_new(
        x_traj, u_traj, W_traj, Wy_traj
    )
    ini_traj = []
    ini_traj.append(x_traj)
    ini_traj.append(u_traj)
    ini_traj.append(f_traj)
    ini_traj.append(A_traj)
    ini_traj.append(B_traj)
    ini_traj.append(C_traj)
    ini_traj.append(D_traj)
    ini_traj.append(G_traj)
    plt.plot(x_traj[:,0],x_traj[:,1])
    plt.show()
    return ini_traj


def generate_initial_trajectory(max_state_iter=25):
    """Generate the nominal trajectory without doing work at import time."""
    x_traj = np.zeros((T, nx))
    x_traj[0] = mode["x_0"]
    u_traj = np.zeros((T - 1, nu))
    return traj_gen(x_traj, u_traj, max_state_iter=max_state_iter)


if __name__ == "__main__":
    generate_initial_trajectory()
