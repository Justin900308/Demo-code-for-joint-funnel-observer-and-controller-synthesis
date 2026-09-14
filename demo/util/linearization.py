from .Integrator import RK4, STEP_NEW
import jax
from .config import current_mode

T = current_mode["T"]
Tf = current_mode["Tf"]
dt = Tf / (T - 1)


def linearization_fun(integrator, dt, x_t, u_t, W_t) -> tuple:
    f = lambda x, u, W: integrator(dt, x, u, W)  ## f has two outputs, state and measurements
    # Compute the Jacobian of f(x, u),y(x,w) with respect to x (A,C matrix)
    A_k, C_k = jax.jacobian(lambda x: f(x, u_t, W_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    B_k, _ = jax.jacobian(lambda u: f(x_t, u, W_t))(u_t)
    # Compute the Jacobian of y(x, w) with respect to w (D matrix)
    _, D_k = jax.jacobian(lambda w: f(x_t, u_t, w))(W_t)

    return A_k, B_k, C_k, D_k


def linearization_fun_new(integrator, dt, x_t, u_t, W_t, Wy_t) -> tuple:
    f = lambda x, u, W, Wy: integrator(dt, x, u, W, Wy)  ## f has two outputs, state and measurements
    # Compute the Jacobian of f(x, u),y(x,w) with respect to x (A,C matrix)
    A_k, C_k = jax.jacobian(lambda x: f(x, u_t, W_t, Wy_t))(x_t)
    # Compute the Jacobian of f(x, u) with respect to u (B matrix)
    B_k, _ = jax.jacobian(lambda u: f(x_t, u, W_t, Wy_t))(u_t)
    # Compute the Jacobian of y(x, w) with respect to w (G matrix)
    G_k, _ = jax.jacobian(lambda w: f(x_t, u_t, w, Wy_t))(W_t)
    # Compute the Jacobian of y(x, w) with respect to wy (D matrix)
    _, D_k = jax.jacobian(lambda wy: f(x_t, u_t, W_t, wy))(Wy_t)
    ## A,B,G for system
    ## C, D for observer
    return A_k, B_k, G_k, C_k, D_k


linearization_jit = jax.jit(linearization_fun, static_argnums=0)
linearization_jit_new = jax.jit(linearization_fun_new, static_argnums=0)


def linearize_new(x_traj, u_traj, W_traj, Wy_traj):
    [A_list, B_list, G_list, C_list, D_list] = jax.vmap(
        lambda x, u, w, wy: linearization_jit_new(STEP_NEW, dt, x, u, w, wy),
        in_axes=(0, 0, 0, 0)
    )(x_traj[0:T - 1, :], u_traj, W_traj, Wy_traj)

    return A_list, B_list, G_list, C_list, D_list


def linearize(x_traj, u_traj, W_traj):
    [A_list, B_list, C_list, D_list] = jax.vmap(
        lambda x, u, W: linearization_jit(RK4, dt, x, u, W),
        in_axes=(0, 0, 0)
    )(x_traj[0:T - 1, :], u_traj, W_traj)

    return A_list, B_list, C_list, D_list
