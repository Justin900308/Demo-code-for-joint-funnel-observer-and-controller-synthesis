import jax.numpy as jnp
from .dynamics import unicycle
import jax
from . import config

mode = config.current_mode
n = mode["nx"]
m = mode["nu"]
C = mode["C"]
D = mode["D"]


def _measurement(x_t, Wy_t):
    """Discrete measurement y_k=C x_k+D w^y_k used by the observer LMI."""
    return C @ x_t + D @ Wy_t


def RK4(dt: jnp.array, x_t: jnp.array, u_t: jnp.array, W_t: jnp.array):
    """RK4 state step, retained for numerical comparisons."""
    k1x = unicycle(x_t, u_t, W_t)
    k2x = unicycle(x_t + dt * k1x / 2, u_t, W_t)
    k3x = unicycle(x_t + dt * k2x / 2, u_t, W_t)
    k4x = unicycle(x_t + dt * k3x, u_t, W_t)
    x_tp1 = x_t + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    return x_tp1, _measurement(x_t, W_t)


def RK4_new(dt: jnp.array, x_t: jnp.array, u_t: jnp.array,
            W_t: jnp.array, Wy_t: jnp.array):
    """RK4 state step with independent process and measurement disturbances."""
    k1x = unicycle(x_t, u_t, W_t)
    k2x = unicycle(x_t + dt * k1x / 2, u_t, W_t)
    k3x = unicycle(x_t + dt * k2x / 2, u_t, W_t)
    k4x = unicycle(x_t + dt * k3x, u_t, W_t)
    x_tp1 = x_t + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    return x_tp1, _measurement(x_t, Wy_t)


def Euler_new(dt: jnp.array, x_t: jnp.array, u_t: jnp.array,
              W_t: jnp.array, Wy_t: jnp.array):
    """Euler model matching the paper's discrete Cq/Dq/E factorization."""
    return x_t + dt * unicycle(x_t, u_t, W_t), _measurement(x_t, Wy_t)


RK_jit = jax.jit(RK4)
RK_jit_new = jax.jit(RK4_new)
Euler_jit_new = jax.jit(Euler_new)


def Euler(dt: jnp.array, x_t: jnp.array, u_t: jnp.array, W_t: jnp.array):
    return x_t + dt * unicycle(x_t, u_t, W_t)


STEP_NEW = Euler_new if mode.get("integrator", "euler").lower() == "euler" else RK4_new
STEP_JIT_NEW = Euler_jit_new if mode.get("integrator", "euler").lower() == "euler" else RK_jit_new
