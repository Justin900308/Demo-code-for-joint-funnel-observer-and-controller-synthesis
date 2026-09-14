import jax.numpy as jnp
from util import config

mode = config.current_mode
C = mode["C"]
D = mode["D"]
G = mode["G"]


def unicycle(x_t: jnp.array, u_t: jnp.array, W_t: jnp.array) -> jnp.array:
    ## state: px, py, theta
    ## control: v, omega
    theta = x_t[2]
    v = u_t[0]
    omega = u_t[1]
    # Keep the simulated dynamics identical to the G matrix used in the LMI.
    x_dot = jnp.array([v * jnp.cos(theta),
                       v * jnp.sin(theta),
                       omega]) + jnp.asarray(G) @ W_t
    return x_dot


def double_integrator(x_t: jnp.array, u_t: jnp.array, w_t: jnp.array) -> jnp.array:
    ## state: x,y,u,v
    ## control: ax,ay
    A = jnp.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    B = jnp.array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 1]])
    x_dot = A @ x_t + B @ u_t
    return x_dot


def double_integrator_new(x_t: jnp.array, u_t: jnp.array, w_t: jnp.array) -> jnp.array:
    ## state: x,y,u,v
    ## control: ax,ay
    A = jnp.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])
    B = jnp.array([[0, 0],
                   [0, 0],
                   [1, 0],
                   [0, 1]])
    x_dot = A @ x_t + B @ u_t + G @ w_t ## for the system disturbance
    return x_dot
