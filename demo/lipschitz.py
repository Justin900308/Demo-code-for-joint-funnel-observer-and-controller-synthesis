"""Local incremental-nonlinearity bound used by the funnel IQCs."""

import jax
import numpy as np

from util import Integrator, config

jax.config.update("jax_enable_x64", True)

mode = config.current_mode
T = mode["T"]
Tf = mode["Tf"]
dt = Tf / (T - 1)
nx = mode["nx"]
nw = mode["nw"]
nw_y = mode["nwy"]
Cq = np.asarray(mode["Cq"], dtype=float)
Dq = np.asarray(mode["Dq"], dtype=float)
Gq = np.asarray(mode["Gq"], dtype=float)
E_d = dt * np.asarray(mode["E"], dtype=float)


def _psd_sqrt(matrix):
    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    values, vectors = np.linalg.eigh(matrix)
    if values.min() < -1e-7:
        raise ValueError(f"Funnel matrix is not PSD (lambda_min={values.min():.3e}).")
    return vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))


def _unit_directions(rng, count, dimension):
    """Deterministic axes plus random unit-sphere directions."""
    axes = np.vstack((np.eye(dimension), -np.eye(dimension)))
    random = rng.normal(size=(max(count - len(axes), 0), dimension))
    random /= np.maximum(np.linalg.norm(random, axis=1, keepdims=True), 1e-12)
    return np.vstack((axes, random))[:count]


def _max_ratio(delta_p, delta_q):
    numerator = np.linalg.norm(delta_p, axis=1)
    denominator = np.linalg.norm(delta_q, axis=1)
    singular = (denominator <= 1e-10) & (numerator > 1e-8)
    if np.any(singular):
        raise RuntimeError(
            "The chosen Cq/Dq/Gq channel cannot bound the simulated nonlinear "
            "remainder. Use the Euler model or derive stage-augmented RK4 channels."
        )
    valid = denominator > 1e-10
    return float(np.max(numerator[valid] / denominator[valid], initial=0.0))


def lipschitz_estimator(
    A_list_sim,
    B_list_sim,
    G_traj,
    K_traj,
    Q_traj,
    P_traj,
    x_traj_sim,
    u_traj_sim,
    *_legacy_measurement_arguments,
    sample_count=320,
    safety_factor=1.20,
):
    """Estimate one conservative gamma_k for both controller and observer IQCs.

    This evaluates the full combined error model, uses Cq/Dq/Gq, and maps the
    discrete residual through pinv(E_d). The latter is required because the
    paper uses E_d = dt E_c while delta-p is a continuous-channel quantity.
    """
    A_list_sim = np.asarray(A_list_sim)
    B_list_sim = np.asarray(B_list_sim)
    G_traj = np.asarray(G_traj)
    K_traj = np.asarray(K_traj)
    x_traj_sim = np.asarray(x_traj_sim)
    u_traj_sim = np.asarray(u_traj_sim)
    E_pinv = np.linalg.pinv(E_d)
    zero_wy = np.zeros(nw_y)
    gamma_traj = np.zeros(T - 1)

    for t in range(T - 1):
        rng = np.random.default_rng(98173 + t)
        eta = _unit_directions(rng, sample_count, nx) @ _psd_sqrt(Q_traj[t]).T
        err = _unit_directions(rng, sample_count, nx) @ _psd_sqrt(P_traj[t]).T
        w = _unit_directions(rng, sample_count, nw)

        scales = rng.uniform(0.15, 1.0, size=(sample_count, 1))
        eta *= scales
        err *= np.roll(scales, 17, axis=0)
        w *= np.roll(scales, 41, axis=0)

        x_nom_next = np.asarray(
            Integrator.STEP_NEW(
                dt, x_traj_sim[t], u_traj_sim[t], np.zeros(nw), zero_wy
            )[0]
        )

        u_closed = u_traj_sim[t] + (eta - err) @ K_traj[t].T
        x_closed = x_traj_sim[t] + eta
        closed_next, _ = jax.vmap(
            lambda x, u, wi: Integrator.STEP_NEW(dt, x, u, wi, zero_wy)
        )(x_closed, u_closed, w)
        closed_next = np.asarray(closed_next)

        A_cl = A_list_sim[t] + B_list_sim[t] @ K_traj[t]
        B_cl = -B_list_sim[t] @ K_traj[t]
        residual_ctrl = (
            closed_next
            - x_nom_next
            - eta @ A_cl.T
            - err @ B_cl.T
            - w @ G_traj[t].T
        )
        delta_p_ctrl = residual_ctrl @ E_pinv.T
        delta_q_ctrl = (
            eta @ (Cq + Dq @ K_traj[t]).T
            + err @ (-Dq @ K_traj[t]).T
            + w @ Gq.T
        )

        # The paper's observer IQC is local to the nominal trajectory and uses
        # Delta q = Cq e + Gq w.  Evaluate that exact channel here.  A fully
        # coupled off-nominal observer model would require an additional eta
        # channel in the paper's observer LMI.
        observer_next, _ = jax.vmap(
            lambda e, wi: Integrator.STEP_NEW(
                dt, x_traj_sim[t] + e, u_traj_sim[t], wi, zero_wy
            )
        )(err, w)
        observer_next = np.asarray(observer_next)
        residual_obs = (
            observer_next
            - x_nom_next
            - err @ A_list_sim[t].T
            - w @ G_traj[t].T
        )
        delta_p_obs = residual_obs @ E_pinv.T
        delta_q_obs = err @ Cq.T + w @ Gq.T

        sampled_bound = max(
            _max_ratio(delta_p_ctrl, delta_q_ctrl),
            _max_ratio(delta_p_obs, delta_q_obs),
        )
        gamma_traj[t] = max(1e-4, safety_factor * sampled_bound)

    return gamma_traj
