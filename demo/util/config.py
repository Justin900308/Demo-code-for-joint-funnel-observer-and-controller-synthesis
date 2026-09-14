import numpy as np

MODES = {
    "double_integrator": dict(
        T=60, Tf=4, nx=4, nu=2, ny=2, nw=2, nwy=2, x_0=np.array([0, 0, 0, 0]), x_des=np.array([9.5, 4.5, 0, 0]),
        N=11,
        C=np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]]),
        D=np.array([[0.5, 0],
                    [0, 0.1]]),
        G=np.array([[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]]),
        num_obs=2,
        obs=np.array([[4, 3], [9, 3]]) * 1,
        obs_r=1,

    ),
    "unicycle": dict(
        T=30, Tf=6, nx=3, nu=2, ny=2, nw=2, nwy=2, nq=2,
        x_0=np.array([0.0, 0.0, 0.0]),
        x_des=np.array([9.5, 4.5, 0.0]),
        N=11, alpha=0.99, beta=0.99, tau_x=0.05, tau_y=0.05,
        vx=0.1, vy=0.1,
        trust_Q=0.20, trust_P=0.20, trust_K=0.75, trust_L=0.75,
        trust_sigma=0.30, trust_x=0.60, trust_u=1.50,
        funnel_step=0.5,
        # The paper's Cq/Dq/E factorization is exact for forward Euler.
        # RK4 is retained in Integrator.py for comparison, but using it here
        # requires a stage-augmented nonlinear channel.
        integrator="euler",
        ## system channels
        C=np.array([[1, 0, 0],
                    [0, 1, 0]]),
        D=np.array([[0.03, 0],
                    [0, 0.01]]),
        G=np.array([[0.1, 0.0],
                    [0.0, 0.5],
                    [0.0, 0.0]]),
        Cq=np.array([[0, 0, 1],
                     [0, 0, 0]]),
        Dq=np.array([[0, 0],
                     [1, 0]]),
        E=np.array([[1, 0],
                     [0, 1],
                     [0, 0]]),
        Gq=np.array([[0, 0],
                     [0, 0]]),
        num_obs=2,
        obs=np.array([[4, 3], [8, 3]]) * 1,
        obs_r=1,

    ),
}

current_mode = MODES["unicycle"]
