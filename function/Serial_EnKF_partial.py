import numpy as np
from function.Lorenz96 import rk4


def Serial_EnSRF_cycle_partial(
    ensrf_ensemble_a,
    y_obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_matrix
):
    """
    部分観測に対応したSerial EnSRFの1サイクル。
    """

    m, N = ensrf_ensemble_a.shape

    # =====================================================
    # 1. Forecast
    # =====================================================

    ensrf_ensemble_b = np.zeros_like(
        ensrf_ensemble_a
    )

    for member in range(m):
        ensrf_ensemble_b[member] = rk4(
            ensrf_ensemble_a[member],
            dt,
            F
        )

    ensrf_xb_mean = np.mean(
        ensrf_ensemble_b,
        axis=0
    )

    # =====================================================
    # 2. Forecast perturbations and inflation
    # =====================================================

    ensrf_X = (
        ensrf_ensemble_b
        - ensrf_xb_mean
    )

    ensrf_X = (
        np.sqrt(1.0 + inflation)
        * ensrf_X
    )

    ensrf_ensemble = (
        ensrf_xb_mean
        + ensrf_X
    )

    # Hの各行がどの状態変数を観測するか
    obs_state_indices = np.argmax(
        H,
        axis=1
    )

    num_obs = len(y_obs)

    # =====================================================
    # 3. Serial assimilation
    # =====================================================

    for obs_position in range(num_obs):

        state_index = obs_state_indices[
            obs_position
        ]

        # 現在の平均と摂動を再計算
        ensrf_x_mean = np.mean(
            ensrf_ensemble,
            axis=0
        )

        ensrf_X = (
            ensrf_ensemble
            - ensrf_x_mean
        )

        # この観測に対応する予報観測
        yb_mean = ensrf_x_mean[
            state_index
        ]

        y_pert = ensrf_X[
            :,
            state_index
        ]

        # 状態と観測の共分散
        Pxy = (
            ensrf_X.T @ y_pert
        ) / (m - 1)

        # 観測空間の予報分散
        Pyy = (
            y_pert @ y_pert
        ) / (m - 1)

        R_j = R[
            obs_position,
            obs_position
        ]

        K = Pxy / (Pyy + R_j)

        # 実際の観測地点に基づく局所化
        rho = localization_matrix[
            :,
            state_index
        ]

        K_loc = rho * K

        # Mean update
        innovation = (
            y_obs[obs_position]
            - yb_mean
        )

        ensrf_x_mean_new = (
            ensrf_x_mean
            + K_loc * innovation
        )

        # Perturbation update
        alpha = 1.0 / (
            1.0
            + np.sqrt(
                R_j / (Pyy + R_j)
            )
        )

        K_tilde_loc = alpha * K_loc

        ensrf_X_new = (
            ensrf_X
            - np.outer(
                y_pert,
                K_tilde_loc
            )
        )

        ensrf_ensemble = (
            ensrf_x_mean_new
            + ensrf_X_new
        )

    # =====================================================
    # 4. Final analysis
    # =====================================================

    ensrf_ensemble_a_new = ensrf_ensemble

    ensrf_xa_mean = np.mean(
        ensrf_ensemble_a_new,
        axis=0
    )

    ensrf_Xa = (
        ensrf_ensemble_a_new
        - ensrf_xa_mean
    )

    ensrf_Pa_approx = (
        ensrf_Xa.T @ ensrf_Xa
    ) / (m - 1)

    ensrf_spread_a = np.sqrt(
        np.trace(ensrf_Pa_approx) / N
    )

    return (
        ensrf_ensemble_a_new,
        ensrf_ensemble_b,
        ensrf_xb_mean,
        ensrf_xa_mean,
        ensrf_Pa_approx,
        ensrf_spread_a
    )