import numpy as np
from function.Lorenz96 import rk4


def LETKF_cycle_partial(
    letkf_ensemble_a,
    y_obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_radius_sigma
):
    """
    部分観測に対応したLETKFの1サイクル。
    """

    m, N = letkf_ensemble_a.shape

    # =====================================================
    # 1. Forecast
    # =====================================================

    letkf_ensemble_b = np.zeros_like(
        letkf_ensemble_a
    )

    for member in range(m):
        letkf_ensemble_b[member] = rk4(
            letkf_ensemble_a[member],
            dt,
            F
        )

    letkf_xb_mean = np.mean(
        letkf_ensemble_b,
        axis=0
    )

    Xb = (
        letkf_ensemble_b
        - letkf_xb_mean
    )

    # Inflation
    Xb = (
        np.sqrt(1.0 + inflation)
        * Xb
    )

    # Hの各行が観測している状態地点
    obs_indices = np.argmax(
        H,
        axis=1
    )

    # 観測空間での予報平均
    yb_mean_obs = letkf_xb_mean[
        obs_indices
    ]

    # 観測空間での予報摂動
    Yb = Xb[
        :,
        obs_indices
    ]

    letkf_ensemble_a_new = np.zeros_like(
        letkf_ensemble_b
    )

    # =====================================================
    # 2. Local analysis
    # =====================================================

    for state_index in range(N):

        # 注目地点と各観測地点との周期距離
        distances = np.array([
            min(
                abs(state_index - obs_index),
                N - abs(state_index - obs_index)
            )
            for obs_index in obs_indices
        ])

        cutoff = (
            3.6
            * localization_radius_sigma
        )

        weights = np.zeros(
            len(obs_indices)
        )

        for obs_position in range(
            len(obs_indices)
        ):
            if distances[obs_position] < cutoff:
                weights[obs_position] = np.exp(
                    -(
                        distances[obs_position] ** 2
                    )
                    / (
                        2.0
                        * localization_radius_sigma**2
                    )
                )

        local_positions = np.where(
            weights > 0.001
        )[0]

        # 局所観測がない場合
        if len(local_positions) == 0:
            letkf_ensemble_a_new[
                :,
                state_index
            ] = (
                letkf_xb_mean[state_index]
                + Xb[:, state_index]
            )
            continue

        # 局所観測データ
        y_local = y_obs[
            local_positions
        ]

        yb_mean_local = yb_mean_obs[
            local_positions
        ]

        Yb_local = Yb[
            :,
            local_positions
        ]

        R_diag_raw = np.diag(R)[
            local_positions
        ]

        R_local_diag = (
            R_diag_raw
            / weights[local_positions]
        )

        R_local_inv = np.diag(
            1.0 / R_local_diag
        )

        # =================================================
        # Ensemble-space analysis
        # =================================================

        I = np.eye(m)

        Pa_tilde_inv = (
            (m - 1) * I
            + Yb_local
            @ R_local_inv
            @ Yb_local.T
        )

        Pa_tilde = np.linalg.inv(
            Pa_tilde_inv
        )

        innovation_local = (
            y_local
            - yb_mean_local
        )

        wa_mean = (
            Pa_tilde
            @ Yb_local
            @ R_local_inv
            @ innovation_local
        )

        evals, evecs = np.linalg.eigh(
            Pa_tilde
        )

        evals = np.maximum(
            evals,
            1e-12
        )

        Pa_tilde_sqrt = (
            evecs
            @ np.diag(np.sqrt(evals))
            @ evecs.T
        )

        Wt_a = (
            np.sqrt(m - 1)
            * Pa_tilde_sqrt
        )

        W = (
            wa_mean[:, None]
            + Wt_a
        )

        letkf_ensemble_a_new[
            :,
            state_index
        ] = (
            letkf_xb_mean[state_index]
            + Xb[:, state_index] @ W
        )

    # =====================================================
    # 3. Final analysis
    # =====================================================

    letkf_xa_mean = np.mean(
        letkf_ensemble_a_new,
        axis=0
    )

    letkf_spread_by_state = np.std(
        letkf_ensemble_a_new,
        axis=0,
        ddof=1
    )

    return (
        letkf_ensemble_a_new,
        letkf_ensemble_b,
        letkf_xb_mean,
        letkf_xa_mean,
        letkf_spread_by_state
    )