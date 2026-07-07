import numpy as np
from function.Lorenz96 import rk4
from function.adaptive_inflation import estimate_local_delta


def _build_distance_matrix(N):
    all_distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            all_distances[i, j] = min(d, N - d)
    return all_distances


def _local_weights(distances, localization_radius_sigma):
    cutoff = 3.6 * localization_radius_sigma
    weights = np.zeros(len(distances))
    for j in range(len(distances)):
        if distances[j] < cutoff:
            weights[j] = np.exp(-(distances[j] ** 2) / (2.0 * localization_radius_sigma ** 2))
    return weights


def _letkf_local_analysis(m, Yb_local, R_local_inv, innovation_local, Xb_inf_i, xb_mean_i):
    """共通のLETKFローカル解析（Yb_localはインフレ済みを渡す）"""
    I_m = np.eye(m)
    Pa_tilde_inv = (m - 1) * I_m + Yb_local @ R_local_inv @ Yb_local.T
    Pa_tilde = np.linalg.inv(Pa_tilde_inv)

    wa_mean = Pa_tilde @ Yb_local @ R_local_inv @ innovation_local

    evals, evecs = np.linalg.eigh(Pa_tilde)
    evals = np.maximum(evals, 1e-12)
    Pa_tilde_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    Wt_a = np.sqrt(m - 1) * Pa_tilde_sqrt

    W = np.zeros((m, m))
    for k in range(m):
        W[:, k] = wa_mean + Wt_a[:, k]

    return xb_mean_i + Xb_inf_i @ W


def LETKF_fixed_cycle(ensemble_a, y_obs, dt, F, R, inflation, localization_radius_sigma):
    """
    固定インフレーションのLETKF（既存コードと同等）。
    inflation : スカラー δ（Δ = 1 + δ として適用）
    """
    m, N = ensemble_a.shape

    ensemble_b = np.zeros_like(ensemble_a)
    for i in range(m):
        ensemble_b[i] = rk4(ensemble_a[i], dt, F)

    xb_mean = np.mean(ensemble_b, axis=0)
    Xb = ensemble_b - xb_mean  # (m, N)

    Xb_inf = np.sqrt(1.0 + inflation) * Xb
    Yb = Xb_inf.copy()

    ensemble_a_new = np.zeros_like(ensemble_b)
    all_distances = _build_distance_matrix(N)

    for i in range(N):
        weights = _local_weights(all_distances[i], localization_radius_sigma)
        local_obs_indices = np.where(weights > 0.001)[0]

        if len(local_obs_indices) == 0:
            ensemble_a_new[:, i] = ensemble_b[:, i]
            continue

        y_local = y_obs[local_obs_indices]
        yb_mean_local = xb_mean[local_obs_indices]
        Yb_local = Yb[:, local_obs_indices]
        innovation_local = y_local - yb_mean_local

        R_diag_raw = np.diag(R)[local_obs_indices]
        R_local_diag = np.maximum(R_diag_raw / weights[local_obs_indices], R_diag_raw)
        R_local_inv = np.diag(1.0 / R_local_diag)

        ensemble_a_new[:, i] = _letkf_local_analysis(
            m, Yb_local, R_local_inv, innovation_local, Xb_inf[:, i], xb_mean[i]
        )

    xa_mean = np.mean(ensemble_a_new, axis=0)
    spread_a = np.std(ensemble_a_new, axis=0, ddof=1)

    return ensemble_a_new, ensemble_b, xb_mean, xa_mean, spread_a


def LETKF_adaptive_cycle(ensemble_a, y_obs, dt, F, R, delta, v_b, localization_radius_sigma):
    """
    Miyoshi (2011) 動的インフレーションのLETKF。
    delta  : (N,) 各格子点のインフレーション係数Δ（前ステップの解析値）
    v_b    : スカラー、Δの事前分散
    戻り値に delta_new (N,) を追加。
    """
    m, N = ensemble_a.shape

    ensemble_b = np.zeros_like(ensemble_a)
    for i in range(m):
        ensemble_b[i] = rk4(ensemble_a[i], dt, F)

    xb_mean = np.mean(ensemble_b, axis=0)
    Xb = ensemble_b - xb_mean      # un-inflated (m, N)
    Yb_raw = Xb.copy()             # H=I なので obs空間 = 状態空間

    ensemble_a_new = np.zeros_like(ensemble_b)
    delta_new = np.ones(N)
    all_distances = _build_distance_matrix(N)

    for i in range(N):
        weights = _local_weights(all_distances[i], localization_radius_sigma)
        local_obs_indices = np.where(weights > 0.001)[0]

        if len(local_obs_indices) == 0:
            ensemble_a_new[:, i] = ensemble_b[:, i]
            delta_new[i] = delta[i]
            continue

        y_local = y_obs[local_obs_indices]
        yb_mean_local = xb_mean[local_obs_indices]
        Yb_raw_local = Yb_raw[:, local_obs_indices]  # (m, p_local) インフレ前
        innovation_local = y_local - yb_mean_local

        R_diag_raw = np.diag(R)[local_obs_indices]
        R_local_diag = np.maximum(R_diag_raw / weights[local_obs_indices], R_diag_raw)

        # --- Miyoshi (2011) でΔを推定・更新 ---
        delta_new[i] = estimate_local_delta(
            innovation_local, Yb_raw_local, R_local_diag, R_diag_raw, delta[i], v_b
        )

        # 局所的にインフレを適用
        sqrt_d = np.sqrt(delta_new[i])
        Yb_local = sqrt_d * Yb_raw_local
        Xb_inf_i = sqrt_d * Xb[:, i]

        R_local_inv = np.diag(1.0 / R_local_diag)

        ensemble_a_new[:, i] = _letkf_local_analysis(
            m, Yb_local, R_local_inv, innovation_local, Xb_inf_i, xb_mean[i]
        )

    xa_mean = np.mean(ensemble_a_new, axis=0)
    spread_a = np.std(ensemble_a_new, axis=0, ddof=1)

    return ensemble_a_new, ensemble_b, xb_mean, xa_mean, spread_a, delta_new
