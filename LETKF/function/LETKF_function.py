import numpy as np
from function.Lorenz96 import rk4

def LETKF_cycle(letkf_ensemble_a, y_obs, dt, F, R, inflation, localization_radius_sigma):
    """
    スライドの数式を完全再現した LETKF の 1ステップ分の予測・解析サイクル
    """
    m, N = letkf_ensemble_a.shape

    # --- 1. Forecast step (各メンバーの予測) ---
    letkf_ensemble_b = np.zeros_like(letkf_ensemble_a)
    for i in range(m):
        letkf_ensemble_b[i] = rk4(letkf_ensemble_a[i], dt, F)

    # Forecast mean & perturbations
    letkf_xb_mean = np.mean(letkf_ensemble_b, axis=0)
    Xb = letkf_ensemble_b - letkf_xb_mean  # δX^b: shape (m, N)

    # Inflation (スライド式: δX_inf = (1 + δ) * δX^b)
    Xb = np.sqrt(1.0 + inflation) * Xb
    
    # 観測空間への射影 (H = I なので Y^b = Xb)
    Yb = Xb.copy()

    # 解析値を格納する配列
    letkf_ensemble_a_new = np.zeros_like(letkf_ensemble_b)

    # --- 2. Localized Analysis step (グリッドごとに局所化同化) ---
    for i in range(N):
        
        # 注目地点 i から各観測点 j への距離を計算（周期境界対応）
        distances = np.array([min(abs(i - j), N - abs(i - j)) for j in range(N)])
        
        # ガウシアン局所化の重み付け (スライドの L(d) の計算式)
        cutoff = 2.0 * np.sqrt(10.0 / 3.0) * localization_radius_sigma
        
        weights = np.zeros(N)
        for j in range(N):
            if distances[j] < cutoff:
                weights[j] = np.exp(- (distances[j] ** 2) / (2.0 * (localization_radius_sigma ** 2)))
            else:
                weights[j] = 0.0

        # ゴミのような極小の重みを排除して、局所観測のインデックスを取得 (安全装置)
        local_obs_indices = np.where(weights > 0.001)[0]

        # もし近所に観測がなければ、予測値をそのまま解析値とする
        if len(local_obs_indices) == 0:
            letkf_ensemble_a_new[:, i] = letkf_ensemble_b[:, i]
            continue

        # 局所空間の切り出し
        y_local = y_obs[local_obs_indices]
        yb_mean_local = letkf_xb_mean[local_obs_indices]
        Yb_local = Yb[:, local_obs_indices]  # shape (m, 局所観測数)

        # スライド式: (R_loc)_ii <- R_ii * L(d)^-1  (重みで割り算して誤差を増大)
        R_local_diag = np.diag(R)[local_obs_indices] / weights[local_obs_indices]
        R_local_inv = np.diag(1.0 / R_local_diag)

        # --- アンサンブル空間（m×m）でのカルマンフィルタ計算 ---
        I = np.eye(m)
        
        # スライド式: (P~^a)^-1 = I + (Y^b)^T * R_loc^-1 * Y^b
        Pa_tilde_inv = I + Yb_local @ R_local_inv @ Yb_local.T / (m - 1)
        Pa_tilde = np.linalg.inv(Pa_tilde_inv)  # shape (m, m)

        # スライド式: T の第一項（平均更新用のウェイト）
        # wa_mean = P~^a * (Y^b)^T * R_loc^-1 * d^o-b
        innovation_local = y_local - yb_mean_local
        wa_mean = Pa_tilde @ Yb_local @ R_local_inv @ innovation_local  # shape (m,)

        # スライド式: T の第二項（偏差更新用の平方根行列）
        evals, evecs = np.linalg.eigh(Pa_tilde)
        evals = np.maximum(evals, 0.0)  # 負の固有値をクリップ
        Pa_tilde_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        Wt_a = np.sqrt(m - 1) * Pa_tilde_sqrt  # shape (m, m)

        # ウェイト行列 T の完成
        T = wa_mean[:, np.newaxis] + Wt_a

        # スライド式: X^a = X^b + Z^b * T 
        # 地点 i の全メンバーを転置を噛み合わせて一撃で更新
        letkf_ensemble_a_new[:, i] = letkf_xb_mean[i] + T.T @ Xb[:, i]

    # --- 3. Final analysis (評価用の統計量計算) ---
    letkf_xa_mean = np.mean(letkf_ensemble_a_new, axis=0)
    letkf_Xa = letkf_ensemble_a_new - letkf_xa_mean
    letkf_Pa_approx = (letkf_Xa.T @ letkf_Xa) / (m - 1)
    letkf_spread_a = np.sqrt(np.trace(letkf_Pa_approx) / N)

    return (
        letkf_ensemble_a_new,
        letkf_ensemble_b,
        letkf_xb_mean,
        letkf_xa_mean,
        letkf_spread_a
    )