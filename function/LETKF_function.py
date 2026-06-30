import numpy as np
from function.Lorenz96 import rk4

def LETKF_cycle(letkf_ensemble_a, y_obs, dt, F, R, inflation, localization_radius_sigma):
    """
    LETKF の 1ステップ分の予測・解析サイクル (返り値を完全に前と同様に統一した安全版)
    """
    m, N = letkf_ensemble_a.shape

    # --- 1. Forecast step (各メンバーの予測) ---
    letkf_ensemble_b = np.zeros_like(letkf_ensemble_a)
    for i in range(m):
        letkf_ensemble_b[i] = rk4(letkf_ensemble_a[i], dt, F)

    #メンバーの平均値を引き算することで、平均からのズレである予測摂動行列X^bを出す。
    letkf_xb_mean = np.mean(letkf_ensemble_b, axis=0)
    Xb = letkf_ensemble_b - letkf_xb_mean  # δX^b: shape (m, N)

    #メンバー減少による分散の過小評価を防ぐためにInflationを適応
    Xb = np.sqrt(1.0 + inflation) * Xb
    
    # 観測空間での予測摂動の箱の用意
    Yb = Xb.copy()

    # 解析値を格納する配列を用意する(Xa)
    letkf_ensemble_a_new = np.zeros_like(letkf_ensemble_b)

    # 全格子点について、周期境界を考慮した距離行列を事前計算する
    # ここだけ分けるのは、毎回この計算をするのは時間がかかるため
    all_distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j) #インデックスだけの引き算
            all_distances[i, j] = min(d, N - d) #二つある二点の距離の近い方を採用する

    # --- 2. Localized Analysis step (各格子点ごとに独立して計算) ---
    for i in range(N):
        distances = all_distances[i]  # 注目格子点 i から各観測点 j への距離
        cutoff = 3.6 * localization_radius_sigma #局所化半径の3.6倍の値：この値より小さいか大きいかで評価する

        weights = np.zeros(N)
        # 重み(weight)を計算している。cutoffの閾値で0とみなすかそのままの重みの値を使うかを評価している
        for j in range(N):
            if distances[j] < cutoff:
                weights[j] = np.exp(- (distances[j] ** 2) / (2.0 * (localization_radius_sigma ** 2)))
            else:
                weights[j] = 0.0

        # 念のため、もう一回極小な重みを排除する
        local_obs_indices = np.where(weights > 0.001)[0]

        # 重みが０のとき、予測値をそのまま解析値とする
        if len(local_obs_indices) == 0:
            letkf_ensemble_a_new[:, i] = letkf_ensemble_b[:, i]
            continue

        # 局所データをだす
        y_local = y_obs[local_obs_indices]
        yb_mean_local = letkf_xb_mean[local_obs_indices]
        Yb_local = Yb[:, local_obs_indices]  # shape (m, len(local_obs_indices))

        # スライド式: (R_loc)_ii <- R_ii * L(d)^-1  (重みで割り算して誤差を増大)
        R_diag_raw = np.diag(R)[local_obs_indices]
        R_local_diag = R_diag_raw / weights[local_obs_indices]
        
        # 【安全装置】観測を過信して誤差をゼロに潰さないための床
        R_local_diag = np.maximum(R_local_diag, R_diag_raw)
        R_local_inv = np.diag(1.0 / R_local_diag)

        # --- アンサンブル空間（m×m）でのカルマンフィルタ計算 ---
        I = np.eye(m)
        
        # 【数式修正】 (P~^a)^-1 = (m-1)*I + (Y^b) * R_loc^-1 * (Y^b)^T
        Pa_tilde_inv = (m - 1) * I + Yb_local @ R_local_inv @ Yb_local.T
        Pa_tilde = np.linalg.inv(Pa_tilde_inv)  # shape (m, m)

        # 平均更新用の重みベクトル wa_mean (shape: m,)
        innovation_local = y_local - yb_mean_local
        wa_mean = Pa_tilde @ Yb_local @ R_local_inv @ innovation_local

        # 摂動更新用の平方根行列 Wt_a (shape: m, m)
        evals, evecs = np.linalg.eigh(Pa_tilde)
        evals = np.maximum(evals, 1e-12)  # 負や極小の固有値をクリップして安定化
        Pa_tilde_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        Wt_a = np.sqrt(m - 1) * Pa_tilde_sqrt

        # --- 解析アンサンブルの再構成 ---
        W = np.zeros((m, m))
        for k in range(m):
            W[:, k] = wa_mean + Wt_a[:, k]

        # 格子点 i における全メンバーの解析値を一括更新
        letkf_ensemble_a_new[:, i] = letkf_xb_mean[i] + Xb[:, i] @ W

    # ご指摘の通りの変数名に揃えるための計算
    letkf_xa_mean = np.mean(letkf_ensemble_a_new, axis=0)
    letkf_spread_a = np.std(letkf_ensemble_a_new, axis=0, ddof=1)

    # 【数・順番をご指摘通りに修正完了】
    return (
        letkf_ensemble_a_new,  # letkf_ensemble_a (解析アンサンブル全体)
        letkf_ensemble_b,      # letkf_ensemble_b (予報アンサンブル全体)
        letkf_xb_mean,         # letkf_xb_mean     (予報平均)
        letkf_xa_mean,         # letkf_xa_mean     (解析平均)
        letkf_spread_a         # letkf_spread_a    (解析スプレッド)
    )