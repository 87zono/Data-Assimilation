import numpy as np
from function.Lorenz96 import rk4

def PO_EnKF_cycle(po_ensemble_a, y_obs, dt, F, H, R, inflation, localization_matrix=None):
    """
    観測摂動（PO法）アンサンブルカルマンフィルタの1サイクル
    ※ localization_matrix を渡せば局所化あり、Noneなら局所化なし(通常のPO)として動きます。
    """
    m, N = po_ensemble_a.shape
    num_obs = len(y_obs)

    # 1. 予報ステップ
    po_ensemble_b = np.zeros_like(po_ensemble_a)
    for i in range(m):
        po_ensemble_b[i] = rk4(po_ensemble_a[i], dt, F)

    xb_mean = np.mean(po_ensemble_b, axis=0)
    Xb = po_ensemble_b - xb_mean
    # インフレーション
    Xb = np.sqrt(1.0 + inflation) * Xb
    po_ensemble = xb_mean + Xb

    # 2. 観測摂動の生成 (R の次元に合わせる)
    rng = np.random.default_rng()
    # 観測誤差共分散 R から対角成分の標準偏差を取り出す
    R_diag = np.diag(R)
    # 各メンバー用に平均0、標準偏差 sqrt(R) のノイズを生成
    perturbed_obs = np.zeros((m, num_obs))
    for i in range(m):
        perturbed_obs[i] = y_obs + rng.normal(0.0, np.sqrt(R_diag))

    # 3. 解析（アップデート）ステップ
    # 現在のアンサンブルの平均と摂動を再取得
    xb_mean = np.mean(po_ensemble, axis=0)
    Xb = po_ensemble - xb_mean

    # 観測空間への射影
    # H 行列は（観測の偏りに応じて変形している可能性を考慮し）動的に掛け算
    Yb = Xb @ H.T  # shape: (m, num_obs)
    
    # 予測された観測の共分散 H*Pf*H^T
    HPfHT = (Yb.T @ Yb) / (m - 1)
    # 予測共分散 Pf*H^T
    PfHT = (Xb.T @ Yb) / (m - 1)

    # 局所化の適用 (行列サイズを num_obs x num_obs に合わせる)
    if localization_matrix is not None:
        # H がどの観測地点を選んでいるかに合わせて局所化行列をスライス
        # ※全変数観測ベースの偏りなら、観測インデックスに合わせてスライス
        obs_indices = [np.where(H[r] == 1)[0][0] for r in range(num_obs)]
        loc_sub = localization_matrix[np.ix_(obs_indices, obs_indices)]
        HPfHT = loc_sub * HPfHT
        
        # PfHT は N x num_obs 次元
        loc_pf = localization_matrix[:, obs_indices]
        PfHT = loc_pf * PfHT

    # カルマンゲイン K の計算
    # K = Pf*H^T * (H*Pf*H^T + R)^-1
    try:
        K = PfHT @ np.linalg.inv(HPfHT + R)
    except np.linalg.LinAlgError:
        # 万が一、逆行列が解けない場合は更新しない（安全装置）
        return po_ensemble, xb_mean

    # 各メンバーの更新
    po_ensemble_a_new = np.zeros_like(po_ensemble)
    for i in range(m):
        # イノベーション (摂動を加えた観測値 - モデルの予測観測値)
        innovation = perturbed_obs[i] - (H @ po_ensemble[i])
        po_ensemble_a_new[i] = po_ensemble[i] + K @ innovation

    xa_mean = np.mean(po_ensemble_a_new, axis=0)
    
    return po_ensemble_a_new, xa_mean