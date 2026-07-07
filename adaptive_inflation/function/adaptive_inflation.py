import numpy as np


def estimate_local_delta(d_local, Yb_raw_local, R_diag_local, R_diag_raw, delta_prior, v_b):
    """
    Miyoshi (2011) の式でインフレーション係数Δを局所的に推定する。

    Parameters
    ----------
    d_local       : (p_local,) イノベーション y_obs - H xb_mean（局所）
    Yb_raw_local  : (m, p_local) インフレ前の予報摂動（観測空間、局所）
    R_diag_local  : (p_local,) 観測誤差分散の対角成分（局所化済み: R/weight）
    R_diag_raw    : (p_local,) 観測誤差分散の対角成分（局所化前の原値）
    delta_prior   : スカラー、前ステップのΔ（事前値）
    v_b           : スカラー、Δの事前分散（チューニングパラメータ）

    Returns
    -------
    delta_new : スカラー、更新後のΔ（クリップ済み）
    """
    m, p_local = Yb_raw_local.shape
    if p_local == 0:
        return delta_prior

    # 局所化ありの場合、p は count ではなく sum(weights) = sum(R_raw/R_local)
    # E[tr(d dT ∘ R_local^{-1})] = Δ * tr(HBH^T ∘ R_local^{-1}) + sum(weights)
    p_eff = float(np.sum(R_diag_raw / R_diag_local))

    # HBH^T の対角成分を近似
    HBH_diag = np.sum(Yb_raw_local ** 2, axis=0) / (m - 1)

    # tr(HBH^T ∘ R_local^{-1}) と tr(d d^T ∘ R_local^{-1})
    tr_HBH_Rinv = np.sum(HBH_diag / R_diag_local)
    tr_dd_Rinv = np.sum(d_local ** 2 / R_diag_local)

    if tr_HBH_Rinv < 1e-10:
        return delta_prior

    # 観測からΔを推定（Miyoshi 2011 式、p → p_eff）
    delta_obs = (tr_dd_Rinv - p_eff) / tr_HBH_Rinv

    # 観測側の分散 v^o（Miyoshi 2011 式、p → p_eff）
    v_obs = (2.0 / p_eff) * ((delta_prior * tr_HBH_Rinv + p_eff) / tr_HBH_Rinv) ** 2

    # カルマン的加重平均でΔを更新
    delta_new = (delta_prior * v_obs + delta_obs * v_b) / (v_obs + v_b)

    return float(np.clip(delta_new, 1.0, 2.0))
