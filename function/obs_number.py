import numpy as np

def create_observation_setup(N, num_obs, mode):
    """
    部分観測用の観測地点、H、Rを作成する。

    Parameters
    ----------
    N : int
        状態変数数
    num_obs : int
        観測地点数
    mode : str
        "homogeneous" または "dense"

    Returns
    -------
    obs_indices : ndarray, shape (num_obs,)
        観測する状態変数の番号
    H : ndarray, shape (num_obs, N)
        観測演算子
    R : ndarray, shape (num_obs, num_obs)
        観測誤差共分散
    """

    if not 1 <= num_obs <= N:
        raise ValueError("num_obs must satisfy 1 <= num_obs <= N")

    if mode == "homogeneous":
        # 全40地点にできるだけ均等に配置
        obs_indices = np.floor(
            np.arange(num_obs) * N / num_obs
        ).astype(int)

    elif mode == "dense":
        # 0番地点から連続して密集配置
        obs_indices = np.arange(num_obs)

    else:
        raise ValueError(
            "mode must be 'homogeneous' or 'dense'"
        )

    H = np.zeros((num_obs, N))
    H[np.arange(num_obs), obs_indices] = 1.0

    # 観測誤差分散1
    R = np.eye(num_obs)

    return obs_indices, H, R