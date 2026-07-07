import numpy as np
from function.LETKF_adaptive import LETKF_fixed_cycle, LETKF_adaptive_cycle
from function.Serial_EnKF_function import make_initial_ensemble_from_spinup

_DIVERGE_THRESHOLD = 100.0


def run_LETKF_fixed(truth, obs, dt, F, R, inflation, m, sigma,
                    spinup_states, initial_background):
    """
    固定インフレーションのLETKFを全タイムステップ実行する。
    inflation : スカラー δ（Δ = 1 + δ）

    Returns
    -------
    rmse_b, rmse_a, spread_a, delta_hist : 各 (T,) 配列
    delta_hist は固定値 1+inflation の定数配列。
    """
    N = truth.shape[1]
    ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states, initial_background, m, seed=42
    )

    rmse_b_list, rmse_a_list, spread_list, delta_list = [], [], [], []
    T = truth.shape[0]

    for t in range(1, T):
        (ensemble_a, ensemble_b, xb_mean, xa_mean, spread_a) = LETKF_fixed_cycle(
            ensemble_a, obs[t], dt, F, R, inflation, sigma
        )

        rmse_b = np.sqrt(np.mean((xb_mean - truth[t]) ** 2))
        rmse_a = np.sqrt(np.mean((xa_mean - truth[t]) ** 2))

        rmse_b_list.append(rmse_b)
        rmse_a_list.append(rmse_a)
        spread_list.append(float(np.mean(spread_a)))
        delta_list.append(1.0 + inflation)

        if rmse_a > _DIVERGE_THRESHOLD or np.isnan(rmse_a):
            n_fill = T - 1 - t
            rmse_b_list += [np.nan] * n_fill
            rmse_a_list += [np.nan] * n_fill
            spread_list += [np.nan] * n_fill
            delta_list += [1.0 + inflation] * n_fill
            break

    return (
        np.array(rmse_b_list),
        np.array(rmse_a_list),
        np.array(spread_list),
        np.array(delta_list),
    )


def run_LETKF_adaptive(truth, obs, dt, F, R, v_b, m, sigma,
                       spinup_states, initial_background, delta_init=1.0):
    """
    Miyoshi (2011) 動的インフレーションのLETKFを全タイムステップ実行する。
    v_b        : Δの事前分散
    delta_init : Δの初期値（デフォルト1.0 = インフレーションなし）

    Returns
    -------
    rmse_b, rmse_a, spread_a, delta_hist : 各 (T,) 配列
    delta_hist は各ステップの格子点平均Δ。
    """
    N = truth.shape[1]
    ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states, initial_background, m, seed=42
    )

    delta = np.ones(N) * delta_init

    rmse_b_list, rmse_a_list, spread_list, delta_list = [], [], [], []
    T = truth.shape[0]

    for t in range(1, T):
        (ensemble_a, ensemble_b, xb_mean, xa_mean, spread_a, delta) = LETKF_adaptive_cycle(
            ensemble_a, obs[t], dt, F, R, delta, v_b, sigma
        )

        rmse_b = np.sqrt(np.mean((xb_mean - truth[t]) ** 2))
        rmse_a = np.sqrt(np.mean((xa_mean - truth[t]) ** 2))

        rmse_b_list.append(rmse_b)
        rmse_a_list.append(rmse_a)
        spread_list.append(float(np.mean(spread_a)))
        delta_list.append(float(np.mean(delta)))

        if rmse_a > _DIVERGE_THRESHOLD or np.isnan(rmse_a):
            n_fill = T - 1 - t
            rmse_b_list += [np.nan] * n_fill
            rmse_a_list += [np.nan] * n_fill
            spread_list += [np.nan] * n_fill
            delta_list += [np.nan] * n_fill
            break

    return (
        np.array(rmse_b_list),
        np.array(rmse_a_list),
        np.array(spread_list),
        np.array(delta_list),
    )
