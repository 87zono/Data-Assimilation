import numpy as np
from function.Lorenz96 import rk4
from function.create_data import calculate_rmse

def cost_gradient(x, xb, B_inv, y_obs, H, R_inv):
    dx = x - xb
    dy = y_obs - np.dot(H, x)
    return np.dot(B_inv, dx) - np.dot(H.T, np.dot(R_inv, dy))

def threedvar_cycle_gradient_descent(xa, y_obs, dt, F, H, B_inv, R_inv):
    # 1. 予測（背景値 xb の計算）
    xb = rk4(xa, dt, F)

    # 2. 最急降下法によるコスト関数の最小化ループ
    x_iter = xb.copy()        # 探索の初期値は背景値 xb
    learning_rate = 0.01      # 更新のステップサイズ（学習率）
    max_iter = 100            # 最大反復回数
    tolerance = 1e-4          # 収束判定のしきい値

    for i in range(max_iter):
        grad = cost_gradient(x_iter, xb, B_inv, y_obs, H, R_inv)
        if np.linalg.norm(grad) < tolerance:
            break
        x_iter = x_iter - learning_rate * grad

    xa_new = x_iter
    return xa_new, xb

def run_3DVar(
    truth,
    obs,
    dt,
    F,
    H,
    B_inv,
    R_inv,
    initial_background=None,
    learning_rate=0.01,
    max_iter=100,
    tolerance=1e-4
):
    """
    3D-Varを全タイムステップで実行する。
    """

    if initial_background is None:
        threedvar_xa = obs[0].copy()
    else:
        threedvar_xa = initial_background.copy()

    threedvar_xb_save = []
    threedvar_xa_save = []

    threedvar_rmse_b_list = []
    threedvar_rmse_a_list = []

    threedvar_iteration_list = []
    threedvar_grad_norm_list = []

    for t in range(1, truth.shape[0]):

        (
            threedvar_xa,
            threedvar_xb,
            iteration_count,
            final_grad_norm
        ) = threedvar_cycle_gradient_descent(
            xa=threedvar_xa,
            y_obs=obs[t],
            dt=dt,
            F=F,
            H=H,
            B_inv=B_inv,
            R_inv=R_inv,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tolerance=tolerance
        )

        threedvar_rmse_b = calculate_rmse(
            threedvar_xb,
            truth[t]
        )

        threedvar_rmse_a = calculate_rmse(
            threedvar_xa,
            truth[t]
        )

        threedvar_xb_save.append(threedvar_xb.copy())
        threedvar_xa_save.append(threedvar_xa.copy())

        threedvar_rmse_b_list.append(threedvar_rmse_b)
        threedvar_rmse_a_list.append(threedvar_rmse_a)

        threedvar_iteration_list.append(iteration_count)
        threedvar_grad_norm_list.append(final_grad_norm)

    return (
        np.asarray(threedvar_xb_save),
        np.asarray(threedvar_xa_save),
        np.asarray(threedvar_rmse_b_list),
        np.asarray(threedvar_rmse_a_list),
        np.asarray(threedvar_iteration_list),
        np.asarray(threedvar_grad_norm_list)
    )