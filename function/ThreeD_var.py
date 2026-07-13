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
