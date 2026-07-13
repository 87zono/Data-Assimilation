import numpy as np
import matplotlib.pyplot as plt

from function.create_data import createdata
from function.run import run_EKF, run_Serial_EnSRF
from function.create_ensemble_data import create_localization_matrix

# --- 設定パラメータ ---
N = 40
F = 8.0
H = np.eye(N)
R = np.eye(N)
dt = 0.05
inflation = 0.1
m = 20
localization_radius_sigma = 2.0

#データ作成
truth, obs, spinup_states = createdata(N, F, dt)
initial_background = obs[0].copy()

#データ同化
# EKFを走らせる
xb_hist, xa_hist, rmse_b, rmse_a, spread_a = run_EKF(
    truth, obs, dt, F, H, R, inflation
)
# Serial_EnSRFを走らせる
localization_matrix = create_localization_matrix(N, localization_radius_sigma)

ensrf_xb_mean, ensrf_xa_mean, ensrf_rmse_b, ensrf_rmse_a, ensrf_spread_a = run_Serial_EnSRF(
    truth, obs, dt, F, H, R, inflation, localization_matrix, m,
    spinup_states, initial_background
)

#ここから結果をplot
# データのステップ数を取得（最初の1ステップ目を引いているので、len(rmse_a) でOK）
num_data_points = len(rmse_a)

# 時間軸（日換算）の作成：1ステップは dt=0.05（6時間分 ＝ 1/4日）
time_days = np.arange(1, num_data_points + 1) / 4.0

plt.figure(figsize=(10, 4))

# EKF の解析RMSEをプロット
plt.plot(
    time_days,
    rmse_a,               # run_EKF から返ってきた解析RMSE
    label=f"EKF, δ={inflation}",
    color="blue"
)

# Serial EnSRF の解析RMSEをプロット
plt.plot(
    time_days,
    ensrf_rmse_a,         # run_Serial_EnSRF から返ってきた解析RMSE
    label=f"Serial EnSRF, m={m}, δ={inflation}, σ={localization_radius_sigma}",
    color="orange"
)

plt.xlabel("Time [days]")
plt.ylabel("Analysis RMSE")
plt.title("Analysis RMSE Comparison: EKF vs Serial EnSRF")
plt.grid(True)
plt.legend()
plt.show()