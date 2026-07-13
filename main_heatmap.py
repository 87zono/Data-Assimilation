import numpy as np
from function.create_data import createdata
from function.run import run_Serial_EnSRF
from function.create_ensemble_data import create_localization_matrix 
from function.plot_heatmap import plot_rmse_heatmap_slide

# --- 基本設定 ---
N = 40
F = 8.0
dt = 0.05
H = np.eye(N)
R = np.eye(N)
m = 10 

inflation_values = np.array([0.00, 0.05, 0.10, 0.15, 0.20]) 
sigma_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

#データ作成
truth, obs, spinup_states = createdata(N, F, dt) # 前回の作成関数
initial_background = obs[0].copy()

rmse_map_m10_p40 = np.zeros((len(inflation_values), len(sigma_values)))

for i, inf in enumerate(inflation_values):
    for j, sig in enumerate(sigma_values):
        print(f"計算中... inflation={inf}, sigma={sig}")
        
        localization_matrix = create_localization_matrix(N, sig)
        
        # 2. シミュレーション実行
        _, _, _, rmse_a_list, _ = run_Serial_EnSRF(
            truth, obs, dt, F, H, R, inf, localization_matrix, m,
            spinup_states, initial_background
        )
        
        # 3. 安定した後半期間（または全期間）の「平均RMSE」を代表値としてマップに記録
        # ※フィルターが落ち着いた後半（例: 最後の500ステップなど）の平均をとるのが一般的です
        rmse_map_m10_p40[i, j] = np.mean(rmse_a_list[-500:])

levels_m10_p40 = [0.20, 0.21, 0.225, 0.25, 0.275, 0.30, 0.335, 1.0]

plot_rmse_heatmap_slide(
    rmse_map=rmse_map_m10_p40,
    inflation_values=inflation_values,
    sigma_values=sigma_values,
    title="Serial EnSRF RMSE: m=10, p=40",
    vmin=0.20,
    vmax=0.35,
    cmap_name="viridis",
    show_values=True
)