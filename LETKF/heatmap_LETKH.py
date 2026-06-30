import numpy as np
from function.create_data import createdata
from function.run import run_LETKF
from function.plot_heatmap import plot_rmse_heatmap_slide

# --- 1. 基本実験設定 ---
N = 40       # 変数の数
F = 8.0      # 強制項
dt = 0.05    # タイムステップ
m = 10       # アンサンブルメンバー数 (m=10)
H = np.eye(N)
R = np.eye(N)

# --- 2. 総当たりするパラメータのリストを設定 ---
inflation_values = np.array([0.12, 0.13, 0.14, 0.15, 0.16, 0.17])
sigma_values = np.array([1.5, 1.7, 1.9, 2.1, 2.3, 2.5])

# --- 3. データの作成と準備 (共通データを1回だけ作成) ---
truth, obs, spinup_states = createdata(N, F, dt)

# 同化開始時点の初期予測値（ここは今まで通り）
initial_background = obs[0].copy()

# --- 4. 結果を格納する2次元の箱（グリッド）を用意 ---
rmse_map_m10_p40 = np.zeros((len(inflation_values), len(sigma_values)))

print("--- LETKF パラメータ総当たり計算を開始します (m=10) ---")

# --- 5. 2重ループで総当たり計算 ---
for i, inf in enumerate(inflation_values):
    for j, sig in enumerate(sigma_values):
        print(f"計算中... [Inflation (δ): {inf:.4f} | Sigma (σ): {sig:.4f}]")
        
        # LETKFを実行
        _, _, _, letkf_rmse_a, _ = run_LETKF(
            truth, obs, dt, F, H, R, inf, m, sig,
            spinup_states, initial_background
        )
        
        rmse_map_m10_p40[i, j] = np.nanmean(letkf_rmse_a[-500:])

print("--- すべての計算が完了しました。ヒートマップを描画します ---")

# --- 6. お持ちの関数を呼び出して描画を実行 ---
plot_rmse_heatmap_slide(
    rmse_map=rmse_map_m10_p40,
    inflation_values=inflation_values,
    sigma_values=sigma_values,
    title="Local Ensemble Transform Kalman Filter (LETKF) RMSE: m=10, p=40",
    vmin=0.20,
    vmax=0.35,
    cmap_name="viridis",
    show_values=True
)