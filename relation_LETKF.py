import numpy as np
import matplotlib.pyplot as plt

# 各ファイルから作成した関数をインポート
from function.create_data import createdata
from function.run import run_EKF, run_LETKF

# --- 1. 実験設定パラメータ ---
N = 40       # 変数の数
F = 8.0      # 強制項
dt = 0.05    # タイムステップ (6時間相当)
inflation = 0.1  # インフレーション係数 (10%)

# LETKF用のパラメータ
m = 20                        # アンサンブルメンバー数
localization_radius_sigma = 2.0  # ローカライゼーション半径 (σ)

# 観測とモデルの設定
H = np.eye(N)  # 全変数観測
R = np.eye(N)  # 観測ノイズの分散 = 1.0

# --- 2. データの作成と準備 ---
steps_per_year = 365 * 4
total_steps = steps_per_year * 2

x = F * np.ones(N)
x[0] += 0.01  # 微小な摂動

all_data = np.zeros((total_steps + 1, N))
all_data[0] = x

# 2年分のシミュレーションを回す
from function.Lorenz96 import rk4
for t in range(total_steps):
    all_data[t+1] = rk4(all_data[t], dt, F)

# 切り分ける
spinup_states = all_data[:steps_per_year + 1]  # 最初の1年分
truth = all_data[steps_per_year + 1:]          # 後半の1年分（真値）

# 観測データの作成（真値にノイズを乗せる）
rng = np.random.Generator(np.random.MT19937(seed=42))
noise = rng.normal(loc=0.0, scale=1.0, size=truth.shape)
obs = truth + noise

# 同化開始時点の初期予測値
initial_background = obs[0].copy()

print("--- データ準備完了。データ同化を開始します ---")

# --- 3. 各データ同化手法の実行 ---

# ① EKF (拡張カルマンフィルタ)
print("1/2: 拡張カルマンフィルタ (EKF) を実行中...")
_, _, _, ekf_rmse_a, _ = run_EKF(
    truth, obs, dt, F, H, R, inflation
)

# ② LETKF (局所アンサンブル変分カルマンフィルタ)
print("2/2: LETKF を実行中...")
_, _, _, letkf_rmse_a, _ = run_LETKF(
    truth, obs, dt, F, H, R, inflation, m, localization_radius_sigma,
    spinup_states, initial_background
)

print("--- すべての同化計算が完了しました。グラフを描画します ---")

# --- 4. グラフのプロット ---
num_data_points = len(ekf_rmse_a)
time_days = np.arange(1, num_data_points + 1) / 4.0  # 1ステップ = 0.25日

plt.figure(figsize=(10, 4.5))

# EKF
plt.plot(time_days, ekf_rmse_a, label=f"EKF (δ={inflation})", color="blue", alpha=0.7)

# LETKF
plt.plot(time_days, letkf_rmse_a, label=f"LETKF (m={m}, σ={localization_radius_sigma})", color="green", alpha=0.8)

plt.xlabel("Time [days]")
plt.ylabel("Analysis RMSE")
plt.title("Data Assimilation Performance Comparison: EKF vs LETKF")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")

# 各手法の全期間平均RMSEをターミナルに表示
print(f"【平均解析RMSE】")
print(f"  EKF  : {np.mean(ekf_rmse_a):.4f}")
print(f"  LETKF: {np.mean(letkf_rmse_a):.4f}")

plt.show()