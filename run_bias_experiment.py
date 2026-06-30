# run_bias_experiment.py
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run_bias import run_bias_experiment_all_methods

# --- 1. 基本設定 ---
N = 40
F = 8.0
dt = 0.05
m = 20
inflation = 0.02  # 共通パラメータ 8%
sigma = 10.0

print("--- データの作成中 ---")
truth, obs_full, _ = createdata(N, F, dt)

# 200ステップの平均
test_steps = 365*4*2
truth_sub = truth[:test_steps]
obs_sub = obs_full[:test_steps]

# 横軸となる観測地点数のリスト
obs_counts = [20, 25, 30, 35, 40]

# ★ Serial_EnSRF をリストから除外しました
methods_list = ["EKF", "LETKF"]
results_uniform = {m: [] for m in methods_list}
results_biased = {m: [] for m in methods_list}

# --- 2. 実験ループ ---
for count in obs_counts:
    print(f"\n--- 観測地点数: {count} の実験を実行中 ---")
    
    # パターンA: 均等配置 (Uniform)
    uniform_indices = [int(i) for i in np.linspace(0, N-1, count, endpoint=False)]
    
    # パターンB: 偏り配置 (Biased)
    biased_indices = list(range(0, count))
    
    # 全手法走らせる関数はそのまま利用し、必要なデータだけを抽出します
    res_u = run_bias_experiment_all_methods(truth_sub, obs_sub, dt, F, m, inflation, sigma, uniform_indices)
    for method in methods_list:
        results_uniform[method].append(np.mean(res_u[method]))
        
    res_b = run_bias_experiment_all_methods(truth_sub, obs_sub, dt, F, m, inflation, sigma, biased_indices)
    for method in methods_list:
        results_biased[method].append(np.mean(res_b[method]))

# --- 3. グラフ描画 (左右に並べる) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ★ カラーとマーカーの設定から Serial を除外
colors = {"EKF": "blue", "LETKF": "green", "PO_EnKF": "black"}
markers = {"EKF": "o", "LETKF": "^", "PO_EnKF": "x"}

# 左側：均等配置 (Uniform)
for method in methods_list:
    ax1.plot(obs_counts, results_uniform[method], label=method, 
             color=colors[method], marker=markers[method], linewidth=2.0)
ax1.set_xlabel("Number of Observation Stations")
ax1.set_ylabel("Mean Analysis RMSE")
ax1.set_title("A. Uniform Observation Placement")
ax1.set_xticks(obs_counts) 
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.legend()

# 右側：偏り配置 (Biased)
for method in methods_list:
    ax2.plot(obs_counts, results_biased[method], label=method, 
             color=colors[method], marker=markers[method], linewidth=2.0)
ax2.set_xlabel("Number of Observation Stations")
ax2.set_title("B. Biased Observation Placement (Data Void)")
ax2.set_xticks(obs_counts) 
ax2.grid(True, linestyle="--", alpha=0.7)
ax2.legend()

plt.suptitle("Data Assimilation Accuracy vs. Number of Observations", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("observation_density_comparison2.png", dpi=300)
plt.show()

print("\n--- 実験完了！Serialなしの比較折れ線グラフを保存しました ---")