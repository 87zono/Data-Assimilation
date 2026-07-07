"""
スライド2: 実験設定の確認 + 前回ヒートマップ（最適インフレーションの確認）
  - 前回のLETKF設定と最適δをリマインド
  - 小さいグリッドでミニヒートマップを再描画
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed

# --- 実験設定 ---
N, F, dt = 40, 8.0, 0.05
m     = 20
R     = np.eye(N)

# ヒートマップ探索範囲（前回の最適付近: delta=0.02, sigma=10.0）
inflation_values = np.array([0.00, 0.02, 0.05, 0.10, 0.15])
sigma_values     = np.array([5.0,  7.0,  10.0, 12.0, 15.0])

print("データ準備中...")
truth, obs, spinup_states = createdata(N, F, dt)
initial_background = obs[0].copy()

rmse_map = np.zeros((len(inflation_values), len(sigma_values)))

total = len(inflation_values) * len(sigma_values)
count = 0
for i, inf in enumerate(inflation_values):
    for j, sig in enumerate(sigma_values):
        count += 1
        print(f"  [{count}/{total}] δ={inf:.2f}, σ={sig:.1f} ...", end="\r")
        _, rmse_a, _, _ = run_LETKF_fixed(
            truth, obs, dt, F, R, inflation=inf, m=m, sigma=sig,
            spinup_states=spinup_states, initial_background=initial_background
        )
        rmse_map[i, j] = np.nanmean(rmse_a[-500:])

print("\n\nヒートマップ描画中...")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    rmse_map, cmap="viridis_r",
    vmin=np.nanmin(rmse_map), vmax=min(np.nanmax(rmse_map), 2.0),
    aspect="auto", origin="lower"
)

ax.set_xticks(range(len(sigma_values)))
ax.set_xticklabels([f"{s:.1f}" for s in sigma_values])
ax.set_yticks(range(len(inflation_values)))
ax.set_yticklabels([f"{d:.2f}" for d in inflation_values])
ax.set_xlabel("Localization radius σ")
ax.set_ylabel("Inflation factor δ")
ax.set_title(f"LETKF RMSE Heatmap (m={m}, last 500 steps mean)")

for i in range(len(inflation_values)):
    for j in range(len(sigma_values)):
        val = rmse_map[i, j]
        color = "white" if val > (np.nanmax(rmse_map) + np.nanmin(rmse_map)) / 2 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax, label="Mean Analysis RMSE")
plt.tight_layout()
plt.savefig("slide2_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# 最適値の確認
best_idx = np.unravel_index(np.nanargmin(rmse_map), rmse_map.shape)
print(f"\n【最適パラメータ】")
print(f"  δ = {inflation_values[best_idx[0]]:.2f}")
print(f"  σ = {sigma_values[best_idx[1]]:.1f}")
print(f"  RMSE = {rmse_map[best_idx]:.4f}")
