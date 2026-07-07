"""
m=2 のヒートマップ: 最適固定インフレーションパラメータの探索
m が小さいほど大きな δ・小さな σ が必要なため、探索範囲を広めに設定
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed

N, F, dt = 40, 8.0, 0.05
m = 2
R = np.eye(N)

inflation_values = np.array([0.20, 0.40, 0.60, 0.80, 1.00])
sigma_values     = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

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
        text = f"{val:.3f}" if np.isfinite(val) else "DIV"
        ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, ax=ax, label="Mean Analysis RMSE")
plt.tight_layout()
plt.savefig("slide_m2_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

best_idx = np.unravel_index(np.nanargmin(rmse_map), rmse_map.shape)
print(f"\n【最適パラメータ (m={m})】")
print(f"  δ = {inflation_values[best_idx[0]]:.2f}")
print(f"  σ = {sigma_values[best_idx[1]]:.1f}")
print(f"  RMSE = {rmse_map[best_idx]:.4f}")
print(f"\n→ slide4_ensemble.py の OPTIMAL[2] をこの値に更新してください")
