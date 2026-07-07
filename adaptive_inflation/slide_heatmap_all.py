"""
全アンサンブル数 m=2,3,8,20 のヒートマップを1枚にまとめる
"""
import numpy as np
import matplotlib.pyplot as plt
from function.create_data import createdata
from function.run import run_LETKF_fixed

N, F, dt = 40, 8.0, 0.05
R = np.eye(N)

SEARCH = {
    2:  {"inflation": [0.20, 0.40, 0.60, 0.80, 1.00],
         "sigma":     [1.0,  1.5,  2.0,  2.5,  3.0]},
    3:  {"inflation": [0.10, 0.20, 0.30, 0.40, 0.50],
         "sigma":     [1.0,  2.0,  3.0,  4.0,  5.0]},
    8:  {"inflation": [0.05, 0.08, 0.10, 0.15, 0.20],
         "sigma":     [3.0,  4.0,  6.0,  8.0,  10.0]},
    20: {"inflation": [0.00, 0.02, 0.05, 0.10, 0.15],
         "sigma":     [5.0,  7.0,  10.0, 12.0, 15.0]},
}
MEMBERS = [2, 3, 8, 20]

print("データ準備中...")
truth, obs, spinup_states = createdata(N, F, dt)
initial_background = obs[0].copy()

rmse_maps = {}
best_params = {}

for m in MEMBERS:
    inf_vals = SEARCH[m]["inflation"]
    sig_vals = SEARCH[m]["sigma"]
    rmse_map = np.zeros((len(inf_vals), len(sig_vals)))
    total = len(inf_vals) * len(sig_vals)
    count = 0
    for i, inf in enumerate(inf_vals):
        for j, sig in enumerate(sig_vals):
            count += 1
            print(f"  m={m} [{count}/{total}] δ={inf:.2f}, σ={sig:.1f} ...", end="\r")
            _, rmse_a, _, _ = run_LETKF_fixed(
                truth, obs, dt, F, R, inflation=inf, m=m, sigma=sig,
                spinup_states=spinup_states, initial_background=initial_background
            )
            rmse_map[i, j] = np.nanmean(rmse_a[-500:])
    rmse_maps[m] = rmse_map
    best_idx = np.unravel_index(np.nanargmin(rmse_map), rmse_map.shape)
    best_params[m] = {
        "inf": inf_vals[best_idx[0]],
        "sig": sig_vals[best_idx[1]],
        "rmse": rmse_map[best_idx],
    }
    print(f"\n  m={m}: best δ={best_params[m]['inf']:.2f}, σ={best_params[m]['sig']:.1f}, RMSE={best_params[m]['rmse']:.4f}")

print("\nヒートマップ描画中...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LETKF RMSE Heatmap (last 500 steps mean)", fontsize=14)

for ax, m in zip(axes.flat, MEMBERS):
    inf_vals = np.array(SEARCH[m]["inflation"])
    sig_vals = np.array(SEARCH[m]["sigma"])
    rmse_map = rmse_maps[m]

    vmax = min(np.nanmax(rmse_map), rmse_maps[m].mean() * 2.0)
    im = ax.imshow(rmse_map, cmap="viridis_r",
                   vmin=np.nanmin(rmse_map), vmax=vmax,
                   aspect="auto", origin="lower")

    ax.set_xticks(range(len(sig_vals)))
    ax.set_xticklabels([f"{s:.1f}" for s in sig_vals])
    ax.set_yticks(range(len(inf_vals)))
    ax.set_yticklabels([f"{d:.2f}" for d in inf_vals])
    ax.set_xlabel("Localization radius σ")
    ax.set_ylabel("Inflation factor δ")

    bp = best_params[m]
    ax.set_title(f"m={m}  (best: δ={bp['inf']:.2f}, σ={bp['sig']:.1f}, RMSE={bp['rmse']:.3f})",
                 fontsize=11)

    for i in range(len(inf_vals)):
        for j in range(len(sig_vals)):
            val = rmse_map[i, j]
            mid = (np.nanmax(rmse_map) + np.nanmin(rmse_map)) / 2
            color = "white" if val > mid else "black"
            text = f"{val:.3f}" if np.isfinite(val) else "DIV"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    # 最適セルを枠で強調
    best_idx = np.unravel_index(np.nanargmin(rmse_map), rmse_map.shape)
    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="red", lw=2.5
    ))

    plt.colorbar(im, ax=ax, label="RMSE")

plt.tight_layout()
plt.savefig("slide_heatmap_all.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n保存: slide_heatmap_all.png")
