import matplotlib.pyplot as plt
import numpy as np

def plot_rmse_heatmap_slide(
    rmse_map,
    inflation_values,
    sigma_values,
    title,
    vmin=None,
    vmax=None,
    cmap_name="viridis",
    show_values=True
):
    rmse_plot = rmse_map.copy()

    if vmin is None:
        vmin = np.nanmin(rmse_plot)

    if vmax is None:
        vmax = np.nanmax(rmse_plot)

    # vmax より大きい値は vmax に丸める
    # 真っ赤な発散領域を大きく見せすぎないため
    rmse_clipped = np.clip(rmse_plot, vmin, vmax)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap = plt.colormaps[cmap_name]

    x_edges = np.arange(len(sigma_values) + 1)
    y_edges = np.arange(len(inflation_values) + 1)

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        rmse_clipped,
        cmap=cmap,
        shading="flat",
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Mean Analysis RMSE", fontsize=12)

    ax.set_xticks(np.arange(len(sigma_values)) + 0.5)
    ax.set_xticklabels(sigma_values, fontsize=11)

    ax.set_yticks(np.arange(len(inflation_values)) + 0.5)
    ax.set_yticklabels([f"{v:.2f}" for v in inflation_values], fontsize=11)

    ax.set_xlabel("Localization length scale σ", fontsize=13)
    ax.set_ylabel("Inflation factor δ", fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)

    # 白い罫線でセルを見やすくする
    ax.set_xticks(np.arange(len(sigma_values) + 1), minor=True)
    ax.set_yticks(np.arange(len(inflation_values) + 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 各セルに数値を書く
    if show_values:
        threshold = (vmin + vmax) / 2

        for i in range(len(inflation_values)):
            for j in range(len(sigma_values)):
                value = rmse_map[i, j]

                text_color = "white" if rmse_clipped[i, j] < threshold else "black"

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color
                )

    # best point
    best_i, best_j = np.unravel_index(
        np.nanargmin(rmse_map),
        rmse_map.shape
    )

    ax.scatter(
        best_j + 0.5,
        best_i + 0.5,
        marker="*",
        s=260,
        facecolor="red",
        edgecolor="black",
        linewidth=1.0,
        label="Best"
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    print("Best setting")
    print("delta =", inflation_values[best_i])
    print("sigma =", sigma_values[best_j])
    print("RMSE =", rmse_map[best_i, best_j])