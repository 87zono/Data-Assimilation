import time
import numpy as np
import matplotlib.pyplot as plt

from function.create_data import createdata
from function.create_ensemble_data import create_localization_matrix

from function.compare_all_normal import (
    run_PO_EnKF,
    run_Serial_EnSRF,
    run_LETKF,
)

# ============================================================
# 共通設定
# ============================================================

N = 40
F = 8.0
dt = 0.05

H = np.eye(N)
R = np.eye(N)

burn_in = 100

truth, obs, spinup_states = createdata(
    N=N,
    F=F,
    dt=dt
)

initial_background = obs[0].copy()

methods = [
    "PO-EnKF",
    "Serial EnSRF",
    "LETKF"
]


# ============================================================
# 1条件で3手法を実行
# ============================================================

def run_ensemble_methods(
    m,
    inflation,
    localization_radius
):
    localization_matrix = create_localization_matrix(
        N=N,
        localization_radius=localization_radius
    )

    results = {}

    # --------------------------------------------------------
    # PO-EnKF
    # --------------------------------------------------------

    start_time = time.perf_counter()

    po_result = run_PO_EnKF(
        truth=truth,
        obs=obs,
        dt=dt,
        F=F,
        H=H,
        R=R,
        inflation=inflation,
        localization_matrix=localization_matrix,
        m=m,
        spinup_states=spinup_states,
        initial_background=initial_background
    )

    po_time = time.perf_counter() - start_time

    results["PO-EnKF"] = {
        "mean_rmse": np.mean(
            po_result["rmse_a"][burn_in:]
        ),
        "mean_spread": np.mean(
            po_result["spread_a"][burn_in:]
        ),
        "runtime": po_time
    }

    # --------------------------------------------------------
    # Serial EnSRF
    # --------------------------------------------------------

    start_time = time.perf_counter()

    ensrf_result = run_Serial_EnSRF(
        truth=truth,
        obs=obs,
        dt=dt,
        F=F,
        H=H,
        R=R,
        inflation=inflation,
        localization_matrix=localization_matrix,
        m=m,
        spinup_states=spinup_states,
        initial_background=initial_background
    )

    ensrf_time = time.perf_counter() - start_time

    results["Serial EnSRF"] = {
        "mean_rmse": np.mean(
            ensrf_result["rmse_a"][burn_in:]
        ),
        "mean_spread": np.mean(
            ensrf_result["spread_a"][burn_in:]
        ),
        "runtime": ensrf_time
    }

    # --------------------------------------------------------
    # LETKF
    # --------------------------------------------------------

    start_time = time.perf_counter()

    letkf_result = run_LETKF(
        truth=truth,
        obs=obs,
        dt=dt,
        F=F,
        R=R,
        inflation=inflation,
        localization_radius_sigma=localization_radius,
        m=m,
        spinup_states=spinup_states,
        initial_background=initial_background
    )

    letkf_time = time.perf_counter() - start_time

    results["LETKF"] = {
        "mean_rmse": np.mean(
            letkf_result["rmse_a"][burn_in:]
        ),
        "mean_spread": np.mean(
            letkf_result["spread_a"][burn_in:]
        ),
        "runtime": letkf_time
    }

    return results


# ============================================================
# m と inflation の設定
# ============================================================

heatmap_m_values = [
    5,
    10,
    20,
    40
]

heatmap_inflation_values = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10
]

# sigmaは固定
fixed_sigma = 5.0


# ============================================================
# 結果保存用
# ============================================================

rmse_heatmap_results = {
    method: np.zeros(
        (
            len(heatmap_m_values),
            len(heatmap_inflation_values)
        )
    )
    for method in methods
}

spread_heatmap_results = {
    method: np.zeros(
        (
            len(heatmap_m_values),
            len(heatmap_inflation_values)
        )
    )
    for method in methods
}

runtime_heatmap_results = {
    method: np.zeros(
        (
            len(heatmap_m_values),
            len(heatmap_inflation_values)
        )
    )
    for method in methods
}


# ============================================================
# 実験
# ============================================================

for m_index, m_value in enumerate(
    heatmap_m_values
):

    for inflation_index, inflation_value in enumerate(
        heatmap_inflation_values
    ):

        print(
            f"\nRunning m={m_value}, "
            f"inflation={inflation_value:.2f}"
        )

        current_results = run_ensemble_methods(
            m=m_value,
            inflation=inflation_value,
            localization_radius=fixed_sigma
        )

        for method in methods:

            rmse_heatmap_results[
                method
            ][m_index, inflation_index] = (
                current_results[method]["mean_rmse"]
            )

            spread_heatmap_results[
                method
            ][m_index, inflation_index] = (
                current_results[method]["mean_spread"]
            )

            runtime_heatmap_results[
                method
            ][m_index, inflation_index] = (
                current_results[method]["runtime"]
            )

            print(
                f"{method}: "
                f"RMSE="
                f"{current_results[method]['mean_rmse']:.4f}, "
                f"Spread="
                f"{current_results[method]['mean_spread']:.4f}"
            )

for method in methods:

    rmse_matrix = rmse_heatmap_results[
        method
    ]

    plt.figure(figsize=(11, 6))

    image = plt.imshow(
        rmse_matrix,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(
        image,
        label="Mean analysis RMSE"
    )

    plt.xticks(
        np.arange(
            len(heatmap_inflation_values)
        ),
        [
            f"{value:.2f}"
            for value in heatmap_inflation_values
        ]
    )

    plt.yticks(
        np.arange(
            len(heatmap_m_values)
        ),
        heatmap_m_values
    )

    plt.xlabel("Inflation δ")
    plt.ylabel("Ensemble size m")

    plt.title(
        f"{method}: RMSE for m and inflation\n"
        f"Localization radius σ={fixed_sigma}"
    )

    for i in range(
        len(heatmap_m_values)
    ):
        for j in range(
            len(heatmap_inflation_values)
        ):
            plt.text(
                j,
                i,
                f"{rmse_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.show()