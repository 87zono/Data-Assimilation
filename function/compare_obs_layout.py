import numpy as np
import matplotlib.pyplot as plt

from function.create_data import createdata
from function.create_ensemble_data import  make_initial_ensemble_from_spinup, create_localization_matrix
from function.obs_number import create_observation_setup
from function.create_data import calculate_global_spread, calculate_rmse

from function.EKF_function import EKF_cycle
from function.ThreeD_var import threedvar_cycle_gradient_descent
from function.PO_EnKF_function import PO_EnKF_cycle
from function.Serial_EnKF_partial import Serial_EnSRF_cycle_partial
from function.LETKF_partial import   LETKF_cycle_partial

def run_EKF_partial(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    initial_background,
):
    N = truth.shape[1]

    xa = initial_background.copy()
    Pa = 10.0 * np.eye(N)

    rmse_a_list = []

    for t in range(1, len(truth)):
        xa, Pa, xb, Pb, K = EKF_cycle(
            xa,
            Pa,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation,
        )

        rmse_a_list.append(
            calculate_rmse(xa, truth[t])
        )

    return {
        "rmse_a": np.asarray(rmse_a_list)
    }

def run_3DVar_partial(
    truth,
    obs,
    dt,
    F,
    H,
    B_inv,
    R_inv,
    initial_background,
):
    xa = initial_background.copy()

    rmse_a_list = []

    for t in range(1, len(truth)):
        xa, xb = threedvar_cycle_gradient_descent(
            xa,
            obs[t],
            dt,
            F,
            H,
            B_inv,
            R_inv,
        )

        rmse_a_list.append(
            calculate_rmse(xa, truth[t])
        )

    return {
        "rmse_a": np.asarray(rmse_a_list)
    }

def run_PO_EnKF_partial(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_matrix,
    m,
    spinup_states,
    initial_background,
):
    ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42,
    )

    rmse_a_list = []

    for t in range(1, len(truth)):
        (
            ensemble_a,
            xa_mean,
            Pa_approx,
            spread_a,
            spread_by_state,
        ) = PO_EnKF_cycle(
            ensemble_a,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation,
            localization_matrix,
        )

        rmse_a_list.append(
            calculate_rmse(xa_mean, truth[t])
        )

    return {
        "rmse_a": np.asarray(rmse_a_list)
    }

def run_Serial_EnSRF_partial(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_matrix,
    m,
    spinup_states,
    initial_background,
):
    ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42,
    )

    rmse_a_list = []

    for t in range(1, len(truth)):
        (
            ensemble_a,
            ensemble_b,
            xb_mean,
            xa_mean,
            Pa_approx,
            spread_a,
        ) = Serial_EnSRF_cycle_partial(
            ensemble_a,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation,
            localization_matrix,
        )

        rmse_a_list.append(
            calculate_rmse(xa_mean, truth[t])
        )

    return {
        "rmse_a": np.asarray(rmse_a_list)
    }

def run_LETKF_partial(
    truth,
    obs,
    dt,
    F,
    H,
    R,
    inflation,
    localization_radius_sigma,
    m,
    spinup_states,
    initial_background,
):
    ensemble_a = make_initial_ensemble_from_spinup(
        spinup_states=spinup_states,
        initial_background=initial_background,
        m=m,
        seed=42,
    )

    rmse_a_list = []

    for t in range(1, len(truth)):
        (
            ensemble_a,
            ensemble_b,
            xb_mean,
            xa_mean,
            spread_by_state,
        ) = LETKF_cycle_partial(
            ensemble_a,
            obs[t],
            dt,
            F,
            H,
            R,
            inflation,
            localization_radius_sigma,
        )

        rmse_a_list.append(
            calculate_rmse(xa_mean, truth[t])
        )

    return {
        "rmse_a": np.asarray(rmse_a_list)
    }

N = 40
F = 8.0
dt = 0.05

m = 20
inflation = 0.04
localization_radius = 4.0

B_value = 0.25
B_inv = (1.0 / B_value) * np.eye(N)

observation_counts = [20, 25, 30, 35, 40]
observation_modes = ["homogeneous", "dense"]

burn_in = 100

truth, full_obs, spinup_states = createdata(
    N=N,
    F=F,
    dt=dt,
)

initial_background = full_obs[0].copy()

localization_matrix = create_localization_matrix(
    N=N,
    localization_radius=localization_radius,
)
methods = [
    "EKF",
    "3D-Var",
    "PO-EnKF",
    "Serial EnSRF",
    "LETKF",
]

results = {
    mode: {
        method: []
        for method in methods
    }
    for mode in observation_modes
}
for mode in observation_modes:

    print(f"\n===== {mode.upper()} =====")

    for num_obs in observation_counts:

        print(
            f"\nNumber of observations = {num_obs}"
        )

        obs_indices, H, R = create_observation_setup(
            N=N,
            num_obs=num_obs,
            mode=mode,
        )

        obs = full_obs[:, obs_indices]
        R_inv = np.linalg.inv(R)

        ekf_result = run_EKF_partial(
            truth=truth,
            obs=obs,
            dt=dt,
            F=F,
            H=H,
            R=R,
            inflation=inflation,
            initial_background=initial_background,
        )

        var_result = run_3DVar_partial(
            truth=truth,
            obs=obs,
            dt=dt,
            F=F,
            H=H,
            B_inv=B_inv,
            R_inv=R_inv,
            initial_background=initial_background,
        )

        po_result = run_PO_EnKF_partial(
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
            initial_background=initial_background,
        )

        ensrf_result = run_Serial_EnSRF_partial(
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
            initial_background=initial_background,
        )

        letkf_result = run_LETKF_partial(
            truth=truth,
            obs=obs,
            dt=dt,
            F=F,
            H=H,
            R=R,
            inflation=inflation,
            localization_radius_sigma=localization_radius,
            m=m,
            spinup_states=spinup_states,
            initial_background=initial_background,
        )

        mean_values = {
            "EKF": np.mean(
                ekf_result["rmse_a"][burn_in:]
            ),
            "3D-Var": np.mean(
                var_result["rmse_a"][burn_in:]
            ),
            "PO-EnKF": np.mean(
                po_result["rmse_a"][burn_in:]
            ),
            "Serial EnSRF": np.mean(
                ensrf_result["rmse_a"][burn_in:]
            ),
            "LETKF": np.mean(
                letkf_result["rmse_a"][burn_in:]
            ),
        }

        for method in methods:
            results[mode][method].append(
                mean_values[method]
            )

        print(
            ", ".join(
                f"{method}={mean_values[method]:.4f}"
                for method in methods
            )
        )

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6),
)

for method in methods:
    axes[0].plot(
        observation_counts,
        results["homogeneous"][method],
        marker="o",
        label=method,
    )

axes[0].set_title(
    "Case 1: Homogeneous observation placement"
)
axes[0].set_xlabel(
    "Number of observations"
)
axes[0].set_ylabel(
    "Mean analysis RMSE"
)
axes[0].set_xticks(
    observation_counts
)
axes[0].grid()
axes[0].legend()


for method in methods:
    axes[1].plot(
        observation_counts,
        results["dense"][method],
        marker="o",
        label=method,
    )

axes[1].set_title(
    "Case 2: Dense observation placement"
)
axes[1].set_xlabel(
    "Number of observations"
)
axes[1].set_ylabel(
    "Mean analysis RMSE"
)
axes[1].set_xticks(
    observation_counts
)
axes[1].grid()
axes[1].legend()

fig.suptitle(
    "Effect of Observation Number and Placement\n"
    f"m={m}, inflation={inflation}, "
    f"localization radius={localization_radius}"
)

plt.tight_layout()
plt.show()