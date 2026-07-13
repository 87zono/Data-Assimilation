import numpy as np
from function.Lorenz96 import rk4
 
def Serial_EnSRF_cycle(ensrf_ensemble_a, y_obs, dt, F, H, R, inflation, localization_matrix):
    m, N = ensrf_ensemble_a.shape

    #Forecast each member
    ensrf_ensemble_b = np.zeros_like(ensrf_ensemble_a)

    for i in range(m):
        ensrf_ensemble_b[i] = rk4(ensrf_ensemble_a[i], dt, F)

    #Forecast mean for RMSE
    ensrf_xb_mean = np.mean(ensrf_ensemble_b, axis=0)

    #Forecast perturbations
    ensrf_x_mean = np.mean(ensrf_ensemble_b, axis=0)
    ensrf_X = ensrf_ensemble_b - ensrf_x_mean  # δx = xb - xb_mean
    #inflation only once
    ensrf_X = np.sqrt(1.0 + inflation) * ensrf_X
    ensrf_ensemble = ensrf_x_mean + ensrf_X

    #Serial observation loop
    #Serial observation loop (実際の観測数でループを回す)
    num_obs = len(y_obs)
    for obs_index in range(num_obs):

      # Current ensemble mean and perturbations
        ensrf_x_mean = np.mean(ensrf_ensemble, axis=0)
        ensrf_X = ensrf_ensemble - ensrf_x_mean

      # Since H = I, the predicted observation is x[obs_index]
        yb_mean = ensrf_x_mean[obs_index]

      # Observation-space perturbation
        # y_pert_i = H x_pert_i
        y_pert = ensrf_X[:, obs_index]   # shape (m,)

      # Background covariance between state and this observation
        Pxy = (ensrf_X.T @ y_pert) / (m - 1)   # shape (N,)

      # Background variance in observation space
        Pyy = (y_pert @ y_pert) / (m - 1)      # scalar

      # Observation error variance
        R_j = R[obs_index, obs_index]

      # Kalman gain for this observation
        K = Pxy / (Pyy + R_j)                  # shape (N,)

      # 4. Localization
        rho = localization_matrix[:, obs_index]
        K_loc = rho * K

      # 5. Mean update
        innovation = y_obs[obs_index] - yb_mean
        ensrf_x_mean_new = ensrf_x_mean + K_loc * innovation

      # 6. Perturbation update
        # K_tilde = K / (1 + sqrt(R / (Pyy + R)))
        alpha = 1.0 / (1.0 + np.sqrt(R_j / (Pyy + R_j)))
        K_tilde_loc = alpha * K_loc

        # Each member perturbation update:
        # x'_i = x_i - K_tilde * y'_
        ensrf_X_new = ensrf_X - np.outer(y_pert, K_tilde_loc)

        # Reconstruct ensemble
        ensrf_ensemble = ensrf_x_mean_new + ensrf_X_new

    # 7. Final analysis
    ensrf_ensemble_a_new = ensrf_ensemble
    ensrf_xa_mean = np.mean(ensrf_ensemble_a_new, axis=0)
    ensrf_Xa = ensrf_ensemble_a_new - ensrf_xa_mean
    ensrf_Pa_approx = (ensrf_Xa.T @ ensrf_Xa) / (m - 1)
    ensrf_spread_a = np.sqrt(np.trace(ensrf_Pa_approx) / N)

    return (
        ensrf_ensemble_a_new,
        ensrf_ensemble_b,
        ensrf_xb_mean,
        ensrf_xa_mean,
        ensrf_Pa_approx,
        ensrf_spread_a
    )
