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
    for obs_index in range(N):

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



def make_initial_ensemble_from_spinup(
    spinup_states,
    initial_background,
    m,
    seed=123,
    recenter=True,
    spread_scale=1.0
):
    rng = np.random.Generator(np.random.MT19937(seed))

    indices = rng.integers(0, len(spinup_states), size=m)
    ensemble = spinup_states[indices].copy()

    ensemble_mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - ensemble_mean

    perturbations = spread_scale * perturbations

    if recenter:
        ensemble = initial_background + perturbations
    else:
        ensemble = ensemble_mean + perturbations

    return ensemble

#局所化するための距離に応じた重み付け
def create_localization_matrix(N, localization_radius):
    loc_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            distance = abs(i - j)
            cyclic_distance = min(distance, N - distance)
            r = cyclic_distance / (np.sqrt(10.0 / 3.0)* localization_radius)
            loc_matrix[i, j] = gaspari_cohn(np.array([r]))[0]

    return loc_matrix

#局所化するための距離に応じた重み付けの値の計算
def gaspari_cohn(r):
    r = np.abs(r)
    rho = np.zeros_like(r)

    mask1 = r <= 1
    rr = r[mask1]
    rho[mask1] = (
        1
        - 5/3 * rr**2
        + 5/8 * rr**3
        + 1/2 * rr**4
        - 1/4 * rr**5
    )

    mask2 = (r > 1) & (r <= 2)
    rr = r[mask2]
    rho[mask2] = (
        4
        - 5 * rr
        + 5/3 * rr**2
        + 5/8 * rr**3
        - 1/2 * rr**4
        + 1/12 * rr**5
        - 2/(3 * rr)
    )

    return rho