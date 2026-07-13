import numpy as np
from function.Lorenz96 import rk4

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