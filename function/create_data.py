import numpy as np
from function.Lorenz96 import rk4

#creating data for 2 years
def createdata(N, F, dt):
    steps_per_years = 365 * 4
    total_steps = steps_per_years * 2

    x = F * np.ones(N)
    x[0] += 0.01  #small perturbation

    all_data = np.zeros((total_steps+1, N))
    all_data[0] = x

    for t in range(total_steps):
        x = rk4(x, dt, F)
        all_data[t+1] = x

    truth = all_data[steps_per_years+1:]

    rng = np.random.Generator(np.random.MT19937(seed = 42))
    noise = rng.normal(loc=0.0, scale=1.0, size=truth.shape)

    obs = truth + noise

    return truth, obs, all_data[:1461]

def calculate_rmse(estimate, truth):
    return np.sqrt(np.mean((estimate - truth) ** 2))



def calculate_global_spread(ensemble):
    """
    アンサンブルから全体Spreadを計算する。
    Spread = sqrt(trace(P) / N)
    """
    m, N = ensemble.shape
    ensemble_mean = np.mean(
        ensemble,
        axis=0
    )
    perturbations = ( ensemble - ensemble_mean )
    covariance = ( perturbations.T @ perturbations ) / (m - 1)
    spread = np.sqrt(np.trace(covariance) / N )

    return spread