import numpy as np

# Lorenz96 model
def lorenz96(x, F = 8.0):
  #dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
  N = len (x)
  dxdt = np.zeros(N)

  for i in range(N):
     x_prev2 = x[(i - 2) % N]  # X_{j-2}
     x_prev1 = x[(i - 1) % N]  # X_{j-1}
     x_next1 = x[(i + 1) % N]  # X_{j+1}
     dxdt[i] = (x_next1 - x_prev2) * x_prev1 - x[i] + F

  return dxdt

#RK4
def rk4(x, dt, F=8.0):
  k1 = lorenz96(x, F)
  k2 = lorenz96(x + 0.5 * dt * k1, F)
  k3 = lorenz96(x + 0.5 * dt * k2, F)
  k4 = lorenz96(x + dt * k3, F)
  x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

  return x_next