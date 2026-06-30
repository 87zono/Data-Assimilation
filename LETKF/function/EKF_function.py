import numpy as np
from function.Lorenz96 import rk4

#Tangent Linear Matrix (to create matrix point)
#\delta x(t+dt)=M \delta x(t)
def tangent_linear_matrix (x, dt):
  N = len(x)
  M = np.zeros((N,N))

  for j in range(N):
    j_minus2 = (j - 2)% N
    j_minus1 = (j - 1)% N
    j_plus1 = (j + 1) % N

    M[j, j_minus2] = -x[j_minus1] * dt
    M[j, j_minus1] = (x[j_plus1] - x[j_minus2]) * dt
    M[j, j] = 1.0 - dt
    M[j, j_plus1] = x[j_minus1] * dt

  return M

#KF cycle
def EKF_cycle(xa, Pa, y_obs, dt, F, H, R, inflation):
  #1. tangent linear matrix (M)
  M = tangent_linear_matrix(xa, dt)

  #2. prediction forecast state (xb)
  xb = rk4(xa, dt, F)

  #3. prediction covariance (Pb)
  Pb = M @ Pa @ M.T
  Pb = (1 + inflation) * Pb

  #4. Kalman Gain
  S = H @ Pb @ H.T + R
  K = Pb @ H.T @ np.linalg.inv(S)

  #5. analysis
  xa_new = xb + K @ (y_obs - H @ xb)

  #6. analysis covarience
  I = np.eye(len(xa))
  Pa_new = (I - K @ H) @ Pb

  return xa_new, Pa_new, xb, Pb, K