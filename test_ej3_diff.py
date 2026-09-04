from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
import numpy as np

RUIDO = 0.2
COSTE = -0.02
mi_env = GridWorld(noise=RUIDO, step_reward=COSTE)

v05, p05, n05 = value_iteration(mi_env, gamma=0.5)
v09, p09, n09 = value_iteration(mi_env, gamma=0.9)
v99, p99, n99 = value_iteration(mi_env, gamma=0.99)

print("0.5 vs 0.9", np.sum(p05 != p09))
print("0.9 vs 0.99", np.sum(p09 != p99))
