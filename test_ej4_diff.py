from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
import numpy as np

e0 = GridWorld(noise=0.0, step_reward=-0.02)
v0, p0, n0 = value_iteration(e0, gamma=0.9)

e2 = GridWorld(noise=0.2, step_reward=-0.02)
v2, p2, n2 = value_iteration(e2, gamma=0.9)

print("0.0 vs 0.2", np.sum(p0 != p2))
for i in range(20):
    if p0[i] != p2[i]:
        print(f"diff at {e0.state_pos(i)}: 0.0={p0[i]} 0.2={p2[i]}")
