from rlrs.dp import value_iteration
from rlrs.envs import GridWorld

for ruido in (0.0, 0.2, 0.6):
    e = GridWorld(noise=ruido, step_reward=-0.02)
    v, p, n = value_iteration(e, gamma=0.9)
    print(f'\nruido = {ruido}  {n} barridos')
    print(e.render_values(v, p))
