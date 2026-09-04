from rlrs.dp import value_iteration
from rlrs.envs import GridWorld

RUIDO = 0.2
COSTE = -0.02
mi_env = GridWorld(noise=RUIDO, step_reward=COSTE)

for g in (0.5, 0.9, 0.99):
    v, p, n = value_iteration(mi_env, gamma=g)
    print(f'\ngamma = {g} {n} barridos  V(3,0) = {v[mi_env.state_index((3, 0))]:+.4f}')
    print(mi_env.render_values(v, p))
