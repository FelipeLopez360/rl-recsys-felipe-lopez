from rlrs.evaluation import compare
from rlrs.policies import GreedyTabularPolicy, RandomPolicy
from rlrs.dp import value_iteration
from rlrs.envs import GridWorld

RUIDO = 0.2
COSTE = -0.02

mi_env = GridWorld(noise=RUIDO, step_reward=COSTE)
valores, politica, barridos = value_iteration(mi_env, gamma=0.9)
print(f'{barridos} barridos · V(3,0) = {valores[mi_env.state_index((3, 0))]:+.4f}')
print(mi_env.render_values(valores, politica))

for r in compare(mi_env,
                 [GreedyTabularPolicy(politica, name='optima'),
                  RandomPolicy(mi_env.n_actions, seed=0)],
                 episodes=300, base_seed=0):
    print(' ', r)
