from rlrs.dp import value_iteration
from rlrs.envs import GridWorld
from rlrs.evaluation import evaluate
from rlrs.policies import GreedyTabularPolicy

for coste in (-0.001, -0.02, -2.0):
    e = GridWorld(step_reward=coste, noise=0.2)
    v, p, n = value_iteration(e, gamma=0.9)
    ev = evaluate(GridWorld(step_reward=coste, noise=0.2), GreedyTabularPolicy(p), episodes=300)
    print(f'coste {coste:>7} · {n:>2} barridos · {ev}')
    print(e.render_values(v, p), '\n')
