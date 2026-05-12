from minigrid.manual_control import env
from agent.genetic_algorithm import GeneticAlgorithm
from baselines.bfs_agent import BFSAgent
from baselines.random_agent import RandomAgent
from matplotlib.pyplot import plt

N_EVAL = 30 
SEEDS = list(range(N_EVAL)) 
def evaluate_agent(agent, env_params, seeds): 
    env = MazeEnv(**env_params) 
    rewards = [] 
    for s in seeds: 
        obs, _ = env.reset(seed=s) 
        done, total = False, 0 
        while not done: 
            obs, r, term, trunc, _ = env.step(agent.decide(obs)) 
            total += r 
            done = term or trunc 
        rewards.append(total) 
    return np.mean(rewards), np.std(rewards) 
# Entraîner l'AG 
ga = GeneticAlgorithm(pop_size=50, n_generations=80) 
best_agent, best_hist, avg_hist = ga.run(env) 
# Évaluer toutes les baselines 
results = { 
    'AG (notre agent)': evaluate_agent(best_agent, env_params, SEEDS), 
    'Agent aléatoire':  evaluate_agent(RandomAgent(), env_params, SEEDS), 
    'Agent BFS':        evaluate_agent(BFSAgent(env), env_params, SEEDS), } 
# Courbe de convergence 
plt.figure(figsize=(10, 4)) 
plt.plot(best_hist, label='Meilleur fitness') 
plt.plot(avg_hist, label='Fitness moyen', alpha=0.7) 
plt.xlabel('Génération'); plt.ylabel('Fitness') 
plt.title('Convergence de l\'algorithme génétique') 
plt.legend(); plt.grid(alpha=0.3) 
plt.savefig('results/plots/convergence.png', dpi=150) 