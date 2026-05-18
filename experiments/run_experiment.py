import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from env.maze_env import MazeEnv
from agent.genetic_algorithm import GeneticAlgorithm, evaluate_fitness
from baselines.random_agent import RandomAgent
from baselines.bfs_agent import BFSAgent

# ------------------------------------------------------------------
N_EVAL = 30
SEEDS  = list(range(N_EVAL))

env_params = dict(size=9, n_keys=1, n_traps=1, max_steps=200)

# ------------------------------------------------------------------
def evaluate_agent(agent, env_params, seeds, bfs=False):
    """bfs=True : l'agent partage l'env pour accéder à grid/agent_pos."""
    env = MazeEnv(**env_params)
    if bfs:
        agent.env = env
    rewards = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.decide(obs))
            total += r
            done = term or trunc
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))

# ------------------------------------------------------------------
# 1. Entraîner l'AG
print("=== Entraînement de l'algorithme génétique ===")
train_env = MazeEnv(**env_params)
ga = GeneticAlgorithm(pop_size=100, n_generations=150, mutation_rate=0.30,elitism_k=2)
best_agent, best_hist, avg_hist = ga.run(train_env)

print("\nMeilleur génome évolué :")
best_agent.describe()

# ------------------------------------------------------------------
# 2. Évaluer toutes les baselines
print("\n=== Évaluation sur 30 épisodes de test ===")

results = {
    "AG (notre agent)": evaluate_agent(best_agent, env_params, SEEDS),
    "Agent aléatoire":  evaluate_agent(RandomAgent(), env_params, SEEDS),
    "Agent BFS":        evaluate_agent(BFSAgent(MazeEnv(**env_params)), env_params, SEEDS, bfs=True),
}

print(f"\n{'Agent':<20} {'Récompense moy':>15}  {'Écart-type':>12}")
print("-" * 50)
for name, (mean, std) in results.items():
    print(f"{name:<20} {mean:>15.2f}  {std:>12.2f}")

# ------------------------------------------------------------------
# 3. Graphiques
os.makedirs("results/plots", exist_ok=True)

# Courbe de convergence
plt.figure(figsize=(10, 4))
plt.plot(best_hist, label="Meilleur fitness")
plt.plot(avg_hist,  label="Fitness moyen", alpha=0.7)
plt.xlabel("Génération")
plt.ylabel("Fitness")
plt.title("Convergence de l'algorithme génétique")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/convergence.png", dpi=150)
print("\nCourbe sauvegardée : results/plots/convergence.png")

# Boxplot comparatif
fig, ax = plt.subplots(figsize=(8, 5))
all_rewards = {}
for name, _ in results.items():
    env_tmp = MazeEnv(**env_params)
    if name == "AG (notre agent)":
        agent = best_agent
    elif name == "Agent aléatoire":
        agent = RandomAgent()
    else:
        agent = BFSAgent(env_tmp)
    rews = []
    for s in SEEDS:
        obs, _ = env_tmp.reset(seed=s)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env_tmp.step(agent.decide(obs))
            total += r
            done = term or trunc
        rews.append(total)
    all_rewards[name] = rews

ax.boxplot(all_rewards.values(), labels=all_rewards.keys())
ax.set_ylabel("Récompense cumulée")
ax.set_title("Comparaison des agents (30 épisodes)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/boxplot.png", dpi=150)
print("Boxplot sauvegardé  : results/plots/boxplot.png")