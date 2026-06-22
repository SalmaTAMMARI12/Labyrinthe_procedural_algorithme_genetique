import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import matplotlib.pyplot as plt

from env.maze_env import MazeEnv
from agent.genetic_algorithm import GeneticAlgorithm, evaluate_fitness
from baselines.random_agent import RandomAgent
from baselines.bfs_agent import BFSAgent

N_EVAL = 30
SEEDS  = list(range(N_EVAL))
BFS_SEEDS = SEEDS[:5]

env_params = dict(size=9, n_keys=1, n_traps=1, max_steps=200)

def evaluate_agent(agent, env_params, seeds, bfs=False):
    """Retourne (mean, std, liste_rewards) pour pouvoir réutiliser les rewards."""
    env = MazeEnv(**env_params)
    max_guard = env_params.get('max_steps', 200) + 10
    rewards = []

    for s in seeds:
        obs, _ = env.reset(seed=s)
        current_agent = BFSAgent(env) if bfs else agent
        if hasattr(current_agent, 'reset_memory'):
            current_agent.reset_memory()

        done, total, step_count = False, 0.0, 0
        while not done and step_count < max_guard:
            action = current_agent.decide(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
            step_count += 1
        rewards.append(total)

    return float(np.mean(rewards)), float(np.std(rewards)), rewards


print("=== Entraînement de l'algorithme génétique ===")
train_env = MazeEnv(**env_params)
ga = GeneticAlgorithm(pop_size=100, n_generations=150, mutation_rate=0.30, elitism_k=2)
best_agent, best_hist, avg_hist = ga.run(train_env)

print("\nMeilleur génome évolué :")
best_agent.describe()

print("\n=== Démonstration de l'agent AG (seed=0) ===")
time.sleep(1)

demo_env = MazeEnv(**env_params)
obs, _ = demo_env.reset(seed=0)
best_agent.reset_memory()
done, step_count, total_demo = False, 0, 0.0

while not done:
    os.system('cls')
    print(f"=== Agent AG — Pas {step_count} | Récompense cumulée: {total_demo:.2f} ===")
    print("Légende : A=Agent  K=Clé  D=Porte  X=Piège  E=Sortie  █=Mur\n")
    demo_env.render()
    action = best_agent.decide(obs)
    print(f"\nAction choisie : {['haut','bas','gauche','droite'][action]}")
    obs, r, term, trunc, _ = demo_env.step(action)
    total_demo += r
    step_count += 1
    done = term or trunc
    time.sleep(0.25)

os.system('cls')
print(f"=== Fin de l'épisode — {step_count} pas | Récompense finale: {total_demo:.2f} ===")
demo_env.render()
if total_demo > 5:
    print("\n L'agent a trouvé la sortie ")
elif total_demo < -1.5:
    print("\n L'agent est tombé dans un piège ou a timeout.")
else:
    print("\n~ Episode terminé.")
time.sleep(2)

print("\n=== Évaluation sur 30 épisodes de test ===")

print("   évaluation AG...")
ag_mean, ag_std, ag_rews   = evaluate_agent(best_agent, env_params, SEEDS)
print("   évaluation aléatoire...")
rd_mean, rd_std, rd_rews   = evaluate_agent(RandomAgent(), env_params, SEEDS)
print("   évaluation BFS...")
bfs_mean, bfs_std, bfs_rews = evaluate_agent(None, env_params, BFS_SEEDS, bfs=True)
print("   done")

print(f"\n{'Agent':<20} {'Récompense moy':>15}  {'Écart-type':>12}")
print("-" * 50)
print(f"{'AG (notre agent)':<20} {ag_mean:>15.2f}  {ag_std:>12.2f}")
print(f"{'Agent aléatoire':<20} {rd_mean:>15.2f}  {rd_std:>12.2f}")
print(f"{'Agent BFS':<20} {bfs_mean:>15.2f}  {bfs_std:>12.2f}")

os.makedirs("results/plots", exist_ok=True)

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

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(
    [ag_rews, rd_rews, bfs_rews],
    tick_labels=["AG (notre agent)", "Agent aléatoire", "Agent BFS"]
)
ax.set_ylabel("Récompense cumulée")
ax.set_title("Comparaison des agents (30 épisodes)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/boxplot.png", dpi=150)
print("Boxplot sauvegardé  : results/plots/boxplot.png")