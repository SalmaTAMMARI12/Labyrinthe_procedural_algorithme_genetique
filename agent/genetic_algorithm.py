import random
import numpy as np
from agent.genome import RuleBasedGenome


def evaluate_fitness(genome, env, n_episodes=10, seed_start=0):
    """Fitness = récompense moyenne sur n_episodes épisodes avec seeds variés."""
    total = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = genome.decide(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_reward += r
            done = term or trunc
        total += ep_reward
    return total / n_episodes


class GeneticAlgorithm:
    def __init__(self, pop_size=100, n_generations=150,
                 mutation_rate=0.30, elitism_k=2, tournament_size=5):
        self.pop_size        = pop_size
        self.n_generations   = n_generations
        self.mutation_rate   = mutation_rate
        self.elitism_k       = elitism_k
        self.tournament_size = tournament_size

    def _tournament_select(self, population, fitnesses):
        contestants = random.sample(list(zip(fitnesses, population)), self.tournament_size)
        contestants.sort(key=lambda x: x[0], reverse=True)
        return contestants[0][1]

    def _inject_diversity(self, population, fitnesses, ratio=0.15):
        """Remplace les ratio% les plus faibles par des individus aléatoires."""
        n = int(len(population) * ratio)
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        worst_indices = [population.index(ind) for _, ind in sorted_pop[:n]]
        for i in worst_indices:
            population[i] = RuleBasedGenome()
        return population

    def run(self, env):
        population = [RuleBasedGenome() for _ in range(self.pop_size)]
        best_hist = []
        avg_hist  = []
        stagnation = 0
        last_best  = -999

        for gen in range(self.n_generations):
            # Seeds différents à chaque génération pour éviter la sur-adaptation
            seed_start = gen * 10
            fitnesses = [evaluate_fitness(ind, env, seed_start=seed_start)
                         for ind in population]

            best_val = max(fitnesses)
            best_hist.append(best_val)
            avg_hist.append(float(np.mean(fitnesses)))

            print(f"Gen {gen+1:3d}/{self.n_generations}  "
                  f"best={best_hist[-1]:.2f}  avg={avg_hist[-1]:.2f}")

            # Détection stagnation → injection de diversité
            if best_val <= last_best + 0.01:
                stagnation += 1
            else:
                stagnation = 0
                last_best = best_val

            if stagnation >= 10:
                population = self._inject_diversity(population, fitnesses, ratio=0.20)
                stagnation = 0
                print(f"  *** Injection de diversité (gen {gen+1}) ***")

            # Élitisme
            sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
            elites = [ind for _, ind in sorted_pop[:self.elitism_k]]

            # Nouvelle population
            new_pop = elites[:]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                child = RuleBasedGenome.crossover(p1, p2).mutate(self.mutation_rate)
                new_pop.append(child)

            population = new_pop

        # Évaluation finale
        fitnesses = [evaluate_fitness(ind, env) for ind in population]
        best_idx  = int(np.argmax(fitnesses))
        return population[best_idx], best_hist, avg_hist