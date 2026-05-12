from minigrid.manual_control import env

from agent import genome
from agent.genome import RuleBasedGenome
class GeneticAlgorithm: 

    def run(self, env): 
        population = [RuleBasedGenome() for _ in range(self.pop_size)] 
        best_hist, avg_hist = [], [] 
        for gen in range(self.n_generations): 
            # 1. Évaluation de toute la population 
            fitnesses = [self.evaluate_fitness(ind, env) for ind in population] 
            best_hist.append(max(fitnesses)) 
            avg_hist.append(np.mean(fitnesses)) 
            # 2. Élitisme : copier les meilleurs directement 
            sorted_pop = sorted(zip(fitnesses, population), reverse=True) 
            elites = [ind for _, ind in sorted_pop[:self.elitism_k]] 
            # 3. Reproduction par tournoi + croisement + mutation 
            new_pop = elites[:] 
            while len(new_pop) < self.pop_size: 
                p1 = self._tournament_select(population, fitnesses) 
                p2 = self._tournament_select(population, fitnesses) 
                child = RuleBasedGenome.crossover(p1, p2).mutate(self.mutation_rate) 
                new_pop.append(child) 
            population = new_pop 
        best_idx = np.argmax(fitnesses) 
        return population[best_idx], best_hist, avg_hist
    def evaluate_fitness(self, genome, env, n_episodes=5, seed_start=42): 
        total = 0 
        for ep in range(n_episodes): 
            obs, _ = env.reset(seed=seed_start + ep) 
            done, ep_reward = False, 0 
            while not done: 
                action = genome.decide(obs) 
                obs, r, term, trunc, _ = env.step(action) 
                ep_reward += r 
                done = term or trunc 
            total += ep_reward 
        return total / n_episodes 