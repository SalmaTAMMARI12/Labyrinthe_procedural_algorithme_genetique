import random
class RuleBasedGenome: 
    def __init__(self, rules=None): 
        self.rules = rules if rules else self._random_rules() 

    def decide(self, obs): 
            """Choisit l'action selon les règles actives les plus prioritaires.""" 
            view = obs['local_view'] 
            keys_held = obs['keys_held'] 
            active = [] 
            for cond_idx, action, priority in self.rules: 
                if self._eval(CONDITIONS[cond_idx], view, keys_held): 
                    active.append((priority, action)) 
            if not active: return random.randint(0, 3) 
            active.sort(reverse=True)  # priorité décroissante 
            return active[0][1] 

    def mutate(self, rate=0.1): 
        """Mutation par gène : modifie condition, action ou priorité.""" 
        rules = [r[:] for r in self.rules] 
        for rule in rules: 
            if random.random() < rate: 
                gene = random.randint(0, 2) 
                if gene == 0: rule[0] = random.randint(0, 7)   # condition 
                elif gene == 1: rule[1] = random.randint(0, 3) # action 
                else: rule[2] = random.uniform(0, 1)           # priorité 
        return RuleBasedGenome(rules) 

    @staticmethod 
    def crossover(p1, p2): 
        """Croisement 1 point sur les règles.""" 
        point = random.randint(1, N_RULES - 1) 
        return RuleBasedGenome(p1.rules[:point] + p2.rules[point:]) 