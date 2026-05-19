import random
import numpy as np

N_RULES = 12

# ------------------------------------------------------------------
def _key_visible(view, keys_held, info):   return int(2 in view)
def _door_visible(view, keys_held, info):  return int(3 in view)
def _trap_adjacent(view, keys_held, info): return int(4 in view[1:4, 1:4])
def _trap_up(view, keys_held, info):       return int(view[1][2] == 4)
def _trap_down(view, keys_held, info):     return int(view[3][2] == 4)
def _trap_left(view, keys_held, info):     return int(view[2][1] == 4)
def _trap_right(view, keys_held, info):    return int(view[2][3] == 4)
def _exit_visible(view, keys_held, info):  return int(5 in view)
def _exit_up(view, keys_held, info):       return int(5 in view[:2, :])
def _exit_down(view, keys_held, info):     return int(5 in view[3:, :])
def _exit_left(view, keys_held, info):     return int(5 in view[:, :2])
def _exit_right(view, keys_held, info):    return int(5 in view[:, 3:])
def _key_up(view, keys_held, info):        return int(2 in view[:2, :])
def _key_down(view, keys_held, info):      return int(2 in view[3:, :])
def _key_left(view, keys_held, info):      return int(2 in view[:, :2])
def _key_right(view, keys_held, info):     return int(2 in view[:, 3:])
def _has_key(view, keys_held, info):       return int(keys_held > 0)
def _no_key(view, keys_held, info):        return int(keys_held == 0)
def _wall_up(view, keys_held, info):       return int(view[1][2] == 0)
def _wall_down(view, keys_held, info):     return int(view[3][2] == 0)
def _wall_left(view, keys_held, info):     return int(view[2][1] == 0)
def _wall_right(view, keys_held, info):    return int(view[2][3] == 0)
def _path_up(view, keys_held, info):       return int(view[1][2] in [1, 2, 5])
def _path_down(view, keys_held, info):     return int(view[3][2] in [1, 2, 5])
def _path_left(view, keys_held, info):     return int(view[2][1] in [1, 2, 5])
def _path_right(view, keys_held, info):    return int(view[2][3] in [1, 2, 5])
def _always(view, keys_held, info):        return 1

CONDITIONS = [
    _key_visible, _door_visible, _trap_adjacent, _exit_visible,
    _has_key, _no_key, _wall_up, _always,
    _exit_up, _exit_down, _exit_left, _exit_right,
    _key_up, _key_down, _key_left, _key_right,
    _path_up, _path_down, _path_left, _path_right,
    _trap_up, _trap_down, _trap_left, _trap_right,
    _wall_down, _wall_left, _wall_right,
]

N_CONDITIONS = len(CONDITIONS)

# ------------------------------------------------------------------

class RuleBasedGenome:

    def __init__(self, rules=None):
        self.rules = rules if rules is not None else self._random_rules()
        self._visited = {}
        self._last_action = None

    def _random_rules(self):
        return [
            [random.randint(0, N_CONDITIONS - 1),
             random.randint(0, 3),
             random.uniform(0, 1)]
            for _ in range(N_RULES)
        ]

    def reset_memory(self):
        """Réinitialise la mémoire de visite entre les épisodes."""
        self._visited = {}
        self._last_action = None

    def decide(self, obs):
        view      = np.array(obs["local_view"])
        keys_held = int(obs["keys_held"])

        active = []
        for cond_idx, action, priority in self.rules:
            if CONDITIONS[cond_idx](view, keys_held, {}):
                visit_penalty = self._visited.get((view.tobytes(), action), 0) * 0.1
                active.append((priority - visit_penalty, action))

        if not active:
            action = random.randint(0, 3)
        else:
            active.sort(reverse=True)
            action = active[0][1]

        # FIX anti-oscillation : on évite l'opposé SEULEMENT si une
        # autre action est praticable (non-mur). Sinon on garde l'action
        # originale pour ne pas boucler indéfiniment sur des murs.
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if self._last_action is not None and action == opposites[self._last_action]:
            # Actions libres = pas de mur dans la vue locale
            wall_for = {
                0: view[1][2] == 0,  # haut
                1: view[3][2] == 0,  # bas
                2: view[2][1] == 0,  # gauche
                3: view[2][3] == 0,  # droite
            }
            other = [
                a for a in [0, 1, 2, 3]
                if a != action and a != self._last_action and not wall_for[a]
            ]
            if other:
                action = random.choice(other)
            # si other est vide (couloir) → on garde l'action choisie par les règles

        key = (view.tobytes(), action)
        self._visited[key] = self._visited.get(key, 0) + 1
        self._last_action = action
        return action

    def mutate(self, rate=0.15):
        rules = [r[:] for r in self.rules]
        for rule in rules:
            if random.random() < rate:
                gene = random.randint(0, 2)
                if gene == 0:
                    rule[0] = random.randint(0, N_CONDITIONS - 1)
                elif gene == 1:
                    rule[1] = random.randint(0, 3)
                else:
                    rule[2] = float(np.clip(rule[2] + random.gauss(0, 0.2), 0, 1))
        return RuleBasedGenome(rules)

    @staticmethod
    def crossover(p1, p2):
        point = random.randint(1, N_RULES - 1)
        return RuleBasedGenome(p1.rules[:point] + p2.rules[point:])

    def describe(self):
        cond_names = [
            "key_visible", "door_visible", "trap_adjacent", "exit_visible",
            "has_key", "no_key", "wall_up", "always",
            "exit_up", "exit_down", "exit_left", "exit_right",
            "key_up", "key_down", "key_left", "key_right",
            "path_up", "path_down", "path_left", "path_right",
            "trap_up", "trap_down", "trap_left", "trap_right",
            "wall_down", "wall_left", "wall_right",
        ]
        action_names = ["haut", "bas", "gauche", "droite"]
        print(f"{'Condition':<18} {'Action':<10} {'Priorité'}")
        print("-" * 42)
        for cond_idx, action, priority in sorted(self.rules, key=lambda r: -r[2]):
            print(f"{cond_names[cond_idx]:<18} {action_names[action]:<10} {priority:.3f}")