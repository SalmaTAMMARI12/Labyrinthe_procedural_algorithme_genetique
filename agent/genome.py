import random
import numpy as np

N_RULES = 12  # plus de règles pour plus d'expressivité

# ------------------------------------------------------------------
# CONDITIONS : chaque fonction prend (view 5x5, keys_held, extra_info)
# extra_info contient la position relative de la sortie si connue
# ------------------------------------------------------------------

def _key_visible(view, keys_held, info):
    return int(2 in view)

def _door_visible(view, keys_held, info):
    return int(3 in view)

def _trap_adjacent(view, keys_held, info):
    # zone 3x3 centrale
    center = view[1:4, 1:4]
    return int(4 in center)

def _trap_up(view, keys_held, info):
    return int(view[1][2] == 4)

def _trap_down(view, keys_held, info):
    return int(view[3][2] == 4)

def _trap_left(view, keys_held, info):
    return int(view[2][1] == 4)

def _trap_right(view, keys_held, info):
    return int(view[2][3] == 4)

def _exit_visible(view, keys_held, info):
    return int(5 in view)

def _exit_up(view, keys_held, info):
    # sortie visible dans la moitié haute de la vue
    return int(5 in view[:2, :])

def _exit_down(view, keys_held, info):
    return int(5 in view[3:, :])

def _exit_left(view, keys_held, info):
    return int(5 in view[:, :2])

def _exit_right(view, keys_held, info):
    return int(5 in view[:, 3:])

def _key_up(view, keys_held, info):
    return int(2 in view[:2, :])

def _key_down(view, keys_held, info):
    return int(2 in view[3:, :])

def _key_left(view, keys_held, info):
    return int(2 in view[:, :2])

def _key_right(view, keys_held, info):
    return int(2 in view[:, 3:])

def _has_key(view, keys_held, info):
    return int(keys_held > 0)

def _no_key(view, keys_held, info):
    return int(keys_held == 0)

def _wall_up(view, keys_held, info):
    return int(view[1][2] == 0)

def _wall_down(view, keys_held, info):
    return int(view[3][2] == 0)

def _wall_left(view, keys_held, info):
    return int(view[2][1] == 0)

def _wall_right(view, keys_held, info):
    return int(view[2][3] == 0)

def _path_up(view, keys_held, info):
    return int(view[1][2] in [1, 2, 5])

def _path_down(view, keys_held, info):
    return int(view[3][2] in [1, 2, 5])

def _path_left(view, keys_held, info):
    return int(view[2][1] in [1, 2, 5])

def _path_right(view, keys_held, info):
    return int(view[2][3] in [1, 2, 5])

def _always(view, keys_held, info):
    return 1

CONDITIONS = [
    _key_visible,      # 0
    _door_visible,     # 1
    _trap_adjacent,    # 2
    _exit_visible,     # 3
    _has_key,          # 4
    _no_key,           # 5
    _wall_up,          # 6
    _always,           # 7
    _exit_up,          # 8
    _exit_down,        # 9
    _exit_left,        # 10
    _exit_right,       # 11
    _key_up,           # 12
    _key_down,         # 13
    _key_left,         # 14
    _key_right,        # 15
    _path_up,          # 16
    _path_down,        # 17
    _path_left,        # 18
    _path_right,       # 19
    _trap_up,          # 20
    _trap_down,        # 21
    _trap_left,        # 22
    _trap_right,       # 23
    _wall_down,        # 24
    _wall_left,        # 25
    _wall_right,       # 26
]

N_CONDITIONS = len(CONDITIONS)

# ------------------------------------------------------------------

class RuleBasedGenome:
    """
    Génome = liste de N_RULES règles, chaque règle = [cond_idx, action, priority]
      - cond_idx : int [0, N_CONDITIONS-1]
      - action   : int [0-3]  -> 0=haut 1=bas 2=gauche 3=droite
      - priority : float [0-1]
    """

    def __init__(self, rules=None):
        self.rules = rules if rules is not None else self._random_rules()

    def _random_rules(self):
        return [
            [random.randint(0, N_CONDITIONS - 1),
             random.randint(0, 3),
             random.uniform(0, 1)]
            for _ in range(N_RULES)
        ]

    def decide(self, obs):
        view      = np.array(obs["local_view"])
        keys_held = int(obs["keys_held"])

        active = []
        for cond_idx, action, priority in self.rules:
            if CONDITIONS[cond_idx](view, keys_held, {}):
                active.append((priority, action))

        if not active:
            return random.randint(0, 3)

        active.sort(reverse=True)
        return active[0][1]

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
                    # mutation gaussienne de la priorité
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