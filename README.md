# Résolution de Labyrinthes Procéduraux par Algorithme Génétique

---

## Description

Un agent intelligent apprend à naviguer dans des labyrinthes 2D générés procéduralement. Il doit collecter des clés pour ouvrir des portes, éviter des pièges mortels et atteindre la sortie dans un nombre limité de pas.

L'agent est basé sur un **algorithme génétique** opérant sur un **génome à règles prioritaires** (RuleBasedGenome). L'environnement respecte l'API standard **Gymnasium**.

---

## Résultats obtenus

| Agent | Récompense µ | Écart-type σ |
|---|---|---|
| Agent BFS (borne supérieure) | 2.62 | 10.30 |
| **Agent AG (notre agent)** | **1.35** | **5.95** |
| Agent aléatoire (borne inférieure) | -4.93 | 3.72 |

> Évaluation sur 30 épisodes de test — labyrinthe 9×9, 1 clé, 1 piège, 200 pas max

---

## Structure du projet

```
projet-JV/
├── env/
│   ├── __init__.py
│   ├── maze_env.py          # Environnement Gymnasium (MazeEnv)
│   └── maze_generator.py    # Génération procédurale (recursive backtracking)
├── agent/
│   ├── __init__.py
│   ├── genome.py            # Génome à règles prioritaires (RuleBasedGenome)
│   └── genetic_algorithm.py # Algorithme génétique (GeneticAlgorithm)
├── baselines/
│   ├── __init__.py
│   ├── random_agent.py      # Agent aléatoire (borne inférieure)
│   └── bfs_agent.py         # Agent BFS optimal (borne supérieure)
├── experiments/
│   └── run_experiment.py    # Script principal
├── results/
│   └── plots/
│       ├── convergence.png  # Courbe de convergence de l'AG
│       └── boxplot.png      # Comparaison des 3 agents
└── requirements.txt
```

---

## Installation

### Prérequis
- Python 3.10+
- pip

### Installer les dépendances

```bash
pip install gymnasium numpy matplotlib
```

---

## Lancer le projet

Depuis la racine `projet-JV/` :

```bash
python -m experiments.run_experiment
```

Le script va :
1. Entraîner l'AG sur 150 générations (quelques minutes)
2. Afficher une démonstration visuelle ASCII de l'agent dans le terminal
3. Évaluer les 3 agents sur 30 épisodes
4. Afficher le tableau comparatif
5. Sauvegarder les graphiques dans `results/plots/`

---

## Environnement — MazeEnv

### Paramètres utilisés

| Paramètre | Valeur |
|---|---|
| Taille de la grille | 9 × 9 |
| Nombre de clés/portes | 1 |
| Nombre de pièges | 1 |
| Pas maximum | 200 |

### Espace d'observation

| Composante | Type | Description |
|---|---|---|
| `local_view` | Box(5,5) int [0-5] | Vue locale 5×5 centrée sur l'agent |
| `keys_held` | Discrete(2) | Nombre de clés portées |
| `steps_left` | Box(1,) float [0-1] | Fraction de temps restant |

Encodage des cases : `0`=mur `1`=chemin `2`=clé `3`=porte `4`=piège `5`=sortie

### Fonction de récompense

| Événement | Récompense |
|---|---|
| Atteindre la sortie | +10.0 |
| Collecter une clé | +1.0 |
| Ouvrir une porte | +0.5 |
| Pas normal | -0.01 |
| Frapper un mur | -0.05 |
| Porte sans clé | -0.10 |
| Tomber dans un piège | -2.0 + fin |

---

## Agent — RuleBasedGenome

Le génome est une liste de **12 règles**, chaque règle = `[condition_index, action, priorité]`

- **27 conditions** disponibles : détection directionnelle de la sortie, clés, pièges, murs, chemins libres
- **4 actions** : 0=haut, 1=bas, 2=gauche, 3=droite
- **Priorité** : float [0-1], la règle active avec la plus haute priorité gagne

### Mécanismes anti-boucle
- **Mémoire de visite** : pénalise les (vue, action) déjà jouées × 0.1
- **Anti-oscillation** : si action = opposée de la précédente → choisit une autre action

---

## Algorithme Génétique

### Paramètres

| Hyperparamètre | Valeur |
|---|---|
| Taille de population | 100 |
| Nombre de générations | 150 |
| Taux de mutation | 0.30 |
| Élitisme k | 2 |
| Taille du tournoi | 5 |
| Épisodes par évaluation | 10 |

### Opérateurs
- **Sélection** : tournoi de taille 5
- **Croisement** : mono-point aléatoire
- **Mutation** : perturbation gaussienne des priorités, remplacement aléatoire des conditions/actions
- **Élitisme** : 2 meilleurs conservés intacts
- **Injection de diversité** : si stagnation ≥ 10 générations → 20% des pires remplacés par des génomes aléatoires

### Seeds variables
Les seeds d'évaluation changent à chaque génération (`seed_start = gen × 10`) pour éviter la sur-adaptation à un labyrinthe fixe.

---

## Baselines

**RandomAgent** — choisit une action aléatoire parmi {0,1,2,3} à chaque pas, sans utiliser l'observation. Borne inférieure.

**BFSAgent** — calcule le chemin optimal par recherche en largeur sur la grille complète à chaque pas. Dispose d'une vision totale de la grille. Borne supérieure théorique.

---

## Légende de la démonstration ASCII

```
A = Agent
K = Clé
D = Porte
X = Piège
E = Sortie
█ = Mur
```

**Réalisé par :** Salma TAMMARI & Assmaa EL HIDANI  
