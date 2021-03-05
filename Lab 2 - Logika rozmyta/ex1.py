import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Obserwacje
dystans = ctrl.Antecedent(np.arange(0, 100, 1), 'dystans')

# Akcje
hamowanie = ctrl.Consequent(np.arange(0, 500, 1), 'hamowanie')

# Funkcje przynależności - generowane automatycznie
dystans.automf(3, names=['maly', 'sredni', 'duzy'])

# Funkcje przynależności - tworzenie ręczne
hamowanie['slabe'] = fuzz.trimf(hamowanie.universe, [0, 0, 250])
hamowanie['srednie'] = fuzz.trimf(hamowanie.universe, [0, 250, 500])
hamowanie['silne'] = fuzz.trimf(hamowanie.universe, [250, 500, 500])

dystans.view()
plt.waitforbuttonpress()
hamowanie.view()
plt.waitforbuttonpress()

rules = []
rules.append(ctrl.Rule(dystans['maly'], hamowanie['silne']))
rules.append(ctrl.Rule(dystans['sredni'], hamowanie['srednie']))
rules.append(ctrl.Rule(dystans['duzy'] , hamowanie['slabe']))

system_ctrl = ctrl.ControlSystem(rules)
system = ctrl.ControlSystemSimulation(system_ctrl)

system.input['dystans'] = float(input("Dystans od przeszkody (od 0 do 100): "))

# Wyliczenie wyjścia systemu
system.compute()
print(system.output['hamowanie'], 'N')
hamowanie.view(sim=system)
plt.waitforbuttonpress()