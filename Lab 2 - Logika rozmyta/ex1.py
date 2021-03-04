import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Obserwacje
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')

# Akcje
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# Funkcje przynależności - generowane automatycznie
quality.automf(3)
service.automf(3)
# Funkcje przynależności - tworzenie ręczne
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# Wyświetlanie funkcji przynależności
quality.view()
plt.waitforbuttonpress()
service.view()
plt.waitforbuttonpress()
tip.view()
plt.waitforbuttonpress()

# Reguły
rules = []
rules.append(ctrl.Rule(quality['poor'] | service['poor'], tip['low']))
rules.append(ctrl.Rule(service['average'], tip['medium']))
rules.append(ctrl.Rule(service['good'] | quality['good'], tip['high']))

# Kompilacja reguł i utworzenie sterownika
tipping_ctrl = ctrl.ControlSystem(rules)
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

tipping.input['quality'] = float(input("Jakość jedzenia (od 0 do 10): "))
tipping.input['service'] = float(input("Jakość obsługi (od 0 do 10): "))

# Wyliczenie wyjścia systemu
tipping.compute()
print(tipping.output['tip'])
tip.view(sim=tipping)
plt.waitforbuttonpress()