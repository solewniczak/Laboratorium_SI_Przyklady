from pyswip import Prolog

prolog = Prolog()

# Fakty + Reguły
prolog.consult("ex1.pl")

# Zapytania
print(bool(list(prolog.query("ancestor(james,john)"))))
print(bool(list(prolog.query("ancestor(gina,john)"))))

for soln in prolog.query("ancestor(Who,michael)"):
    print(soln["Who"], "is the ancestor of michael")

for soln in prolog.query("father(X,Y)"):
    print(soln["X"], "is the father of", soln["Y"])
