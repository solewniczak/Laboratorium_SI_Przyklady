from pyswip import Prolog
from helper import py2pl

prolog = Prolog()

# Fakty + Reguły
prolog.consult("ex2.pl")

# Zapytania
list1 = [1,2,3,4]
list2 = [5,5,6,6]

queries = [
    f'not_in_list(5, {py2pl(list1)})',
    f'not_in_list(5, {py2pl(list2)})',
    f'all_different({py2pl(list1)})',
    f'all_different({py2pl(list2)})',
    f'domain({py2pl(list1)}, 1, 4)',
    f'domain({py2pl(list2)}, 1, 4)',
]

for query in queries:
    print(query, bool(list(prolog.query(query))))