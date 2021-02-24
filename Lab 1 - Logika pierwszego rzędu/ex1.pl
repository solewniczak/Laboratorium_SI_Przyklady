father(michael,john).
father(michael,gina).
father(james,michael).
father(michael,gina).

ancestor(X,Y) :- father(X,Y).
ancestor(X,Y) :- father(X,Z), father(Z,Y).