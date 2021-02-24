not_in_list(_, []).
not_in_list(Elm, [Head|Tail]) :- Elm \= Head, not_in_list(Elm, Tail).

all_different([]).
all_different([Head|Tail]) :- not_in_list(Head, Tail), all_different(Tail).

domain([], _, _).
domain([Head|Tail], Min, Max) :- between(Min, Max, Head), domain(Tail, Min, Max).