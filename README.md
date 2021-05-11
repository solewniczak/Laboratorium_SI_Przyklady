# Laboratorium ze Sztucznej Inteligencji - Przykłady 

Repozytorium zawiera przykładowe programy stanowiące integralną część instrukcji do przedmiotu Sztuczna Inteligencja,
prowadzonego na wydziale Elektroniki Telekomunikacji i Informatyki Politechniki Gdańskiej.

W celu uruchomienia przykładów w systemie **GNU/Linux**, należy przygotować odpowiednie środowisko [conda](https://www.anaconda.com/products/individual).
Definicja środowiska znajduje się w pliku `enviroment.yml`.
W celu utworzenia środowiska należy wykonać polecenie:

    conda env create -f environment.yml

Powyższa komenda utworzy nowe środowisko o nazwie `si2021` oraz zainstaluje wszystkie zależności wymagane do
wykonania zadań laboratoryjnych.

Jeżeli powyższa komenda nie zadziała można też utworzyć środowisko krok po kroku,
wykonując następujące polecenia:

    conda create -n si2021 -c conda-forge swi-prolog python gym scikit-learn matplotlib nltk
    conda activate si2021
    pip install py-sudoku pyswip scikit-fuzzy

W systemie **Windows** niestety nie ma możliwości zainstalowania swi-prolog w z poziomu środowiska conda.
Środowisko SWI Prolog należ pobrać osobno ze strony: https://www.swi-prolog.org/download/stable, a następnie zainstalować.
Następnie możemy utworzyć środowisko conda wraz z wymaganymi zależnościami:

    conda create -n si2021 -c conda-forge python=3.6 gym scikit-learn matplotlib nltk
    conda activate si2021
    pip install py-sudoku pyswip scikit-fuzzy