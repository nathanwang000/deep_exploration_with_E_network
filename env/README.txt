# README

## To get all the plots of tabular setting, simply run the main.py

- There're four function in main.py that generates the plots. 
- Plot 1 compares different gamma_e for the epsilon-greedy algorithm with generalized counters.
- Plot 2 compares the rate of convergence of epsilon-greedy, UCB, and softmax, with or without E-values.
- Plot 3 and 4 shows that E-values can be considered as the measure of missing knowledge. - Plot 5 compares epsilon-greedy (with E-values) with delayed Q-learning.

## All the algorithms are in algorithms.py, including:

- value_iteration

- lll_learning: Implementation of the lll algorithm. The action-selection rule has 3 choices, including epsilon-greedy, UCB, and softmax.

- lll_counter: The extension of algorithms, with E-values served only as generalized counters.

- delayed_Q_learning: Implementation of delayed Q-learning algorithm, served as a baseline.