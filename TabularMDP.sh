#!/bin/bash

# Generate Sub-optimality Gap Experiments
python TabularMDP/TabularMDP_Hybrid_UCB.py

# Generate Regret Experiments
python TabularMDP/TabularMDP_Hybrid_UCB-Regret.py

# Plot
python TabularMDP/MDP_draw.py