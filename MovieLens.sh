#!/bin/bash

# Generate Sub-optimality Gap and Regret Experiments
python MovieLens/MovieLens_Hybrid_UCB.py
python MovieLens/MovieLens_Hybrid_UCB-Regret.py

# Plot Results
python MovieLens/MovieLens_draw.py