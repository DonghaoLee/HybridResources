#!/bin/bash

# Generate Sub-optimality Gap Experiments
python LinearContextualBandit/Contextual_Hybrid_UCB.py

# Generate Regret Experiments
python LinearContextualBandit/Contextual_Hybrid_UCB-Regret.py

# Plot
python LinearContextualBandit/Contextual_draw.py