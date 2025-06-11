#!/bin/bash

# Generate Offline Datasets
python MountCar/MountCar_Generate_OfflineData.py

# Generate Sub-optimality Gap and Regret Experiments
python MountCar/MountCar_Hybrid_UCB.py
python MountCar/MountCar_Hybrid_UCB-Regret.py

# Plot Results
python MountCar/MountCar_draw.py