import model 
from instance import Instance
import solutions
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gurobipy as gp
from gurobipy import GRB
import networkx as nx 
import plotly.express as px 
import random
import pickle
from datetime import datetime

# Load the pickle file (replace with your actual filename)
filename = "instance_objects/instance_5_3_4_202506241522.pkl"  # Adjust based on your saved file
with open(filename, 'rb') as f:
    inst = pickle.load(f)

# Inspect basic attributes
print(f"Blocks (B): {inst.B}")
print(f"Fronts (F): {inst.F}")
print(f"Periods (T): {inst.T}")
print(f"Production (p_j): {inst.p_j[:5]}...")  # Show first 5 for brevity
print(f"Distance Matrix Shape: {inst.dist_ij.shape}")
print(inst.dist_ij)

# Visualize the distance matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(inst.dist_ij, cmap='hot', interpolation='nearest')
plt.colorbar(label='Distance')
plt.title(f'Distance Matrix Heatmap (Size {len(inst.B)}x{len(inst.B)})')
plt.xlabel('Block Index')
plt.ylabel('Block Index')
plt.show()

# Optional: Visualize a bar chart for production
plt.figure(figsize=(8, 6))
plt.bar(range(len(inst.p_j)), inst.p_j, color='blue')
plt.title('Production per Block')
plt.xlabel('Block Index')
plt.ylabel('Production (tons)')
plt.show()