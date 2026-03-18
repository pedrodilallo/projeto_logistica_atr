from copy import copy
import os
import re
import solutions
import pandas as pd
import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gurobipy as gp
import networkx as nx 
import plotly.express as px 
import random
import pickle
from pyomo.environ import *
from pyomo.opt import SolverFactory
from gurobipy import GRB
from datetime import datetime
from pyomo.opt import TerminationCondition, SolverStatus
from instance import Instance
from model import GLSP_model

def generate_hierarchy(base_seed: int = 2002):

    inst_150_3F = Instance(name="150B_3F", seed=base_seed)
    inst_150_3F.generate(size_B=150, size_F=3, size_T=10, seed=base_seed)
    print(f"150B_3F: {len(inst_150_3F.B)} blocks, {np.sum(inst_150_3F.p_j):.0f} tons")

    inst_100_3F = inst_150_3F.create_subset(100, "100B_3F", seed=base_seed)
    inst_100_3F.name = "100B_3F"
    print(f"100B: {len(inst_100_3F.B)} blocks, {np.sum(inst_100_3F.p_j):.0f} tons")    

    inst_150_6F = Instance(name="150B_6F", seed=base_seed)
    inst_150_6F.generate(size_B=150, size_F=6, size_T=10, seed=base_seed)
    print(f"150B_3F: {len(inst_150_6F.B)} blocks, {np.sum(inst_150_6F.p_j):.0f} tons")

    inst_100_6F = inst_150_6F.create_subset(100, "100B_6F", seed=base_seed)
    inst_100_6F.name = "100B_3F"
    print(f"100B: {len(inst_100_6F.B)} blocks, {np.sum(inst_100_6F.p_j):.0f} tons")    

    return inst_150_3F, inst_100_3F, inst_150_6F, inst_100_6F


def save_vars_csv(model, instance_name, directory="model_variables_test_size"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{instance_name}_variables.csv")
    
    rows = []
    for var in model.component_objects(pyo.Var, active=True):
        for index in var:
            val = pyo.value(var[index], exception=False)
            if val is None or abs(val) <= 1e-6:
                continue
            idx = list(index) if isinstance(index, tuple) else ([index] if index is not None else [])
            idx += ["-"] * (4 - len(idx))
            rows.append({"variable": var.name, "idx_1": idx[0], "idx_2": idx[1], "idx_3": idx[2], "idx_4": idx[3], "value": val})
    
    pd.DataFrame(rows).to_csv(path, index=False)

i = 0
all_stats = {}

with open('instance_files/instance_150B_6F_150_6_10_20260310231330.pkl', 'rb') as f:
    instance = pickle.load(f)

for theta in [0.1,0.2,0.3]:
    for gamma in [0.25,0.5,0.75]:
        
        inst = copy(instance)
        inst.Name = str(instance.Name) + f"Robust_gamma_{gamma}_theta{theta}".replace('.','d')
        print(inst.Name)

        model = GLSP_model(inst)
        Gamma,ATR_deviation = model.uncertainty(gamma,theta)
        model.robustness(Gamma,ATR_deviation)
        results, stats = model.solve(TimeLim = 3600,logfile = f"logs/{inst.Name}.log")
  
        model.save_variables_to_csv()
        delta = 0
        all_stats[instance.Name] = stats
        print(stats)

all_stats = pd.DataFrame(all_stats)
all_stats.to_csv('logs/full_stats_batch_3_testing_time.csv')
