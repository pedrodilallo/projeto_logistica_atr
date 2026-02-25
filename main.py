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

    inst_190 = Instance(name="190B", seed=base_seed)
    inst_190.generate(size_B=190, size_F=3, size_T=10, seed=base_seed)
    print(f"190B: {len(inst_190.B)} blocks, {np.sum(inst_190.p_j):.0f} tons")

    inst_140 = inst_190.create_subset(140, "140B", seed=base_seed + 3)
    inst_140.name = "140B"
    print(f"140B: {len(inst_140.B)} blocks, {np.sum(inst_140.p_j):.0f} tons")
    
    inst_90 = inst_140.create_subset(90, "90B", seed=base_seed + 3)
    inst_90.name = "90B"
    print(f"90B: {len(inst_90.B)} blocks, {np.sum(inst_90.p_j):.0f} tons")

    inst_40 = inst_90.create_subset(40, "40B", seed=base_seed + 3)
    inst_40.name = "40B"
    print(f"40B: {len(inst_40.B)} blocks, {np.sum(inst_40.p_j):.0f} tons")    

    return  inst_190,inst_140, inst_90,inst_40


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

instance_list = generate_hierarchy()
i = 0
all_stats = {}
for instance in instance_list:
    instance.save(directory="size_test_instances2")
    print(instance.Name)

    model = GLSP_model(instance)
    results, stats = model.solve(TimeLim = 7200)  
    delta = 0
    all_stats[instance.Name] = stats
    print(stats)

all_stats = pd.DataFrame(all_stats)
all_stats.to_csv('logs/full_stats_batch_1_testing_time.csv')