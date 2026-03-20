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
from pympler import asizeof
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

os.makedirs("logs", exist_ok=True)
os.makedirs("instance_files", exist_ok=True)
os.makedirs("model_variables_csv", exist_ok=True)

inst_150_3F, inst_100_3F, inst_150_6F, inst_100_6F = generate_hierarchy()
 
BASE_INSTANCES = [inst_100_3F]

for base_instance in BASE_INSTANCES:
    print(f"Deterministic: {base_instance.Name}  |  {len(base_instance.B)} blocks")
    inst = copy(base_instance)
    inst.Name = f"{base_instance.Name}_deterministic"
 
    model = GLSP_model(inst)
    results, stats = model.solve(TimeLim=400, logfile=f"logs/{inst.Name}")
 
    print(stats)
    model.save_variables_to_csv()
    model.append_to_master_csv(stats, directory='logs', filename='all_results.csv')

for base_instance in BASE_INSTANCES:
    print(f"\n{'='*60}")
    print(f"Robust base: {base_instance.Name}  |  {len(base_instance.B)} blocks")
 
    for theta in [0.1, 0.2, 0.3]:
        for gamma in [0.25, 0.5, 0.75]:
 
            inst = copy(base_instance)
            run_tag = f"gamma_{gamma}_theta_{theta}".replace('.', 'd')
            inst.Name = f"{base_instance.Name}_robust_{run_tag}"
            print(f"\n  Running: {inst.Name}")
 
            model = GLSP_model(inst)

            Gamma, ATR_deviation = model.uncertainty(gamma, theta)
            model.robustness(Gamma, ATR_deviation)
 
            inst.ATR_deviation = ATR_deviation
            inst.Gamma         = Gamma
            inst.gamma_param   = gamma
            inst.theta_param   = theta
            inst.save(directory='instance_files')
 
            results, stats = model.solve(
                TimeLim=3600,
                logfile=f"logs/{inst.Name}")
 
            print(stats)
            model.save_variables_to_csv()
            model.append_to_master_csv(stats, directory='logs', filename='all_results.csv')
 
