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

size_B = 6 # number of blocks
size_F = 3 # number of fronts
size_T = 4 # number o macro-periods
N = 3 # number of subperiods per period
V_j = {1,2,3,4,5,6}
V_j = sorted(V_j)  # Convert to a sorted list
Blj = {1: [1,2,3,4,5,6],2:[1,2,3,4,5,6],3:[1,2,3,4,5,6]} # which front can harvest each block
Bsj = {1: [1,2,3,4],2:[1,2,3,4], 3:[2,3,4,5,6], 4:[3,4,5,6]} # blocks that can be harvested on each period
p = [1000,1500, 5000, 4000, 2000, 400]
mind = [2000 ,2000,2000,2000]
maxd = [5000 ,3000,4000,3000]
vin = [15/3, 15/3, 30/3, 20/3]
fi = [0.9,1,0.95,0.9,0.8,1]
TCH = [ 100,80,100,70,60,60] # 80 ton/hec na media
Nm =  [30,20,10]
col = [59,79,100,80,90,1]
Ht = 10
Nt = {1: 18,2:10,3:10,4:12}
Htt = 10
K = [36*8*3*8,  36*8*3*8, 36*8*3*8, 36*8*3*8]
transp =  [1.5*80,1.5*60,1.5*90,1.5*85,1.5*100,1.5*1]
dist =np.array([[0, 5, 10, 15, 20, 25],
                   [5, 0, 6, 11, 16, 21],
                   [10, 6, 0, 7, 12, 17],
                   [15, 11, 7, 0, 8, 13],
                   [20, 16, 12, 8, 0, 9],
                   [25, 21, 17, 13, 9, 0]])

st = np.array([[0, 0.125, 0.25, 0.375, 0.5, 0.625],
                   [0.125, 0, 0.15, 0.275, 0.4, 0.525],
                   [0.25, 0.15, 0, 0.175, 0.3, 0.425],
                   [0.375, 0.275, 0.175, 0, 0.2, 0.325],
                   [0.5, 0.4, 0.3, 0.2, 0, 0.225],
                   [0.625, 0.525, 0.425, 0.325, 0.225, 0]])
Np = 30
bm = {
    1: [20*25, 15*25, 10*25, 25*25, 18*25, 1000],  # Frente 1 (mecanizada)
    2: [20*25, 15*25, 10*25, 25*25, 18*25, 1000],  # Frente 2 (mecanizada)
    3: [ 100,   100,   100,   100,   100,    10]         # Frente 3 (manual)
}

ATR = np.array([[11.83011756,  9.9155148,  10.40389997, 10.93808328],
                   [10.86479062,  9.52569484, 10.87596935,  9.86554319],
                   [ 9.26075064,  9.61274446, 11.71629121,  9.8440108 ],
                   [ 9.91329545,  9.52180207, 10.86937126,  9.09463748],
                   [ 9.46793727, 10.92704872, 10.46158836, 10.11583283],
                   [ 9.64246571, 10.80007485,  9.72660241, 10.46389029]])

mo = 100_000
bs = 10_000
md = 5_000

params = {
'B': [1,2,3,4,5,6],
'F': [1,2,3],
'T': [1,2,3,4],
'p_j':  p,
'fi_j':  fi,
'TCH_j':  TCH,
'col_j':  col,
'transp_j':  transp,
'Nm_l':  Nm,
'mind_t': mind ,
'maxd_t':  maxd,
'vin_t':  vin,
'K_t':  K,
'V_J': V_j ,
'Bl_j':  Blj,
'Bs_j':  Bsj,
'st_ij':  st,
'dist_ij':  dist,
'Ht':  Ht,
'N_t':  Nt,
'Htt':  Htt,
'Np':  Np,
'mo':  mo,
'bs':  bs,
'md':  md,
'bm_lj':  bm,
'ATR_jt': ATR,
'pa': md,
'N': 3
}

def extract_instance_id(instance_name):
    match = re.search(r'(Factivel|Robusta)(\d+)', instance_name)
    return int(match.group(2)) if match else None


files = [f for f in os.listdir('instance_objects') if os.path.isfile(os.path.join('instance_objects', f))]

n = 1
instances = ['instance_Factivel21_B30_F3_T9_30_3_9_202507081614.pkl',
       'instance_Factivel42_B30_F4_T9_30_4_9_202507081636.pkl',
       'instance_Factivel63_B30_F5_T9_30_5_9_202507081703.pkl']

# Initialize DataFrame to store all stats
all_stats = []

for file in instances:
    instance_id = extract_instance_id(file)

    with open(f"instance_objects/{file}", 'rb') as f:
        instancia = pickle.load(f)

#    original_TCH_j = np.array(instancia.TCH_j, dtype=float)
#    original_transp_j = np.array(instancia.transp_j, dtype=float)

    # Iterate over scaling factors
    for delta in [0.1,0.2,0.3]:
        #transp_scale = transp_scale/100
        # Modify parameters
        #instancia.transp_j = original_transp_j * transp_scale
        
        # Solve the model
        model = GLSP_model(instancia)
        Gamma,ATR_deviation = model.uncertainty(-1,delta)
        model.robustness(Gamma,ATR_deviation)
        results, stats = model.solve()
        
        stats.update({
            'instance_id': instance_id,
            'instance': file,
            'category': 'Soyster',
            'gamma': -1,
            'delta': delta,
            'timestamp': datetime.now().strftime("%Y%m%d%H%M")
        })
        
        all_stats.append(stats)
        
#        instancia.TCH_j = original_TCH_j.copy()
#        instancia.transp_j = original_transp_j.copy()

stats_df = pd.DataFrame(all_stats)
stats_df.to_csv('soyster.csv', index=False)

