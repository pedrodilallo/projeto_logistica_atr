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
from model import GLSP_model

params = {'microperiods_per_t': 4}
inst = Instance("Validacao1",kwargs=params)
inst.generate(5,5,12,kwargs=params)  
inst.save()
#inst.visualize_instance()

model = GLSP_model(inst)
#model.model.pprint(filename='Constrantslist_Validacao1.txt')
model.solve()
