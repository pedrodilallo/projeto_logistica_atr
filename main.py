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

inst = Instance("Teste1")
inst.generate(30,10,12)

model = GLSP_model(inst)
model.solve()

