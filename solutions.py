import pandas as pd
import hashlib
import numpy as np
import pyomo.environ as pyo
import pandas as pd
from datetime import datetime, timedelta, date
from datetime import datetime, timedelta, date
from instance import Instance


class model_results(): 

    def __init__(self,results,GLSP_model,instance):
        self.results = results
        self.model = GLSP_model
        self.instance = instance

        x = []
        y = []
        z = []
        wm = []
        wb = []

        for var in GLSP_model.component_objects(pyo.Var):
            var_name = var.name

            for index, value in var.get_values().items():
                if value is None or value == 0.0:
                    continue
                
                if var_name == "x":
                    x.append({
                        "l" : index[0],
                        "j" : index[1],
                        "s" : index[2],
                        "Value" : value
                    })
                elif var_name == "y":
                    y.append({
                        "l" : index[0],
                        "j" : index[1],
                        "s" : index[2],
                        "Value" : value
                    })    
                elif var_name == "z":
                    z.append({
                        "l" : index[0],
                        "i" : index[1],
                        "j" : index[2],
                        "s" : index[3],
                        "Value" : value
                    })
                elif var_name == "wm":
                    wm.append({
                        "t" : index[0],
                        "value" : value
                    })
                elif var_name == "wb":
                    wb.append({
                        "j" : index[0],
                        "value" : value
                    })

        self.x = pd.DataFrame(x, columns = ["l","j","s", 'Value'])
        self.y = pd.DataFrame(y, columns = ["l","j","s", 'Value'])
        self.z = pd.DataFrame(z, columns = ["l","i","j","s", 'Value'])
        self.wm = pd.DataFrame(wm, columns = ["t", 'Value'])
        self.wb = pd.DataFrame(wb, columns = ["j", 'Value'])

        def plot_vars(self): 
            pass
