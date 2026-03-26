import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition, SolverStatus
from collections import defaultdict
import numpy as np
import os
from datetime import datetime
import time 
class GLSP_model():

    def __init__(self,instance,sparse=False) -> None:
        self.Robust = False
        self.wall_start = time.perf_counter()
        self.instance = instance
        self.model = ConcreteModel()
        self.model.Name = instance.Name + "_model" 

        self.INDEX_SCHEMA = {
            "x":     ("l", "j", "s"),
            "y":     ("l", "j", "s"),
            "z":     ("l", "i", "j", "s"),
            "wm":    ("t",),
            "wb":    ("j",),
            "theta": (),
            "alpha": ("t",),
            "beta":  ("j", "t")}
        
        self._run_stem = None
        self.gamma = None
        self.theta = None
        self.sparse = False
        
        self.build_params()
        print("Params ok!")
        self.build_vars()
        print("Vars ok!")
        self.build_model()
        print("Model was built!")

    def build_params(self) -> None:
        instance = self.instance
        model = self.model

        # Sets 
        model.B = pyo.Set(initialize = instance.B, dimen = 1, ordered = True) 
        model.F = pyo.Set(initialize = instance.F , dimen = 1, ordered = True) 
        model.T = pyo.Set(initialize = instance.T , dimen = 1, ordered = True) 
        
        model.S = pyo.RangeSet(1, instance.N * len(instance.T)) 
        model.S_t = pyo.Set(model.T, initialize=instance.S_t) 
        model.SO_t = pyo.Set(model.T,initialize=instance.SO_t)
            
        model.V_J = pyo.Set(initialize=instance.V_J, ordered = False) 
        model.Bl_j = pyo.Set(model.F, initialize=instance.Bl_j,ordered = False) 
        model.Bs_j = pyo.Set(model.T, initialize=instance.Bs_j,ordered = False) 
        
        # Params 
        model.fi_j = pyo.Param(model.B, initialize={j: instance.fi_j[j-1] for j in model.B}) #type: ignore
        model.p_j = pyo.Param(model.B, initialize={j: instance.p_j[j-1] for j in model.B}) 
        #model.f_j = pyo.Param(model.B, initialize={j: instance.f_j[j-1] for j in model.B}) 
        model.TCH_j = pyo.Param(model.B, initialize={j: instance.TCH_j[j-1] for j in model.B}) 
        model.col_j = pyo.Param(model.B, initialize={j: instance.col_j[j-1] for j in model.B}) 
        model.transp_j = pyo.Param(model.B, initialize={j: instance.transp_j[j-1] for j in model.B}) 
        model.Nm_l = pyo.Param(model.F, initialize={l: instance.Nm_l[l-1] for l in model.F}) 
        model.mind_t = pyo.Param(model.T, initialize={t: instance.mind_t[t-1] for t in model.T}) 
        model.maxd_t = pyo.Param(model.T, initialize={t: instance.maxd_t[t-1] for t in model.T}) 
        model.vin_t = pyo.Param(model.T, initialize={t: instance.vin_t[t-1] for t in model.T}) 
        model.K_t = pyo.Param(model.T, initialize={t: instance.K_t[t-1] for t in model.T}) 
        model.st_ij = pyo.Param(model.B, model.B, initialize={(i, j): instance.st_ij[i-1, j-1] for i in model.B for j in model.B}) 
        model.dist_ij = pyo.Param(model.B, model.B, initialize={(i, j): instance.dist_ij[i-1, j-1] for i in model.B for j in model.B}) 
        model.ATR_jt = pyo.Param(model.B, model.T, initialize={(j, t): instance.ATR_jt[j-1, t-1] for j in model.B for t in model.T}) 
        model.bm_lj = pyo.Param(model.F, model.B, initialize={(l, j): instance.bm_lj[l][j-1] for l in model.F for j in model.B}) 
        model.Ht = pyo.Param(initialize=instance.Ht) 
        model.N_t = pyo.Param(model.T,initialize=instance.N_t) 
        model.Htt = pyo.Param(initialize=instance.Htt) 
        model.Np = pyo.Param(initialize=instance.Np) 
        model.mo = pyo.Param(initialize=instance.mo) 
        model.bs = pyo.Param(initialize=instance.bs) 
        model.md = pyo.Param(initialize=instance.md) 
        model.pa = pyo.Param(initialize=instance.pa) 

        self.model = model

    def build_vars(self,sparse = False):
        model = self.model

        def x_bounds(model,l, j,s):
            return (0, model.p_j[j])
        
        model.x = pyo.Var(model.F, model.B, model.S, within=NonNegativeReals,bounds=x_bounds) 
        
        model.y = pyo.Var(model.F, model.B, model.S, within=Binary) 

        def wm_bounds(model, t):
            return (0, model.mind_t[t])
        model.wm = Var(model.T,within=NonNegativeReals,bounds=wm_bounds) 

        def wb_bounds(model, j):
            return (0, model.p_j[j])
        model.wb = Var(model.B,within=NonNegativeReals,bounds = wb_bounds) 

        # fi_jxando valores impossiveis de y
        fixed_y= set()
        for j in model.B:
            for t in model.T:
                for l in model.F:
                    if not j in set(value for value in model.Bs_j[t] if value in model.Bl_j[l]):
                        for s in model.S_t[t]:
                            model.y[l,j,s].fix(0)
                            fixed_y.add((l, j, s))
        
        if not self.sparse:
            model.z = pyo.Var(model.F,model.B,model.B,model.S, within=Binary)
            self.model = model
            
            return

        # (l,j) onde y[l,j,s] is NOT fixed to 0
        active_lj_s: dict = {}
        for l in model.F: 
            for j in model.B: 
                for s in model.S: 
                    if (l, j, s) not in fixed_y:
                        active_lj_s.setdefault(s, set()).add((l, j))
        
        valid_z = []
        s_min = min(model.S) 
        for s in model.S: 
            active_j_s  = active_lj_s.get(s,   set())   # y(l,j,s) não fixado em s
            active_i_s1 = active_lj_s.get(s-1, set())   # y(l,j,s) não fixado em s-1
            for (l, j) in active_j_s:
                if s == s_min:
                    # No primeiro S, nada eh fixado
                    for i in model.B: 
                        valid_z.append((l, i, j, s))
                else:
                    for (l2, i) in active_i_s1:
                        if l2 == l:
                            valid_z.append((l, i, j, s))
        
        model.VALID_Z = pyo.Set(initialize=set(valid_z), dimen=4, ordered=False)
        # só gera z quando y permite
        model.z = pyo.Var(model.VALID_Z, within=Binary)

        self.model = model 
    
    def build_model(self,sparse=False):
        model = self.model
        
    
        
        #2 & 3
        model.minimum_demand_list = ConstraintList()
        model.maximum_demand_list = ConstraintList()
        for t in model.T:
            model.minimum_demand_list.add(expr=(sum(model.x[l, j, s] for s in model.S_t[t] for j in model.B for l in model.F) + model.wm[t] >= model.mind_t[t]))
            model.maximum_demand_list.add(expr=(sum(model.x[l, j, s] for s in model.S_t[t] for j in model.B for l in model.F)  <= model.maxd_t[t]))

        print("2 and 3 OK")

        model.limit_harvest_and_transport_list = ConstraintList()
        # 4
        for j in model.B:
            model.limit_harvest_and_transport_list.add(expr=(sum(model.x[l,j,s] for s in model.S for l in model.F) + model.wb[j] == model.p_j[j]))
        print("4 OK")

        model.minimum_amount_of_vin_tasse_list = ConstraintList()
        #5
        for t in model.T:
            model.minimum_amount_of_vin_tasse_list.add(expr=(sum((model.x[l,j,s]/model.TCH_j[j])*model.fi_j[j] for s in model.S_t[t] for l in model.F for j in model.V_J) >= model.vin_t[t]))
        print("5 OK")

        model.limit_transport_capacity_list = ConstraintList()
        #7        
        for t in model.T:
            model.limit_transport_capacity_list.add(expr=(sum((24/(model.transp_j[j]*model.N_t[t]*model.Htt))*model.x[l,j,s] for s in model.S_t[t] for j in model.B for l in model.F) <= model.K_t[t]))
        print("7 OK")

        model.limit_production_to_capacity_list = ConstraintList()
        #8
        for t in model.T:
            for s in model.S_t[t]:
                for j in model.B:
                    for l in model.F:
                        model.limit_production_to_capacity_list.add(expr=(model.x[l,j,s] <= min((model.transp_j[j]*model.N_t[t]*model.Htt)/24,(model.col_j[j]*model.Nm_l[l]*model.Ht)/24)*model.K_t[t]*model.y[l,j,s]))

        print("8 OK")

        model.floor_of_production_capacity_list = ConstraintList()
        #9
        for s in model.S:
            if s <= 1:
                continue
            for l in model.F:
                for j in model.B:
                    model.floor_of_production_capacity_list.add(expr=(model.x[l,j,s] >= model.bm_lj[l,j]*(model.y[l,j,s] - model.y[l,j,s-1])))

        print("9 OK")


        model.harvest_all_blocks_list = ConstraintList()
        #10
        for t in model.T:
            for l in model.F:
                valid_blocks = set(value for value in model.Bs_j[t] if value in model.Bl_j[l])
                if len(valid_blocks) == 0:
                    continue
                for s in model.S_t[t]:
                    model.harvest_all_blocks_list.add(expr=(sum(model.y[l,j,s] for j in valid_blocks) == 1))
        print("10 OK")



        if self.sparse:
            model.objective = Objective(expr=(model.pa*sum(model.ATR_jt[j,t]*sum(model.x[l,j,s] for l in model.F for s in model.S_t[t]) for j in model.B for t in model.T) - (model.mo * sum(model.wm[t]  for t in model.T) + model.bs*sum(model.wb[j] for j in model.B) + model.md*sum(model.dist_ij[i, j] * model.z[l, i, j, s] for (l, i, j, s) in model.VALID_Z))), sense=maximize)
            print("OBJ OK")


            model.consistent_movement_on_period_s_minus_list = ConstraintList()
            z_by_lt= defaultdict(list)   # (l, t) -> [(i, j, s), ...]
            s_to_t = {s: t for t in model.T for s in model.S_t[t]}
            for (l, i, j, s) in model.VALID_Z:
                z_by_lt[(l, s_to_t[s])].append((i, j, s))

            #6 - Só para z válido
            for t in model.T:
                for l in model.F:
                    z_entries = z_by_lt.get((l, t), [])
                    model.limit_harvesting_capacity_list.add(expr=(sum(((24)/(model.col_j[j]*model.Nm_l[l]*model.Ht))*model.x[l,j,s] for s in model.S_t[t] for j in model.B) + sum((model.Nm_l[l]/model.Np)*model.st_ij[i,j]*model.z[l,i,j,s] for (i, j, s) in z_entries) <= model.K_t[t]))
            print("6 OK")
                    
            z_by_lis = defaultdict(list)   # (l,i,s) -> [j, ...]
            for (l, i, j, s) in model.VALID_Z: 
                z_by_lis[(l, i, s)].append(j)

            #11 - só gera para z válidos
            for s in model.S:
                if s > 1:
                    for l in model.F:
                        for i in model.B:
                            js = z_by_lis.get((l, i, s), [])
                            model.consistent_movement_on_period_s_minus_list.add(expr=(sum(model.z[l,i,j,s] for j in js) == model.y[l,i,s-1]))
            print("11 OK")

            model.consistent_movement_on_period_s_list = ConstraintList()
            z_by_ljs = defaultdict(list)   # (l,j,s) -> [i, ...]
            for (l, i, j, s) in model.VALID_Z: 
                z_by_ljs[(l, j, s)].append(i)

            #12 - só para z válido
            for s in model.S:
                for l in model.F:
                    for j in model.B:
                        i_list = z_by_ljs.get((l, j, s), [])
                        model.consistent_movement_on_period_s_list.add(expr=(sum(model.z[l,i,j,s] for i in i_list) == model.y[l,j,s]))
            print("12 OK")

            model.idle_micro_period_list = ConstraintList()
            #13
            for t in model.T:
                for s in [microperiod for microperiod in model.S_t[t] if microperiod != model.SO_t[t].at(1)]:
                    for j in model.B:
                        for l in model.F:
                            model.idle_micro_period_list.add(expr=(model.y[l,j,s-1] >= model.y[l,j,s]))

            print("13 OK")

            self.model = model
        else:

            # Objective 
            model.objective = Objective(expr=(model.pa*sum(model.ATR_jt[j,t]*sum(model.x[l,j,s] for l in model.F for s in model.S_t[t]) for j in model.B for t in model.T) - (model.mo * sum(model.wm[t]  for t in model.T) + model.bs*sum(model.wb[j] for j in model.B) + model.md*sum(model.dist_ij[i, j] * model.z[l, i, j,s] for l in model.F for i in model.B for j in model.B for s in model.S))), sense=maximize)
            print("OBJ OK")

            model.limit_harvesting_capacity_list = ConstraintList()
            for t in model.T:
                for l in model.F:
                    model.limit_harvesting_capacity_list.add(expr=(sum(((24)/(model.col_j[j]*model.Nm_l[l]*model.Ht))*model.x[l,j,s] for s in model.S_t[t] for j in model.B) + sum((model.Nm_l[l]/model.Np)*model.st_ij[i,j]*model.z[l,i,j,s] for s in model.S_t[t] for j in model.B for i in model.B) <= model.K_t[t]))
            print("6 OK")

            model.consistent_movement_on_period_s_minus_list = ConstraintList()

            #1
            for s in model.S:
                if s > 1:
                    for l in model.F:
                        for i in model.B:
                            model.consistent_movement_on_period_s_minus_list.add(expr=(sum(model.z[l,i,j,s] for j in model.B) == model.y[l,i,s-1]))
            print("11 OK")

            model.consistent_movement_on_period_s_list = ConstraintList()
            #12
            for s in model.S:
                for l in model.F:
                    for j in model.B:
                        model.consistent_movement_on_period_s_list.add(expr=(sum(model.z[l,i,j,s] for i in model.B) == model.y[l,j,s]))

            print("12 OK")
            self.model = model

    def uncertainty(self,gamma: float):
        self.gamma = gamma
        self.theta = (0.147 - 0.095)

        Gamma = [np.ceil(gamma*len(self.model.Bs_j[t])) for t  in self.model.T] 
        ATR_deviation = np.zeros((len(self.model.B),len(self.model.T)))
        for j in self.model.B: 
            for t in self.model.T: 
                ATR_deviation[j-1,t-1] = round(0.147 - 0.095,3)
        
        return Gamma,ATR_deviation
    
    def robustness(self,Gamma: list[int],uncertainty: np.array): 
        model = self.model

        model.ATR_deviation = Param(model.B, model.T, initialize={(j, t): uncertainty[j-1, t-1] for j in model.B for t in model.T})
        ATR_deviation = model.ATR_deviation

        B, F, T, S, V_J, S_t, SO_t = model.B, model.F, model.T, model.S, model.V_J, model.S_t, model.SO_t
        p_j, Bl_j, Bs_j, mind_t, maxd_t, vin_t, fi_j, TCH_j, Nm_l, col_j, Ht, N_t, K_t, transp_j, st_ij, dist_ij, bm_lj, Htt, Np, mo, bs, md = model.p_j, model.Bl_j, model.Bs_j, model.mind_t, model.maxd_t, model.vin_t, model.fi_j, model.TCH_j, model.Nm_l, model.col_j, model.Ht, model.N_t, model.K_t, model.transp_j, model.st_ij, model.dist_ij, model.bm_lj, model.Htt, model.Np, model.mo, model.bs, model.md
        
        x, y, z, wm, wb = model.x, model.y, model.z, model.wm, model.wb
        pa, ATR_jt = model.pa, model.ATR_jt

        # New vars for robustness 
        model.theta = pyo.Var( within=NonNegativeReals )
        model.alpha = pyo.Var(model.T, within=NonNegativeReals )
        model.beta = pyo.Var(model.B, model.T, within=NonNegativeReals )
        theta,alpha,beta = model.theta, model.alpha, model.beta

        model.del_component(model.objective)
        if not self.sparse: 
            model.objective = Objective(expr=  theta - (\
                mo*sum(wm[t] for t in T) + \
                bs*sum(wb[j] for j in B) + \
                md*sum(dist_ij[i, j] * z[l, i, j,s] for l in F for i in B for j in B for t in T for s in S_t[t])), sense=maximize)
        else: 
            model.objective = Objective(expr=  theta - (\
                mo*sum(wm[t] for t in T) + \
                bs*sum(wb[j] for j in B) + \
                md*sum(dist_ij[i, j] * z[l, i, j,s] for (l, i, j, s) in model.VALID_Z)), sense=maximize)
        model.obj_revenue = ConstraintList()

        if max(Gamma) == 0:
            raise ValueError ('LOAD DETERMINISC MODEL')

        elif max(Gamma) > 0:
            # ROBUST
            print("ROBUST")
            model.obj_revenue.add(expr=( theta <= pa*( \
                sum(self.model.ATR_jt[j,t] * sum(x[l,j,s] for l in F for s in S_t[t]) for j in B for t in T) - \
                sum(beta[j,t] for j in B for t in T) - \
                sum(Gamma[t-1]*alpha[t] for t in T) ) ) ) # no need make a constraint saying theta >= 0, because the bound was established on variable definition. 
            
            for j in B:
                for t in T:
                    model.obj_revenue.add(expr=( alpha[t] + beta[j,t] >= ATR_deviation[j,t] * sum(x[l,j,s] for l in F for s in S_t[t]) ) ) # LB for both defined at variable definition
        self.Robust = True
        self.model = model


    def _var_rows(self,var_obj):
        schema = self.INDEX_SCHEMA.get(var_obj.name, ())
        
        def row(idx, var_data):
            idx_t = () if idx is None else (idx,) if not isinstance(idx, tuple) else idx
            r = {k: "no_index" for k in ("i", "j", "l", "s", "t")}
            r.update(zip(schema, idx_t))
            return {"variable_name": var_obj.name, **r, "value": pyo.value(var_data, exception=False)}
        
        return [row(idx, vd) for idx, vd in var_obj.items()]

    def save_variables_to_csv(self,save_dir='model_variables_csv'):
        os.makedirs(f"{save_dir}", exist_ok=True)
        
        stem = (os.path.basename(self._run_stem)
                if self._run_stem else self.instance.Name)
        
        path = os.path.join(f'{save_dir}', stem + ".csv")

        rows = [r for v in self.model.component_objects(pyo.Var, active=True) for r in self._var_rows(v)]
        pd.DataFrame(rows, columns=["variable_name", "i", "j", "l", "s", "t", "value"]).to_csv(path, index=False)

        print(f"Variables saved → {path}")
    
    def solve(self,TimeLim: float = 3600 ,MemLim: float = 12,log: bool = True, logfile : str = 'logs/Unnamed'):
        
        date_str  = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        run_stem  = f"{logfile}_{date_str}"
        self._run_stem = run_stem
        
        solver = pyo.SolverFactory('gurobi_persistent')
        solver.options['TimeLimit'] = TimeLim  
        solver.options['SoftMemLimit'] = MemLim
        solver.options['LogFile'] = run_stem + ".log"
        solver.options['MIPGap'] = 0.005
        solver.options['OptimalityTol'] = 0.005
        solver.set_instance(self.model) 

        results = solver.solve(self.model, tee=True, load_solutions = False)

        if solver._solver_model.Status == 17:  # Memory limit reached (pyomo does not handle this alone)
            results.solver.termination_condition = TerminationCondition.resourceInterrupt
            results.solver.status = SolverStatus.warning

        self.model.solutions.load_from(results)

        grb = solver._solver_model
        stats = {
            'instance_name':          self.instance.Name,
            'theta': 0,
            'gamma': 0,
            'objective_value':        grb.ObjVal if grb.SolCount > 0 else None,
            'termination_condition':  str(results.solver.termination_condition),
            'number_of_variables':    self.model.nvariables(),
            'number_of_constraints':  self.model.nconstraints(),
            'mip_gap':                grb.MIPGap if hasattr(grb, 'MIPGap') else None,
            'solver_time_s':          grb.Runtime,
            'total_wall_time_s':      round(time.perf_counter() - self.wall_start, 2),
            'best_bound':             grb.ObjBound,
            'node_count':             results.solver.statistics.get(
                                          'branch_and_bound', {}).get('nodes', None),}

        if self.Robust:
            stats['theta'] = self.theta
            stats['gamma'] = self.gamma        
        

        milling_loss = sum(pyo.value(self.model.wm[t], exception=False) or 0.0 for t in self.model.T)
 
        standover = sum(pyo.value(self.model.wb[j], exception=False) or 0.0 for j in self.model.B)
        
        if self.sparse: 
            total_distance = sum( pyo.value(self.model.dist_ij[i, j]) * (pyo.value(self.model.z[l, i, j, s], exception=False) or 0.0) for (l, i, j, s) in self.model.VALID_Z)
        else: 
            total_distance = sum( pyo.value(self.model.dist_ij[i, j]) * (pyo.value(self.model.z[l, i, j, s], exception=False) or 0.0) for l in self.model.F for i in self.model.B  for j in self.model.B for s in self.model.S )
        
        revenue = pyo.value(self.model.pa) * sum( pyo.value(self.model.ATR_jt[j, t]) * sum(pyo.value(self.model.x[l, j, s], exception=False) or 0.0 for l in self.model.F for s in self.model.S_t[t]) for j in self.model.B for t in self.model.T)
 
        stats['milling_loss_t'] =      milling_loss,
        stats['standover_t'] =         standover,
        stats['total_distance_km'] =   total_distance,
        stats['atr_revenue'] =         revenue

        return (results,stats)
    
    def append_to_master_csv(self, stats: dict, directory: str = "logs", filename: str = "all_results.csv"):

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        df  = pd.DataFrame([stats])
 
        if not os.path.exists(path):
            df.to_csv(path, mode="w", header=True, index=False)
            print(f"Master results file created: {path}")
        else:
            df.to_csv(path, mode="a", header=False, index=False)
            print(f"Master results file updated: {path}")
