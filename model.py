import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np

class GLSP_model():

    def __init__(self,instance) -> None:
        self.instance = instance
        self.model = ConcreteModel()
        self.model.Name = instance.Name + "_model" 
        
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
        model.B = pyo.Set(initialize = instance.B, dimen = 1, ordered = True) # type: ignore
        model.F = pyo.Set(initialize = instance.F , dimen = 1, ordered = True) # type: ignore
        model.T = pyo.Set(initialize = instance.T , dimen = 1, ordered = True) # type: ignore
        
        model.S = pyo.RangeSet(1, instance.N * len(instance.T)) # type: ignore
        model.S_t = pyo.Set(model.T, initialize=instance.S_t) # type: ignore
        model.SO_t = pyo.Set(model.T,initialize=instance.SO_t)
        model.V_J = pyo.Set(initialize=instance.V_J, ordered = False) # type: ignore
        model.Bl_j = pyo.Set(model.F, initialize=instance.Bl_j,ordered = False) # type: ignore
        model.Bs_j = pyo.Set(model.T, initialize=instance.Bs_j,ordered = False) # type: ignore
        
        # Params 
        model.fi_j = pyo.Param(model.B, initialize={j: instance.fi_j[j-1] for j in model.B}) #type: ignore
        model.p_j = pyo.Param(model.B, initialize={j: instance.p_j[j-1] for j in model.B}) # type: ignore
        #model.f_j = pyo.Param(model.B, initialize={j: instance.f_j[j-1] for j in model.B}) # type: ignore
        model.TCH_j = pyo.Param(model.B, initialize={j: instance.TCH_j[j-1] for j in model.B}) # type: ignore
        model.col_j = pyo.Param(model.B, initialize={j: instance.col_j[j-1] for j in model.B}) # type: ignore
        model.transp_j = pyo.Param(model.B, initialize={j: instance.transp_j[j-1] for j in model.B}) # type: ignore
        model.Nm_l = pyo.Param(model.F, initialize={l: instance.Nm_l[l-1] for l in model.F}) # type: ignore
        model.mind_t = pyo.Param(model.T, initialize={t: instance.mind_t[t-1] for t in model.T}) # type: ignore
        model.maxd_t = pyo.Param(model.T, initialize={t: instance.maxd_t[t-1] for t in model.T}) # type: ignore
        model.vin_t = pyo.Param(model.T, initialize={t: instance.vin_t[t-1] for t in model.T}) # type: ignore
        model.K_t = pyo.Param(model.T, initialize={t: instance.K_t[t-1] for t in model.T}) # type: ignore
        model.st_ij = pyo.Param(model.B, model.B, initialize={(i, j): instance.st_ij[i-1, j-1] for i in model.B for j in model.B}) # type: ignore
        model.dist_ij = pyo.Param(model.B, model.B, initialize={(i, j): instance.dist_ij[i-1, j-1] for i in model.B for j in model.B}) # type: ignore
        model.ATR_jt = pyo.Param(model.B, model.T, initialize={(j, t): instance.ATR_jt[j-1, t-1] for j in model.B for t in model.T}) # type: ignore
        model.bm_lj = pyo.Param(model.F, model.B, initialize={(l, j): instance.bm_lj[l][j-1] for l in model.F for j in model.B}) # type: ignore
        model.Ht = pyo.Param(initialize=instance.Ht) # type: ignore
        model.N_t = pyo.Param(initialize=instance.N_t) # type: ignore
        model.Htt = pyo.Param(initialize=instance.Htt) # type: ignore
        model.Np = pyo.Param(initialize=instance.Np) # type: ignore
        model.mo = pyo.Param(initialize=instance.mo) # type: ignore
        model.bs = pyo.Param(initialize=instance.bs) # type: ignore
        model.md = pyo.Param(initialize=instance.md) # type: ignore
        model.pa = pyo.Param(initialize=instance.pa) # type: ignore

        self.model = model

    def build_vars(self):
        model = self.model

        # Aliases for params
        B, F, T, S, V_J, S_t, SO_t = model.B, model.F, model.T, model.S, model.V_J, model.S_t, model.SO_t
        p_j, Bl_j, Bs_j,  mind_t, maxd_t, vin_t, fi_j, TCH_j, Nm_l, col_j, Ht, N_t, K_t, transp_j, st_ij, dist_ij, bm_lj, Htt, Np, mo, bs, md = model.p_j, model.Bl_j, model.Bs_j,  model.mind_t, model.maxd_t, model.vin_t, model.fi_j, model.TCH_j, model.Nm_l, model.col_j, model.Ht, model.N_t, model.K_t, model.transp_j, model.st_ij, model.dist_ij, model.bm_lj, model.Htt, model.Np, model.mo, model.bs, model.md

        def x_bounds(model,l, j,s):
            return (0, model.p_j[j])
        model.x = pyo.Var(F, B, S, within=NonNegativeReals,bounds=x_bounds) # type: ignore
        
        model.y = pyo.Var(F, B, S, within=Binary) # type: ignore
        model.z = pyo.Var(F, B, B, S, within=Binary) # type: ignore

        def wm_bounds(model, t):
            return (0, model.mind_t[t])
        model.wm = Var(T, bounds=wm_bounds) # type: ignore

        def wb_bounds(model, j):
            return (0, model.p_j[j])
        model.wb = Var(B, bounds=wb_bounds) # type: ignore

        # fi_jxando valores impossiveis de y
        for j in B: # type: ignore
            for t in T: # type: ignore
                for l in F: # type: ignore
                    if not j in set(value for value in Bs_j[t] if value in Bl_j[l]): # type: ignore
                        for s in S_t[t]: # type: ignore
                            model.y[l,j,s].fix(0) # type: ignore

        self.model = model 
    
    def build_model(self):
        model = self.model

        # Aliases
        B, F, T, S, V_J, S_t, SO_t = model.B, model.F, model.T, model.S, model.V_J, model.S_t, model.SO_t
        p_j, Bl_j, Bs_j, mind_t, maxd_t, vin_t, fi_j, TCH_j, Nm_l, col_j, Ht, N_t, K_t, transp_j, st_ij, dist_ij, bm_lj, Htt, Np, mo, bs, md = model.p_j, model.Bl_j, model.Bs_j, model.mind_t, model.maxd_t, model.vin_t, model.fi_j, model.TCH_j, model.Nm_l, model.col_j, model.Ht, model.N_t, model.K_t, model.transp_j, model.st_ij, model.dist_ij, model.bm_lj, model.Htt, model.Np, model.mo, model.bs, model.md
        
        x, y, z, wm, wb = model.x, model.y, model.z, model.wm, model.wb
        pa, ATR_jt = model.pa, model.ATR_jt

        # Objective 

        model.objective = Objective(expr=(mo * sum(wm[t]  for t in T) +bs*sum(wb[j] for j in B) +md*sum(dist_ij[i, j] * z[l, i, j,s] for l in F for i in B for j in B for s in S) - pa*sum(ATR_jt[j,t]*sum(x[l,j,s] for l in F for s in S_t) for j in B for t in T)), sense=minimize)
        print("OBJ OK")

        #2 & 3
        model.minimum_demand_list = ConstraintList()
        model.maximum_demand_list = ConstraintList()
        for t in T:
            model.minimum_demand_list.add(expr=(sum(x[l, j, s] for s in S_t[t] for j in B for l in F) + wm[t] >= mind_t[t]))
            model.maximum_demand_list.add(expr=(sum(x[l, j, s] for s in S_t[t] for j in B for l in F)  <= maxd_t[t]))
        print("2 and 3 OK")

        model.limit_harvest_and_transport_list = ConstraintList()
        # 4
        for j in B:
            model.limit_harvest_and_transport_list.add(expr=(sum(x[l,j,s] for s in S for l in F) + wb[j] == p_j[j]))
        print("4 OK")
        model.minimum_amount_of_vin_tasse_list = ConstraintList()
        #5
        for t in T:
            model.minimum_amount_of_vin_tasse_list.add(expr=(sum((x[l,j,s]/TCH_j[j])*fi_j[j] for s in S_t[t] for l in F for j in V_J) >= vin_t[t]))
        print("5 OK")

        model.limit_harvesting_capacity_list = ConstraintList()
        #6
        for t in T:
            for l in F:
                model.limit_harvesting_capacity_list.add(expr=( sum(((24)/(col_j[j]*Nm_l[l]*Ht))*x[l,j,s] for s in S_t[t] for j in B) + sum( (Nm_l[l]/Np)*st_ij[i,j]*z[l,i,j,s] for s in S_t[t] for j in B for i in B) <= K_t[t]))
        print("6 OK")

        model.limit_transport_capacity_list = ConstraintList()
        #7
        for t in T:
            model.limit_transport_capacity_list.add(expr=(sum((24/(transp_j[j]*N_t*Htt))*x[l,j,s] for s in S_t[t] for j in B  for l in F) <= K_t[t]))
        print("7 OK")

        model.limit_production_to_capacity_list = ConstraintList()
        #8
        for t in T:
            for s in S_t[t]:
                for j in B:
                    for l in F:
                        model.limit_production_to_capacity_list.add(expr=(x[l,j,s]<= min((transp_j[j]*N_t*Htt)/24,(col_j[j]*Nm_l[l]*Ht)/24)*K_t[t]*y[l,j,s]))
        print("8 OK")

        model.floor_of_production_capacity_list = ConstraintList()
        #9
        for s in S:
            if s <= 1:
                continue
            for l in F:
                for j in B:
                    model.floor_of_production_capacity_list.add(expr=(x[l,j,s] >= bm_lj[l,j]*(y[l,j,s] - y[l,j,s-1])))
        print("9 OK")


        model.harvest_all_blocks_list = ConstraintList()
        #10
        for t in T:
            for l in F:
                valid_blocks = set(value for value in Bs_j[t] if value in Bl_j[l])
                if len(valid_blocks) == 0:
                    continue
                for s in S_t[t]:
                    model.harvest_all_blocks_list.add(expr=(sum(y[l,j,s] for j in valid_blocks) == 1))
        print("10 OK")


        model.consistent_movement_on_period_s_minus_list = ConstraintList()
        #11
        for s in S:
            if s > 1:
                for l in F:
                    for i in B:
                            model.consistent_movement_on_period_s_minus_list.add(expr=(sum(z[l,i,j,s] for j in B) == y[l,i,s-1]))
        print("11 OK")

        model.consistent_movement_on_period_s_list = ConstraintList()
        #12
        for s in S:
            for l in F:
                for j in B:
                    model.consistent_movement_on_period_s_list.add(expr=(sum(z[l,i,j,s] for i in B) == y[l,j,s]))
        print("12 OK")

        model.idle_micro_period_list = ConstraintList()
        #13
        for t in T:
            for s in [s for s in S_t[t] if s != min(S_t[t])]:
                for j in B:
                    for l in F:
                        model.idle_micro_period_list.add(expr=(y[l,j,s-1] >= y[l,j,s]))
        print("13 OK")

        self.model = model
    
    def solve(self,TimeLim: float = 600 ,MemLim: float = 13.2):
        
        self.model.write(f'debug_{self.model.Name}.lp',io_options={'symbolic_solver_labels': True})
        solver = pyo.SolverFactory('gurobi_persistent')
        solver.options['TimeLimit'] = TimeLim  
        solver.options['SoftMemLimit'] = MemLim
        solver.set_instance(self.model) 
        results = solver.solve(self.model, tee=True, load_solutions = False,logfile=f'log_gurobi_{self.model.Name}.txt')
        
        if solver._solver_model.Status == 17:  # Memory limit reached (pyomo does not handle this)
            results.solver.termination_condition = TerminationCondition.resourceInterrupt
            results.solver.status = SolverStatus.warning

        self.model.solutions.load_from(results)

        grb = solver._solver_model
        stats = { 
                'objective_value': grb.ObjVal if grb.SolCount > 0 else None,
                'termination_condition': str(results.solver.termination_condition),
                'number_of_variables': self.model.nvariables(),
                'number_of_constraints': self.model.nconstraints(),
                'mip_gap': grb.MIPGap if hasattr(grb, 'MIPGap') else None,
                'runtime': grb.Runtime,
                'best_bound': grb.ObjBound,
                'node_count': results.solver.statistics.get('branch_and_bound', {}).get('nodes', None),}
        
        return (results,stats)