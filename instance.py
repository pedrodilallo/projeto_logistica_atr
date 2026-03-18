import os 
from typing import Dict, List, Set, Optional, Any
import random
from scipy.stats import poisson,truncnorm
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class Instance:
    name: str
    B: Optional[List[Any]]  # List of harvest block indices
    F: Optional[List[Any]]  # List of harvest front indices
    T: Optional[List[Any]]  # List of macroperiod indices
    p_j: Optional[List[float]]  # List of estimated productions, len == len(B)
    fi_j: Optional[List[float]]  # List of irrigable fractions, len == len(B)
    TCH_j: Optional[List[float]]  # List of productivities, len == len(B)
    col_j: Optional[List[float]]  # List of harvest capacities, len == len(B)
    transp_j: Optional[List[float]]  # List of transport capacities, len == len(B)
    Nm_l: Optional[List[int]]  # List of machine counts, len == len(F)
    mind_t: Optional[List[float]]  # List of min demands, len == len(T)
    maxd_t: Optional[List[float]]  # List of max demands, len == len(T)
    vin_t: Optional[List[float]]  # List of min areas, len == len(T)
    K_t: Optional[List[float]]  # List of fleet availabilities, len == len(T)
    V_J: Optional[Set[Any]]  # Set of irrigable block indices, subset of B
    Bl_j: Optional[Dict[Any, Set[Any]]]  # Dict with keys in F, values are sets of indices from B
    Bs_j: Optional[Dict[Any, Set[Any]]]  # Dict with keys in T, values are sets of indices from B
    st_ij: Optional[np.ndarray]  # Travel times, shape (len(B), len(B))
    dist_ij: Optional[np.ndarray]  # Distances, shape (len(B), len(B))
    ATR_jt: Optional[np.ndarray]  # Sucrose per ton, shape (len(B), len(T))
    Ht: Optional[float]  # Machine hours per day
    N_t: Optional[Dict[int,int]]  # Number of trucks
    Htt: Optional[float]  # Truck hours per day
    Np: Optional[int]  # Number of platform vehicles
    mo: Optional[float]  # Opportunity cost of unground cane ton when mind_t not met
    bs: Optional[float]  # Opportunity cost of unharvested cane ton
    md: Optional[float]  # Unit cost of distance traveled by front between blocks
    pa: Optional[float]  # Revenue from selling one ton of ATR
    bm_lj: Optional[Dict[Any, List[float]]]  # Dict with keys in F, values are lists len == len(B)
    N: Optional[int] = None
    S_t: Optional[Dict[int, List[int]]] = None
    SO_t: Optional[Dict[int, List[int]]] = None

    def __init__(self, name: str, **kwargs):
        self.Name = name
        self.N = kwargs.get('N', 5)
        self.seed = kwargs.get('seed', 2002)
        self.rng = np.random.default_rng(self.seed)
        
        # Hierarchical tracking for subdividing instances
        self.parent_instance: Optional['Instance'] = None
        self.block_mapping: Optional[Dict[int, int]] = None
        self.is_subset = False
        
        try: 
            self.S_t = {t: [s + self.N * (t - 1) for s in range(1, self.N + 1)] for t in self.T} # type: ignore
            self.SO_t = {t: [self.S_t[t][0]] for t in self.T} # type: ignore
        except:
            self.S_t = None
            self.SO_t =  None

    def generate(self, size_B: int, size_F: int, size_T: int, **kwargs) -> 'Instance':
        seed = kwargs.get('seed', self.seed)
        self.rng = np.random.default_rng(seed)
        
        self.B = list(range(1, size_B + 1))
        self.F = list(range(1, size_F + 1))
        self.T = list(range(1, size_T + 1))
        
        total_S = size_T * self.N
        self.S = list(range(1, total_S + 1))

        # minimum and maximum durations for time windows
        window_min = kwargs.get('window_min', 2)
        window_max = kwargs.get('window_max', 8)
        window_mode = kwargs.get('window_mode', 3.8)

        # No block opens after last period (start + window - 1 <= size_T)
        harvest_window_j = {}
        for j in self.B:
            raw = self.rng.triangular(window_min, window_mode, window_max)
            window_length_j = int(np.ceil(raw))
            window_length_j = np.clip(window_length_j, window_min, window_max)

            # start in [1, size_T - window_length_j + 1] guarantees end <= size_T
            max_start = size_T - window_length_j + 1
            start_j = int(self.rng.integers(1, max_start + 1))
            end_j = start_j + window_length_j - 1

            harvest_window_j[j] = (start_j, end_j)

        self.Bs_j = {t: {j for j in self.B if harvest_window_j[j][0] <= t <= harvest_window_j[j][1]}
        for t in self.T}

        # Microperiods
        self.S_t = {t: list(range((t - 1) * self.N + 1, t * self.N + 1)) for t in self.T}
        self.SO_t = {t: [self.S_t[t][0]] for t in self.T}
    
        # Production parameters
        p_raw = self.rng.triangular(
            kwargs.get('p_j_min', 4000),
            kwargs.get('p_j_mode', 8000),
            kwargs.get('p_j_max', 16000),
            size=size_B)
        
        # this is made so the total is adjusted to a realistic amount of tons based on expert feedback.
        target_total = kwargs.get('p_j_total', 1_500_000)
        self.p_j = np.ceil(p_raw * (target_total / p_raw.sum()))

        self.TCH_j = self.rng.triangular(
            kwargs.get('TCH_j_min', 80),
            kwargs.get('TCH_j_mode', 100),
            kwargs.get('TCH_j_max', 140),
            size_B)

        self.col_j = self.rng.triangular(
            kwargs.get('col_j_min', 49.7),
            kwargs.get('col_j_mode', 56.4),
            kwargs.get('col_j_max', 59.7),
            size=size_B).tolist()

        self.fi_j = [1 for _ in range(size_B)]

        self.block_areas = self.p_j / self.TCH_j        

        # Coordinates and distances
        coord_min = kwargs.get('coord_min', 0.0)
        coord_max = kwargs.get('coord_max', 100.0)
        self.coords = self.rng.uniform(coord_min, coord_max, (size_B, 2))

        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.dist_ij = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(self.dist_ij, 0)

        speed = kwargs.get('speed', 50.0)
        self.st_ij = self.dist_ij / speed
        
        # Calculate distances to mill (at origin)
        mill_coord = np.array([50.0, 50.0])
        dist_to_mill = np.sqrt(np.sum((self.coords - mill_coord) ** 2, axis=1))
        dist_to_mill = np.maximum(dist_to_mill, 0.1)
        truck_capacity = kwargs.get('truck_capacity', 60.0)
        round_trip_factor = kwargs.get('round_trip_factor', 2.0)
        self.transp_j = (truck_capacity * speed) / (round_trip_factor * dist_to_mill)
        
        # Demands
        mind_t_val = (np.sum(self.p_j) / size_T) * 0.70 # FIXED
        self.mind_t = [float(mind_t_val)] * size_T
        self.maxd_t = [float(mind_t_val * 1.30)] * size_T

        # vin_t
        self.vin_t = np.full(size_T, 0.2 * np.mean(self.p_j / self.TCH_j) * 0.10).tolist()
        
        # ATR
        self.ATR_jt = np.zeros((size_B, size_T))
        atr_mean = kwargs.get('ATR_jt_mean', 0.147)
        atr_spread_down = kwargs.get('ATR_jt_spread_down', 0.052)
        atr_spread_up = kwargs.get('ATR_jt_spread_up', 0.016)
        for j in self.B:
            start_j, end_j = harvest_window_j[j]
            for t in self.T:
                if start_j <= t <= end_j:
                    self.ATR_jt[j - 1, t - 1] = self.rng.triangular(
                        atr_mean - atr_spread_down,
                        atr_mean,
                        atr_mean + atr_spread_up)

        Ht = kwargs.get('Ht', 10)
        self.K_t = [float(20 * Ht) for _ in self.T] # 20 operating days per period                 

        mean_K = float(np.mean(self.K_t))

        self.bm_lj = {
            l: [float(self.rng.uniform(
                kwargs.get('bm_lj_alpha_min', 0.05),
                kwargs.get('bm_lj_alpha_max', 0.10)) * 
                self.col_j[j - 1] * mean_K) for j in self.B] for l in self.F}
                
            
        self.Nm_l = [5 for _ in range(size_F)]
        self.V_J = {j for j, fi in zip(self.B, self.fi_j) if fi > 0}
        self.Bl_j = {l: set(self.rng.choice(self.B,size=int(self.rng.integers(int(np.ceil(size_B * 0.5)), size_B + 1)),replace=False).tolist()) for l in self.F} 

        self.Ht = Ht
        self.N_t = kwargs.get('N_t', {x: 100 for x in range(1, size_T + 1)})
        self.Htt = kwargs.get('Htt', 10)
        self.Np = kwargs.get('Np', 10)
        self.mo = kwargs.get('mo', 311) # per ton of cane
        self.bs = kwargs.get('bs', 168) # per ton of cane
        self.md = kwargs.get('md', 25) # per km
        self.pa = kwargs.get('pa', 2112) #R$/ton of ATR, not CANE

        return self
            
    def create_subset(self, num_blocks: int, subset_name: str, seed: Optional[int] = None) -> 'Instance':
        """Create random subset instance."""
        if num_blocks >= len(self.B):
            raise ValueError(f"num_blocks ({num_blocks}) must be less than parent blocks ({len(self.B)})")
        
        if seed is None:
            seed = self.seed + len(self.B) - num_blocks
        
        rng = np.random.default_rng(seed)
        selected_indices = sorted(rng.choice(len(self.B), size=num_blocks, replace=False))
        block_mapping = {new_idx + 1: old_idx + 1 for new_idx, old_idx in enumerate(selected_indices)}
        
        subset = Instance(name=subset_name, seed=seed, N=self.N)
        subset.is_subset = True
        subset.parent_instance = self
        subset.block_mapping = block_mapping
        
        num_F = len(self.F)
        num_T = len(self.T)
        subset.B = list(range(1, num_blocks + 1))
        subset.F = self.F.copy()
        subset.T = self.T.copy()
        subset.S = list(range(1, len(self.T) * subset.N + 1))
        
        # Subset data
        subset.p_j = self.p_j[selected_indices].copy()
        subset.fi_j = [self.fi_j[i] for i in selected_indices]
        subset.TCH_j = self.TCH_j[selected_indices].copy()
        subset.col_j = [self.col_j[i] for i in selected_indices]
        subset.transp_j = [self.transp_j[i] for i in selected_indices]
        subset.block_areas = self.block_areas[selected_indices].copy()
        
        subset.dist_ij = self.dist_ij[np.ix_(selected_indices, selected_indices)].copy()
        subset.st_ij = self.st_ij[np.ix_(selected_indices, selected_indices)].copy()
        subset.coords = self.coords[selected_indices, :].copy()
        subset.ATR_jt = self.ATR_jt[selected_indices, :].copy()
        
        subset.mind_t = self.mind_t.copy()
        subset.maxd_t = self.maxd_t.copy()
        subset.vin_t = np.full(num_T, 0.2 * (sum(subset.TCH_j / subset.p_j)) / num_T if num_T > 0 else 0).tolist()
        
        # Copy period data
        subset.K_t = self.K_t.copy()
        subset.N_t = self.N_t.copy()
        
        subset.Bs_j = {
            t: {new_idx + 1 
                for new_idx, old_idx in enumerate(selected_indices) 
                if old_idx + 1 in self.Bs_j[t]}
            for t in subset.T}
        
        subset.S_t = {t: list(range((t-1)*subset.N + 1, t*subset.N + 1)) for t in subset.T}
        subset.SO_t = {t: [subset.S_t[t][0]] for t in subset.T}
        
        subset.bm_lj = {l: [self.bm_lj[l][i] for i in selected_indices] for l in subset.F}
        subset.Nm_l = self.Nm_l.copy()
        subset.Bl_j = {l: set(subset.B) for l in subset.F}
        subset.V_J = {j for j, fi in zip(subset.B, subset.fi_j) if fi > 0}
        
        # Copy parameters
        subset.Ht = self.Ht
        subset.Htt = self.Htt
        subset.Np = self.Np
        subset.mo = self.mo
        subset.bs = self.bs
        subset.md = self.md
        subset.pa = self.pa
        
        return subset        

    def save(self, directory: str = 'instances'):
        os.makedirs(directory, exist_ok=True)
        sizes = f"{len(self.B)}_{len(self.F)}_{len(self.T)}"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{directory}/instance_{self.Name}_{sizes}_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return filename
        
    def visualize_instance(self): 
        print(f"Blocks (B): {self.B}")
        print(f"Fronts (F): {self.F}")
        print(f"Periods (T): {self.T}")
        print(f"Distance Matrix Shape: {self.dist_ij.shape}")

        sns.set(style='whitegrid', font_scale=1.2)

        fig, ax = plt.subplots(figsize=(10, 8))
        X = self.coords[:, 0]
        Y = self.coords[:, 1]

        ax.scatter(X, Y, c=self.transp_j, s = [p * 2 for p in self.p_j], cmap='coolwarm', label='Harvest Blocks')
        ax.scatter(0, 0, c='black', marker='s', s=200, label='Mill')

        # Colorbar for transport capacity
        scatter = ax.scatter(X, Y, c=self.transp_j, cmap='coolwarm')  
        fig.colorbar(scatter, ax=ax, label='Transport Capacity (units)')

        ax.set_title('Harvest Blocks with Arrows to Mill')
        ax.set_xlabel('X Coordinate (km)')
        ax.set_ylabel('Y Coordinate (km)')
        ax.legend()
        ax.axis('equal')
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.dist_ij, cmap='RdYlGn_r', cbar_kws={'label': 'Distance (km)'})
        plt.title('Distance Matrix Heatmap (Higher = Worse)')
        plt.xlabel('Block Index')
        plt.ylabel('Block Index')
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(14/1.4, 14/1.4))

        # Production Bar Chart
        sns.barplot(x=self.B, y=self.p_j, ax=axes[0, 0], color='blue')
        axes[0, 0].set_title('Estimated Production')
        axes[0, 0].set_ylabel('Tons')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Productivity Bar Chart
        sns.barplot(x=self.B, y=self.TCH_j, ax=axes[0, 1], color='green')
        axes[0, 1].set_title('Productivity')
        axes[0, 1].set_ylabel('TCH')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Irrigable Fraction Bar Chart
        sns.barplot(x=self.B, y=self.fi_j, ax=axes[1, 0], color='orange')
        axes[1, 0].set_title('Irrigable Fraction')
        axes[1, 0].set_ylabel('Fraction')
        axes[1, 0].set_xlabel('Block Index')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Sucrose Content (ATR_jt) Heatmap
        ATR_nonzero = self.ATR_jt[self.ATR_jt > 0]
        vmin = ATR_nonzero.min() if ATR_nonzero.size > 0 else 0
        sns.heatmap(self.ATR_jt, ax=axes[1, 1], cmap='YlOrRd', vmin=vmin, cbar_kws={'label': 'Sucrose Content (ATR)'})
        axes[1, 1].set_title('Sucrose Content per Block and Period')
        axes[1, 1].set_xlabel('Period Index')
        axes[1, 1].set_ylabel('Block Index')
        axes[1, 1].set_xticks(range(len(self.T)))
        axes[1, 1].set_xticklabels(self.T, rotation=45)
        axes[1, 1].set_yticks(range(len(self.B)))
        axes[1, 1].set_yticklabels(self.B)

        plt.tight_layout()
        plt.show()

    def to_txt(self): 
        pass
