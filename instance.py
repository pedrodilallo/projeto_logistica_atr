from typing import Dict, List, Set, Optional, Any
import random
from scipy.stats import poisson,truncnorm
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class Instance:
    """
    A class to hold the master and transactional data for the optimization model.
    Attributes can be optionally provided during initialization.
    """

    # Type hints for all attributes
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

    def __init__(self,name, **kwargs):
        """
        Initialize the Instance with optional keyword arguments.
        Any unspecified attributes will default to None.
        """
        self.Name = name
        self.B = kwargs.get('B', None)
        self.F = kwargs.get('F', None)
        self.T = kwargs.get('T', None)
        self.p_j = kwargs.get('p_j', None)
        self.fi_j = kwargs.get('fi_j', None)
        self.TCH_j = kwargs.get('TCH_j', None)
        self.col_j = kwargs.get('col_j', None)
        self.transp_j = kwargs.get('transp_j', None)
        self.Nm_l = kwargs.get('Nm_l', None)
        self.mind_t = kwargs.get('mind_t', None)
        self.maxd_t = kwargs.get('maxd_t', None)
        self.vin_t = kwargs.get('vin_t', None)
        self.K_t = kwargs.get('K_t', None)
        self.V_J = kwargs.get('V_J', None)
        self.Bl_j = kwargs.get('Bl_j', None)
        self.Bs_j = kwargs.get('Bs_j', None)
        self.st_ij = kwargs.get('st_ij', None)
        self.dist_ij = kwargs.get('dist_ij', None)
        self.ATR_jt = kwargs.get('ATR_jt', None)
        self.Ht = kwargs.get('Ht', None)
        self.N_t = kwargs.get('N_t', None)
        self.Htt = kwargs.get('Htt', None)
        self.Np = kwargs.get('Np', None)
        self.mo = kwargs.get('mo', None)
        self.bs = kwargs.get('bs', None)
        self.md = kwargs.get('md', None)
        self.pa = kwargs.get('pa', None)
        self.bm_lj = kwargs.get('bm_lj', None)
        self.N = kwargs.get('N', 22)
        print(kwargs.get('N'))

        try: 
            self.S_t = {t: [s + self.N * (t - 1) for s in range(1, self.N + 1)] for t in self.T} # type: ignore
            self.SO_t = {t: [self.S_t[t][0]] for t in self.T} # type: ignore
        except:
            self.S_t = None
            self.SO_t =  None

    def generate(self, size_B: int, size_F: int, size_T: int, **kwargs):
        """
        Generate an instance based on sizes using random distributions.
        
        Args:
            size_B (int): Number of harvest blocks.
            size_F (int): Number of harvest fronts.
            size_T (int): Number of macroperiods.
            **kwargs: Distribution parameters (e.g., 'p_j_mean', 'p_j_std', 'col_j_min', 'col_j_max').
        """
        # Index sets
        self.B = list(range(1, size_B + 1))
        self.F = list(range(1, size_F + 1))
        self.T = list(range(1, size_T + 1))

        min_window = kwargs.get('min_window', size_T)
        max_window = kwargs.get('max_window', size_T)
        harvest_window_j = {}
        for j in self.B:
            window_length_j = random.randint(min_window, max_window)
            start_j = random.randint(1, size_T - window_length_j + 1)
            end_j = start_j + window_length_j - 1
            harvest_window_j[j] = (start_j, end_j)

        # Generate Bs_j based on harvesting windows
        self.Bs_j = {t: {j for j in self.B if harvest_window_j[j][0] <= t <= harvest_window_j[j][1]} for t in self.T}

        # Microperiods
        microperiods_per_t = kwargs.get('microperiods_per_t', 3)
        self.S_t = {t: [s + self.N * (t - 1) for s in range(1, self.N + 1)] for t in self.T}
        self.SO_t = {t: [self.S_t[t][0]] for t in self.T} # type: ignore

        # Normal distributions
        self.p_j = np.random.normal(
            kwargs.get('p_j_mean', 10000), 
            kwargs.get('p_j_std', 20), 
            size_B
        ).tolist()

        a, b = 0, np.inf  # Truncate below 0
        mu, sigma = kwargs.get('TCH_j_mean', 100), kwargs.get('TCH_j_std', 15)
        self.TCH_j = truncnorm.rvs(
            a=(a - mu) / sigma, b=(b - mu) / sigma, loc=mu, scale=sigma, size=size_B
        ).tolist()

        self.mind_t = np.random.normal(
            kwargs.get('mind_t_mean', 500), 
            kwargs.get('mind_t_std', 50), 
            size_T
        ).tolist()

        # Compute Euclidean distance matrix in kilometers 
        coord_min = kwargs.get('coord_min', -100.0) 
        coord_max = kwargs.get('coord_max', 100.0) 
        coords = np.random.uniform(coord_min, coord_max, size=(size_B, 2))  # Shape: (size_B, 2)
        
        self.coords = coords
        
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # Shape: (size_B, size_B, 2)
        self.dist_ij = np.sqrt(np.sum(diff ** 2, axis=2))  # Shape: (size_B, size_B)
        np.fill_diagonal(self.dist_ij, 0)  # type: ignore # Set diagonal to 0 (distance to self)

        self.st_ij = self.dist_ij/40

        self.ATR_jt = np.zeros((size_B, size_T))
        for j in self.B:
            start_j, end_j = harvest_window_j[j]
            for t in self.T:
                if start_j <= t <= end_j:
                    self.ATR_jt[j-1, t-1] = np.random.normal(
                        kwargs.get('ATR_jt_mean', 10), 
                        kwargs.get('ATR_jt_std', 1)
                    )

        # Uniform distributions
        col_j_min = max(0.1, kwargs.get('col_j_min', 100))
        self.col_j = np.random.uniform(
            col_j_min, 
            kwargs.get('col_j_max', 300), 
            size_B
        ).tolist()

        transp_j_min = max(0.1, kwargs.get('transp_j_min', 100))
        self.transp_j = np.random.uniform(
            transp_j_min, 
            kwargs.get('transp_j_max', 200), 
            size_B
        ).tolist()

        self.fi_j = np.random.uniform(
            kwargs.get('fi_j_min', 0), 
            kwargs.get('fi_j_max', 1), 
            size_B
        ).tolist()

        self.vin_t = np.random.uniform(
            kwargs.get('vin_t_min', 0), 
            kwargs.get('vin_t_max', 1), 
            size_T
        ).tolist()

        self.K_t = np.random.uniform(
            kwargs.get('K_t_min', 10000), 
            kwargs.get('K_t_max', 10001), 
            size_T
        ).tolist()

        self.bm_lj = {
            l: np.random.uniform(
                kwargs.get('bm_lj_min', 0), 
                kwargs.get('bm_lj_max', 10), 
                size_B
            ).tolist() for l in self.F
        }

        # Poisson distribution
        self.Nm_l =  [max(1, val) for val in poisson.rvs(
            kwargs.get('Nm_l_mean', 5), 
            size=(size_F,) )] # type: ignore

        # Derived parameters
        maxd_t_offset = kwargs.get('maxd_t_offset', 100)
        self.maxd_t = [mind + maxd_t_offset for mind in self.mind_t] # type: ignore

        # Sets
        self.V_J = {j for j, fi in zip(self.B, self.fi_j) if fi > 0.5} # type: ignore
        self.Bl_j = {l: set(random.sample(self.B, k=random.randint(1, size_B))) for l in self.F}
        self.Bs_j = {t: set(random.sample(self.B, k=random.randint(1, size_B))) for t in self.T}

        # Fixed values
        self.Ht = kwargs.get('Ht', 8.0)
        self.N_t = kwargs.get('N_t', {x: 10 for x in range(1,size_T+1)})
        self.Htt = kwargs.get('Htt', 8.0)
        self.Np = kwargs.get('Np', size_F)
        self.mo = kwargs.get('mo', 10.0)
        self.bs = kwargs.get('bs', 5.0)
        self.md = kwargs.get('md', 1.0)
        self.pa = kwargs.get('pa', 20.0)

    def save(self):
        sizes = f"{len(self.B)}_{len(self.F)}_{len(self.T)}" # type: ignore
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"instance_objects/instance_{self.Name}_{sizes}_{current_time}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

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
