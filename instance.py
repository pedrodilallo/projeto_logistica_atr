from typing import Dict, List, Set, Optional, Any
import random
from scipy.stats import poisson
import numpy as np
import pickle
from datetime import datetime


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
    Bl_l: Optional[Dict[Any, Set[Any]]]  # Dict with keys in F, values are sets of indices from B
    Bs_t: Optional[Dict[Any, Set[Any]]]  # Dict with keys in T, values are sets of indices from B
    st_ij: Optional[np.ndarray]  # Travel times, shape (len(B), len(B))
    dist_ij: Optional[np.ndarray]  # Distances, shape (len(B), len(B))
    ATR_jt: Optional[np.ndarray]  # Sucrose per ton, shape (len(B), len(T))
    Ht: Optional[float]  # Machine hours per day
    N_t: Optional[int]  # Number of trucks
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
        self.Bl_l = kwargs.get('Bl_l', None)
        self.Bs_t = kwargs.get('Bs_t', None)
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
        self.N = kwargs.get('microperiods_per_t', 22)
        self.S_t = {t: [s + self.N * (t - 1) for s in range(1, self.N + 1)] for t in self.T} # type: ignore
        self.SO_t = {t: [self.S_t[t][0]] for t in self.T} # type: ignore
    
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

        # Microperiods
        microperiods_per_t = kwargs.get('microperiods_per_t', 3)
        self.S_t = {t: list(range(1, microperiods_per_t + 1)) for t in self.T}
        self.SO_t = {t: 1 for t in self.T} # type: ignore

        # Normal distributions
        self.p_j = np.random.normal(
            kwargs.get('p_j_mean', 100), 
            kwargs.get('p_j_std', 20), 
            size_B
        ).tolist()

        self.TCH_j = np.random.normal(
            kwargs.get('TCH_j_mean', 50), 
            kwargs.get('TCH_j_std', 10), 
            size_B
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

        self.ATR_jt = np.random.normal(
            kwargs.get('ATR_jt_mean', 10), 
            kwargs.get('ATR_jt_std', 1), 
            (size_B, size_T)
        )

        # Uniform distributions
        self.col_j = np.random.uniform(
            kwargs.get('col_j_min', 5), 
            kwargs.get('col_j_max', 10), 
            size_B
        ).tolist()

        self.transp_j = np.random.uniform(
            kwargs.get('transp_j_min', 10), 
            kwargs.get('transp_j_max', 20), 
            size_B
        ).tolist()

        self.fi_j = np.random.uniform(
            kwargs.get('fi_j_min', 0), 
            kwargs.get('fi_j_max', 1), 
            size_B
        ).tolist()

        self.vin_t = np.random.uniform(
            kwargs.get('vin_t_min', 10), 
            kwargs.get('vin_t_max', 20), 
            size_T
        ).tolist()

        self.K_t = np.random.uniform(
            kwargs.get('K_t_min', 0.8), 
            kwargs.get('K_t_max', 1.0), 
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
        self.Nm_l = poisson.rvs(
            kwargs.get('Nm_l_mean', 5), 
            size=(size_F,) ).tolist() # type: ignore

        # Derived parameters
        maxd_t_offset = kwargs.get('maxd_t_offset', 100)
        self.maxd_t = [mind + maxd_t_offset for mind in self.mind_t] # type: ignore

        # Sets
        self.V_J = {j for j, fi in zip(self.B, self.fi_j) if fi > 0.5} # type: ignore
        self.Bl_l = {l: set(random.sample(self.B, k=random.randint(1, size_B))) for l in self.F}
        self.Bs_t = {t: set(random.sample(self.B, k=random.randint(1, size_B))) for t in self.T}

        # Fixed values
        self.Ht = kwargs.get('Ht', 8.0)
        self.N_t = kwargs.get('N_t', 10)
        self.Htt = kwargs.get('Htt', 8.0)
        self.Np = kwargs.get('Np', 5)
        self.mo = kwargs.get('mo', 10.0)
        self.bs = kwargs.get('bs', 5.0)
        self.md = kwargs.get('md', 1.0)
        self.pa = kwargs.get('pa', 20.0)

    def save(self):
        sizes = f"{len(self.B)}_{len(self.F)}_{len(self.T)}" # type: ignore
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"instance_objects/instance_{sizes}_{current_time}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    def to_txt(self): 
        pass
