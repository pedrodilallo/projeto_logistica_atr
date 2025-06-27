# Optimization Model for Harvest Planning

This repository contains a Python-based optimization model for agricultural harvest planning, managing harvest blocks, fronts, and periods with production and logistical data. It uses Pyomo for optimization, with data generation, storage, and visualization capabilities.

$$
\begin{flalign}
& \min mo \sum_{t=1}^T wm_t + bs \sum_{j=1}^B wb_j + md \sum_{l=1}^F \sum_{i=1}^B \sum_{j=1}^B \sum_{s=1}^S dist_{ij} z_{lijs} - pa \sum_{t=1}^T \sum_{j=1}^B ATR_{jt} \sum_{s \in S_t} \sum_{l \in F} x_{ljs} \label{1}
\end{flalign}
$$

## Overview

- **Purpose**: Models harvest planning to optimize block scheduling, front assignments, and production across macro and micro periods, minimizing costs (unmet demand, unharvested cane, transport distance) while maximizing sucrose revenue.
- **Key Files**:
  - `main.py`: Loads and visualizes a saved `Instance` object, displaying attributes and plots.
  - `instance.py`: Defines the `Instance` class for data storage and generation.
  - `model.py`: Defines the `GLSP_model` class for building and solving the optimization model.
  - `solutions.py`: Placeholder for solution processing (not implemented).
  - `requirements.txt`: Lists dependencies (e.g., `numpy`, `pyomo`, `gurobipy`).

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
Ensure a solver (e.g., Gurobi, CBC, or GLPK) is installed and licensed. Replace Gurobi with open-source alternatives if needed.

## File Descriptions

### `main.py`
- **Purpose**: Loads a saved `Instance` object and visualizes data.
- **Functionality**:
  - Loads a pickle file (e.g., `instance_objects/instance_5_3_4_202506241522.pkl`).
  - Prints attributes: blocks (`B`), fronts (`F`), periods (`T`), production (`p_j`), distance matrix (`dist_ij`).
  - Plots:
    - Heatmap of `dist_ij` using `matplotlib` and `seaborn`.
    - Bar chart of `p_j` using `matplotlib`.
- **Usage**:
  1. Update `filename` in `main.py` to match a saved pickle file.
  2. Run:
     ```bash
     python main.py
     ```
  3. View console output and plots.

### `instance.py`
- **Purpose**: Defines the `Instance` class to store and generate harvest planning data.
- **Class: `Instance`**
  - **Attributes**:
    - Sets: `B` (blocks), `F` (fronts), `T` (macroperiods), `V_J` (irrigable blocks), `Bl_l` (blocks per front), `Bs_t` (blocks per period), `S_t` (microperiods per macroperiod), `SO_t` (first microperiod per macroperiod).
    - Parameters: `p_j` (production), `fi_j` (irrigable fraction), `TCH_j` (productivity), `col_j` (harvest capacity), `transp_j` (transport capacity), `Nm_l` (machines per front), `mind_t` (min demand), `maxd_t` (max demand), `vin_t` (min irrigable area), `K_t` (fleet availability), `st_ij` (travel times), `dist_ij` (distances), `ATR_jt` (sucrose per ton), `Ht` (machine hours), `N_t` (trucks), `Htt` (truck hours), `Np` (platform vehicles), `mo` (unmet demand cost), `bs` (unharvested cane cost), `md` (transport cost), `pa` (sucrose revenue), `bm_lj` (min production per front/block).
  - **Methods**:
    - `__init__(**kwargs)`: Initializes attributes, defaulting to `None` unless provided. Sets `N` (microperiods per macroperiod, default 22), `S_t`, and `SO_t`.
    - `generate(size_B, size_F, size_T, **kwargs)`: Generates random data:
      - Sets: `B`, `F`, `T` as index lists; `V_J`, `Bl_l`, `Bs_t` as random subsets.
      - Parameters: Uses normal (`p_j`, `TCH_j`, `mind_t`, `ATR_jt`), uniform (`col_j`, `transp_j`, `fi_j`, `vin_t`, `K_t`, `bm_lj`), or Poisson (`Nm_l`) distributions. Computes `dist_ij` (Euclidean distances from coordinates) and `st_ij` (distances/40). Sets `maxd_t` as `mind_t` + offset.
      - Fixed values: `Ht`, `N_t`, `Htt`, `Np`, `mo`, `bs`, `md`, `pa`.
    - `save()`: Saves instance to a pickle file in `instance_objects/` with format `instance_{size_B}_{size_F}_{size_T}_{timestamp}.pkl`.
    - `to_txt()`: Placeholder for text export.
    - `to_csvs()`: Placeholder for CSV export.
- **Usage**:
  ```python
  from instance import Instance
  inst = Instance()
  inst.generate(size_B=5, size_F=3, size_T=4, p_j_mean=120, coord_min=0, coord_max=50)
  inst.save()
  ```

### `model.py`
- **Purpose**: Defines the `GLSP_model` class for building a Pyomo optimization model.
- **Class: `GLSP_model`**
  - **Attributes**:
    - `instance`: Input `Instance` object.
    - `model`: Pyomo `ConcreteModel` for optimization.
  - **Methods**:
    - `__init__(instance)`: Initializes with an `Instance` and builds model components.
    - `build_params()`: Defines Pyomo sets (`B`, `F`, `T`, `S`, `S_t`, `V_J`, `Bl_l`, `Bs_t`) and parameters (`p_j`, `fi_j`, `TCH_j`, `col_j`, `transp_j`, `Nm_l`, `mind_t`, `maxd_t`, `vin_t`, `K_t`, `st_ij`, `dist_ij`, `ATR_jt`, `bm_lj`, `Ht`, `N_t`, `Htt`, `Np`, `mo`, `bs`, `md`, `pa`) from `instance`.
    - `build_vars()`: Defines variables:
      - `x(l,j,s)`: Non-negative real, production amount for front `l`, block `j`, microperiod `s`, bounded by `p_j`.
      - `y(l,j,s)`: Binary, 1 if front `l` harvests block `j` in microperiod `s`.
      - `z(l,i,j,s)`: Binary, 1 if front `l` moves from block `i` to `j` in microperiod `s`.
      - `wm(t)`: Non-negative real, unmet demand in macroperiod `t`, bounded by `mind_t`.
      - `wb(j)`: Non-negative real, unharvested cane in block `j`, bounded by `p_j`.
      - Fixes `y(l,j,s)` to 0 for invalid block-front-period combinations.
    - `build_model()`: Defines:
      - **Objective**: Minimize `mo * sum(wm[t]) + bs * sum(wb[j]) + md * sum(dist[i,j] * z[l,i,j,s]) - pa * sum(ATR_jt[j,t] * sum(x[l,j,s]))`.
      - **Constraints**:
        1. Min/max demand per macroperiod (`mind_t`, `maxd_t`).
        2. Harvest and transport limit per block (`p_j * f[j]`).
        3. Min irrigable area per macroperiod (`vin_t`).
        4. Harvest capacity per front and macroperiod (`K_t`).
        5. Transport capacity per macroperiod (`K_t`).
        6. Production capacity per front, block, microperiod.
        7. Min production per front, block, microperiod.
        8. Exactly one block harvested per front, microperiod, for valid blocks.
        9. Consistent front movement between blocks across microperiods.
        10. Idle microperiods enforce non-increasing `y`.
- **Usage**:
  ```python
  from model import GLSP_model
  model = GLSP_model(inst)
  solver = SolverFactory('gurobi')
  solver.solve(model.model)
  ```

### `solutions.py`
- **Purpose**: Placeholder for processing and analyzing optimization results.
- **Status**: Not implemented.

## Usage Guide

### Generating and Saving Data
```python
from instance import Instance
inst = Instance()
inst.generate(5, 3, 4, p_j_mean=120, coord_min=0, coord_max=50)
inst.save()  # Saves to instance_objects/
```

### Visualizing Data
1. Update `filename` in `main.py` to the saved pickle file.
2. Run:
   ```bash
   python main.py
   ```
3. View console output (attributes) and plots (distance heatmap, production bar chart).

### Running Optimization
```python
from instance import Instance
from model import GLSP_model
inst = Instance()
inst.generate(5, 3, 4)
model = GLSP_model(inst)
results = solver.solve(model.model)
```

## Contributing
Fork, submit issues, or pull requests. Ensure changes align with optimization goals.

## Notes
- **Solvers**: Gurobi is used by default; replace with CBC/GLPK if unlicensed.
- **Customization**: Adjust `generate` parameters in `instance.py` for different distributions.
- **Future Work**:
  - Implement `solutions.py` for result analysis.
  - Enhance `to_txt` and `to_csvs` for data export.
  - Add error handling and input validation.
