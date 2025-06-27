---
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{multicol}
---

# Optimization Model for Harvest Planning

This repository contains a Python-based optimization model for harvest planning, focusing on managing harvest blocks, fronts, and periods with associated production and logistical data. The code leverages various libraries for data generation, optimization, visualization, and data storage.

$$
\begin{flalign}
& \min mo \sum_{t=1}^T \sum_{m \in M} wm_{kt} + bs \sum_{j=1}^B wb_j + md \sum_{l=1}^F \sum_{i=1}^B \sum_{j=1}^B \sum_{s=2}^N dist_{ij} z_{lijs} - pa \sum_{t=1}^T \sum_{j=1}^B ATR_{jt} \sum_{s \in t} \sum_{l \in F} x_{ljs}  \label{1}
\end{flalign}
$$

## Overview

- **Purpose**: The project models agricultural harvest planning using an `Instance` class to store data and provides tools to generate, save, visualize, and potentially optimize harvest schedules.
- **Key Files**:
  - `main.py`: Entry point for loading and visualizing a saved `Instance` object.
  - `instance.py`: Defines the `Instance` class for data storage and generation.
  - Additional modules (`model`, `solutions`) are placeholders for future optimization logic.

## Requirements

Install the required dependencies listed in `requirements.txt` via:
```bash
pip install -r requirements.txt
```
Ensure all libraries, including any solver dependencies (e.g., Gurobi), are properly licensed or replaced with open-source alternatives if needed.

## File Descriptions

### `main.py`
- **Purpose**: Loads a pre-saved `Instance` object from a pickle file and visualizes its data.
- **Functionality**:
  - Loads an `Instance` object saved as a pickle file.
  - Prints basic attributes (e.g., blocks, fronts, periods, production).
  - Generates a heatmap of the distance matrix and a bar chart of production per block.
- **How to Use**:
  1. Ensure a pickle file (e.g., `instance_objects/instance_5_3_4_202506241757.pkl`) exists from a prior `save` call.
  2. Update the `filename` variable in `main.py` to match the saved file.
  3. Run the script:
     ```bash
     python main.py
     ```
  4. View the printed attributes and generated plots.

### `instance.py`
- **Purpose**: Defines the `Instance` class to hold and generate harvest planning data.
- **Functionality**:
  - Stores attributes like block indices (`B`), front indices (`F`), period indices (`T`), production (`p_j`), distances (`dist_ij`), etc.
  - Includes a `generate` method to create random data based on sizes (`size_B`, `size_F`, `size_T`) with customizable distribution parameters.
  - Includes a `save` method to serialize the instance to a pickle file named by sizes and timestamp.
- **How to Use**:
  1. Import the `Instance` class:
     ```python
     from instance import Instance
     ```
  2. Create an instance and generate data:
     ```python
     inst = Instance()
     inst.generate(size_B=5, size_F=3, size_T=4, p_j_mean=100, p_j_std=20)
     ```
  3. Save the instance:
     ```python
     inst.save()  # Saves to instance_objects/instance_5_3_4_202506241757.pkl
     ```
  4. Customize parameters via `kwargs` (e.g., `p_j_mean`, `coord_min`) to adjust distributions.

## Interaction Guide

### Generating and Saving an Instance
- Use `instance.py` to create and save data:
  ```python
  inst = Instance()
  inst.generate(5, 3, 4, p_j_mean=120, coord_min=0, coord_max=50)
  inst.save()  # File saved in instance_objects/ with timestamp
  ```
- The `generate` method uses random distributions (normal, uniform, Poisson) with default or custom parameters.

### Visualizing Data
- Use `main.py` to load and visualize:
  - Update `filename` to the saved pickle file.
  - Run `python main.py` to see printed attributes and plots (heatmap for `dist_ij`, bar chart for `p_j`).
- Customize visualizations by modifying `plt` commands (e.g., change `cmap` or add labels).

### Future Development
- Implement `model.py` and `solutions.py` for optimization using Pyomo or Gurobi.
- Enhance `to_txt` and `to_csvs` in `instance.py` for alternative data export.
- Add error handling and validation for robustness.

## Contributing
Feel free to fork this repository, submit issues, or pull requests. Ensure all changes align with the project’s optimization focus.

### Notes
- **Requirements**: The README now points to `requirements.txt` for dependency installation, assuming you have created or will create this file with the necessary packages (e.g., `numpy`, `scipy`, `matplotlib`, `pyomo`, `gurobipy`, etc.).
- **Timestamp**: The example filenames use `202506241757` (05:57 PM -03 on June 24, 2025), reflecting the current date and time.
- **Customization**: Adjust the license section and add more details to `requirements.txt` as needed based on your project setup.

Let me know if you need help generating `requirements.txt` or further refinements!