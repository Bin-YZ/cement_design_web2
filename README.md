# Cement Mix Multi-Objective Optimization

A clearly refactored tool for multi-objective optimization of concrete mixes.

## File Structure

```
├── model_wrapper.py      # Model loading and prediction
├── sampler.py            # Sampling tools and constraint projection
├── pareto_optimizer.py   # Pareto dominance calculation
├── metrics.py            # Performance metric calculation
├── nsga_problem.py       # NSGA-II problem definition
├── optimizer_gui.py      # Main GUI interface
└── main.py               # Program entry point
```

## Module Description

### 1. model_wrapper.py
- `ModelWrapper`: Encapsulates Keras model loading and prediction
  - Loads models with custom loss functions
  - Performs predictions on input features

### 2. sampler.py
- `project_to_bounds_with_sum()`: Projects vectors to satisfy boundary constraints and sum constraints
- `Sampler`: Sampling utility class
  - `parse_range()`: Parses range strings
  - `sample_group()`: Samples a group of variables under constraints
  - `sample_mixes()`: Generates feasible concrete mixes

### 3. pareto_optimizer.py
- `ParetoOptimizer`: Pareto optimization utility
  - `dominates()`: Checks dominance relationship between solutions
  - `pareto_mask()`: Returns mask of non-dominated solutions

### 4. metrics.py
- `MetricsCalculator`: Performance metric calculator
  - Calculates CO2 emissions
  - Calculates costs
  - Calculates net emissions
  - Adds fixed parameters

### 5. nsga_problem.py
- `ConcreteMixProblem`: NSGA-II optimization problem definition
  - Defines decision variable boundaries
  - Decodes decision variables into feasible mixes
  - Evaluates multi-objective functions

### 6. optimizer_gui.py
- `OptimizerGUI`: Main interactive interface
  - Input parameter settings
  - Sampling optimizer
  - NSGA-II optimizer
  - Result visualization and export

### 7. main.py
- Program entry point
- GUI creation function

## Usage

### Basic Usage

```python
from main import create_gui

# Create GUI
gui = create_gui('your_model.h5')
```

### Usage in Jupyter Notebook

```python
from optimizer_gui import OptimizerGUI

# Initialize GUI
gui = OptimizerGUI('your_model.h5')

# GUI will display automatically
# Users can:
# 1. Set ranges for clinker phases and SCMs
# 2. Select optimization mode (sampling or NSGA-II)
# 3. Set optimization objectives
# 4. Run optimization
# 5. View Pareto front visualization
# 6. Export result CSV
```

## Dependencies

### Required
- tensorflow
- pandas
- numpy
- ipywidgets
- plotly
- IPython

### Optional
- pymoo (for NSGA-II optimizer)

Install dependencies:
```bash
pip install tensorflow pandas numpy ipywidgets plotly
pip install pymoo  # Optional, for NSGA-II
```

## Optimization Modes

### 1. Sampling Optimizer
- Generates a large number of feasible mixes through random sampling
- Filters non-dominated solutions using Pareto dominance
- Fast, suitable for initial exploration

### 2. NSGA-II Optimizer
- Genetic algorithm-based multi-objective optimization
- Searches the Pareto front more systematically
- Requires pymoo installation
- Suitable for in-depth optimization

## Optimization Objectives

- **Maximize E**: Maximize elastic modulus
- **Maximize CO₂ uptake**: Maximize CO₂ absorption
- **Minimize CO₂ emission**: Minimize CO₂ emissions
- **Minimize Cost**: Minimize cost
- **Minimize Net emission**: Minimize net emissions (emissions - absorption)

## Output Files

- `sampling_pareto_recipes.csv`: Pareto front mixes from sampling optimizer
- `nsga2_pareto_recipes.csv`: Pareto front mixes from NSGA-II optimizer

## Notes

1. All material percentage inputs are percentages of clinker (clinker totals 100%)
2. Fixed parameters:
   - Water-cement ratio (w/c): 0.5
   - Gypsum: 4%
   - Temperature: 25°C
   - Total clinker + SCMs: 96%
3. Time range: 1-40000 days
4. Results are automatically saved as CSV files

## Code Features

- **Modular**: Each file has clear responsibilities, easy to maintain
- **Type hints**: Clear type annotations for function parameters and return values
- **Docstrings**: Detailed descriptions for each class and method
- **Error handling**: Comprehensive exception catching and user prompts
- **Extensibility**: Easy to add new materials, objectives, or constraints