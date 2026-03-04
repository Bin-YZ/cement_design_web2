# 使用示例

## 示例 1: 基本使用

```python
from main import create_gui

# 创建优化界面
gui = create_gui('your_trained_model.h5')

# GUI会自动显示在Jupyter Notebook中
# 用户可以通过界面进行交互操作
```

## 示例 2: 直接使用优化器类

```python
from optimizer_gui import OptimizerGUI

# 初始化
gui = OptimizerGUI('model.h5')

# GUI会显示所有控件
# 用户可以:
# 1. 调整熟料相范围 (C3S, C2S, C3A, C4AF)
# 2. 设置SCM范围 (硅灰, 矿渣, 粉煤灰等)
# 3. 选择优化目标
# 4. 运行优化
# 5. 查看结果可视化
```

## 示例 3: 单独使用各个模块

### 3.1 使用模型预测

```python
from model_wrapper import ModelWrapper
import pandas as pd

# 加载模型
model = ModelWrapper('model.h5')

# 准备输入数据
df = pd.DataFrame({
    'C3S': [55.0],
    'C2S': [20.0],
    'C3A': [8.0],
    'C4AF': [10.0],
    'time': [28.0],
    'silica_fume': [5.0],
    'GGBFS': [30.0],
    'fly_ash': [10.0],
    'calcined_clay': [0.0],
    'limestone': [0.0]
})

# 预测
predictions = model.predict(df)
print(f"E = {predictions[0, 0]:.2f} GPa")
print(f"CO2_abs = {predictions[0, 1]:.4f} kg/kg")
```

### 3.2 使用采样器

```python
from sampler import Sampler
import numpy as np

# 定义边界
clinker_bounds = {
    'C3S': (45, 80),
    'C2S': (10, 32),
    'C3A': (0, 14),
    'C4AF': (0, 15)
}

scms_bounds = {
    'silica_fume': (0, 10),
    'GGBFS': (0, 80),
    'fly_ash': (0, 35),
    'calcined_clay': (0, 35),
    'limestone': (0, 35)
}

# 生成样本
rng = np.random.default_rng(42)
mixes = Sampler.sample_mixes(
    n=100,
    clinker_sum_rng=(20, 96),
    clinker_bounds=clinker_bounds,
    scms_bounds=scms_bounds,
    total_binder_target=96.0,
    rng=rng
)

print(f"Generated {len(mixes)} feasible mixes")
```

### 3.3 使用指标计算器

```python
from metrics import MetricsCalculator
import pandas as pd
import numpy as np

# 初始化
calc = MetricsCalculator(
    fixed_wc=0.5,
    fixed_gypsum=4.0,
    fixed_temp=25.0
)

# 排放和成本因子
emission_factors = {
    'C3S': 0.82, 'C2S': 0.69, 'C3A': 0.73, 'C4AF': 0.55,
    'silica_fume': 0.0035, 'GGBFS': 0.13, 'fly_ash': 0.004,
    'calcined_clay': 0.27, 'limestone': 0.0023, 'Gypsum': 0.0082
}

cost_factors = {
    'C3S': 0.141, 'C2S': 0.141, 'C3A': 0.141, 'C4AF': 0.141,
    'silica_fume': 4.92, 'GGBFS': 0.056, 'fly_ash': 0.02,
    'calcined_clay': 0.11, 'limestone': 0.0227, 'Gypsum': 0.051
}

# 假设的配比和预测值
df = pd.DataFrame({
    'C3S': [55], 'C2S': [20], 'C3A': [8], 'C4AF': [10],
    'silica_fume': [5], 'GGBFS': [30], 'fly_ash': [10],
    'calcined_clay': [0], 'limestone': [0]
})

E_pred = np.array([35.0])
CO2_abs_pred = np.array([0.15])

# 计算指标
df_with_metrics = calc.add_metrics(df, E_pred, CO2_abs_pred, 
                                   emission_factors, cost_factors)

print(df_with_metrics[['E', 'CO2_abs', 'CO2_emission', 'Cost', 'Net_emission']])
```

### 3.4 使用Pareto优化器

```python
from pareto_optimizer import ParetoOptimizer

# 假设有一些解
solutions = [
    {'E': 30, 'Cost': 0.10, 'Net_emission': 0.50},
    {'E': 35, 'Cost': 0.12, 'Net_emission': 0.48},
    {'E': 32, 'Cost': 0.09, 'Net_emission': 0.52},
    {'E': 40, 'Cost': 0.15, 'Net_emission': 0.45},
]

# 定义优化方向
sense = {
    'E': 'max',           # 最大化弹性模量
    'Cost': 'min',        # 最小化成本
    'Net_emission': 'min' # 最小化净排放
}

# 找到Pareto前沿
mask = ParetoOptimizer.pareto_mask(solutions, sense)
pareto_solutions = [sol for sol, is_pareto in zip(solutions, mask) if is_pareto]

print(f"Pareto solutions: {len(pareto_solutions)} out of {len(solutions)}")
for sol in pareto_solutions:
    print(sol)
```

## 示例 4: 自定义优化目标

如果你想添加新的优化目标,可以修改以下部分:

### 在 optimizer_gui.py 中:
```python
# 添加新的目标选项
self.obj_select = w.SelectMultiple(
    options=[
        ("Maximize E", "E_max"),
        ("Maximize CO₂ uptake", "CO2abs_max"),
        ("Minimize CO₂ emission", "CO2_min"),
        ("Minimize Cost", "Cost_min"),
        ("Minimize Net emission", "Net_min"),
        ("Minimize Water demand", "Water_min"),  # 新目标
    ],
    ...
)
```

### 在 metrics.py 中:
```python
# 在 add_metrics 方法中添加新指标
def add_metrics(self, df, E_pred, CO2_abs_pred, emission_factors, cost_factors):
    # ... 现有代码 ...
    
    # 添加新指标
    df["Water_demand"] = df["C3S"] * 0.001 + df["C2S"] * 0.0008  # 示例公式
    
    return df
```

## 示例 5: 批处理优化

```python
from optimizer_gui import OptimizerGUI
import pandas as pd

# 如果想要批量运行多次优化
gui = OptimizerGUI('model.h5')

# 保存多个seed的结果
results_list = []
for seed in [42, 123, 456, 789, 1000]:
    gui.seed_in.value = str(seed)
    # 用户需要手动点击"Run Sampling"按钮
    # 或者可以通过编程方式调用内部方法(不推荐在生产环境)

# 合并结果
# all_results = pd.concat(results_list, ignore_index=True)
# all_results.to_csv('batch_optimization_results.csv', index=False)
```

## 常见问题

### Q1: 如何修改固定参数?
A: 在 `OptimizerGUI` 类中修改类属性:
```python
class OptimizerGUI:
    FIXED_WC = 0.4        # 修改水灰比
    FIXED_GYPSUM = 5.0    # 修改石膏含量
    FIXED_TEMP = 20.0     # 修改温度
```

### Q2: 如何添加新的材料?
A: 在 `MATERIALS` 列表中添加:
```python
MATERIALS = [
    # ... 现有材料 ...
    ("New Material", "new_material", emission_factor, cost_factor),
]
```
同时需要在模型中支持该材料。

### Q3: NSGA-II运行很慢怎么办?
A: 减少种群大小或代数:
- 种群: 160 → 80
- 代数: 120 → 60

### Q4: 如何导出所有评估过的解?
A: 在NSGA-II代码中,`all_F` 包含所有评估的目标值。你可以修改代码保存这些数据。

## 性能建议

1. **采样优化器**: 
   - 1000-2000样本足够初步探索
   - 增加样本可以找到更多Pareto解

2. **NSGA-II优化器**:
   - 种群100-200, 代数50-150较平衡
   - 2目标: 较快收敛
   - 3+目标: 需要更大种群和更多代数

3. **可视化**:
   - 3D图可能较慢,考虑只查看前3个目标
   - 关闭"显示所有点"可加速渲染
