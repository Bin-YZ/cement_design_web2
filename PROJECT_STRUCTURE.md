# 项目结构和模块关系

## 文件列表

```
concrete_mix_optimizer/
│
├── main.py                 # 入口文件 (592 bytes)
├── model_wrapper.py        # 模型封装 (1.3 KB)
├── sampler.py              # 采样工具 (3.7 KB)
├── pareto_optimizer.py     # Pareto优化 (1.8 KB)
├── metrics.py              # 指标计算 (2.9 KB)
├── nsga_problem.py         # NSGA-II问题 (5.2 KB)
├── optimizer_gui.py        # 主GUI (28 KB)
├── requirements.txt        # 依赖项 (219 bytes)
└── README.md               # 文档 (3.6 KB)
```

## 模块依赖关系

```
main.py
  └── optimizer_gui.py
        ├── model_wrapper.py
        ├── sampler.py
        ├── pareto_optimizer.py
        ├── metrics.py
        └── nsga_problem.py
              ├── model_wrapper.py
              ├── metrics.py
              └── sampler.py (project_to_bounds_with_sum)
```

## 数据流

### 采样优化器流程:
```
用户输入参数
    ↓
Sampler.sample_mixes() → 生成可行配比
    ↓
ModelWrapper.predict() → 模型预测 E 和 CO2_abs
    ↓
MetricsCalculator.add_metrics() → 计算所有指标
    ↓
ParetoOptimizer.pareto_mask() → 过滤Pareto前沿
    ↓
可视化 + 导出CSV
```

### NSGA-II优化器流程:
```
用户输入参数
    ↓
创建 ConcreteMixProblem
    ↓
NSGA2算法迭代
    ├── decode() → 解码决策变量为配比
    ├── ModelWrapper.predict() → 预测
    ├── MetricsCalculator.add_metrics() → 计算指标
    └── 评估目标函数
    ↓
收集Pareto前沿
    ↓
可视化 + 导出CSV
```

## 类和函数职责

### model_wrapper.py
- **ModelWrapper**: 
  - 加载Keras模型(带自定义损失)
  - 预测 E 和 CO2_abs

### sampler.py
- **project_to_bounds_with_sum()**: 
  - 投影向量满足边界和总和约束
- **Sampler**:
  - parse_range(): 解析"min,max"字符串
  - sample_group(): 采样满足约束的一组变量
  - sample_mixes(): 批量生成可行配比

### pareto_optimizer.py
- **ParetoOptimizer**:
  - dominates(): 判断支配关系
  - pareto_mask(): 返回非支配解布尔数组

### metrics.py
- **MetricsCalculator**:
  - add_metrics(): 计算CO2排放、成本、净排放等

### nsga_problem.py
- **ConcreteMixProblem**(继承pymoo.Problem):
  - decode(): 决策变量 → 配比DataFrame
  - _evaluate(): 评估多目标函数值

### optimizer_gui.py
- **OptimizerGUI**:
  - 创建所有界面组件
  - _on_sampling_click(): 运行采样优化
  - _on_nsga_click(): 运行NSGA-II优化
  - _plot_pareto_sampling(): 绘制采样结果
  - _plot_nsga_objectives(): 绘制NSGA结果
  - _display_*_results(): 显示和保存结果

### main.py
- **create_gui()**: 简化的入口函数

## 关键设计决策

1. **分离关注点**: 每个模块只负责一个功能领域
2. **可测试性**: 纯函数和类方法易于单元测试
3. **可扩展性**: 添加新材料/目标只需修改少量代码
4. **类型提示**: 提高代码可读性和IDE支持
5. **文档化**: 每个类/函数都有清晰的docstring

## 与原始代码的对比

### 原始代码问题:
- ❌ 所有类混在一个文件
- ❌ 缺少模块化
- ❌ MetricsCalculator未定义
- ❌ 难以维护和测试

### 重构后优势:
- ✅ 7个清晰的模块文件
- ✅ 每个文件职责单一
- ✅ 完整的MetricsCalculator实现
- ✅ 易于维护、测试和扩展
- ✅ 完整的类型提示和文档
- ✅ 包含README和requirements.txt
