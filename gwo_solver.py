import numpy as np

def gwo_inverse_design(
    model, 
    target_dict, 
    active_vars, 
    bounds_dict, 
    fixed_params, 
    pop_size=20, 
    max_iter=50
):
    """
    基于灰狼优化算法 (GWO) 的混凝土逆向设计求解器
    
    参数:
    model: 你的预测模型 wrapper (必须有 model.predict(df) 方法)
    target_dict: 目标字典, 例如 {'E': 50.0, 'Cost': 0.08}
    active_vars: 参与优化的变量名列表, 例如 ['C3S', 'C2S', 'fly_ash']
    bounds_dict: 变量范围字典, 例如 {'C3S': [40, 80], 'fly_ash': [0, 30]}
    fixed_params: 固定参数, 包含 {'total_binder': 96.0, 'gypsum': 4.0, 'wc': 0.5, ...}
    pop_size: 狼群数量 (种群大小)
    max_iter: 最大迭代次数
    
    返回:
    best_position (dict): 最佳配方
    best_score (float): 最小误差值
    convergence_curve (list): 迭代收敛曲线
    """

    # 1. 初始化设置
    dim = len(active_vars)
    lb = np.array([bounds_dict[k][0] for k in active_vars]) # 下限
    ub = np.array([bounds_dict[k][1] for k in active_vars]) # 上限
    
    # 初始化 Alpha, Beta, Delta 狼的位置和得分
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")
    
    # 随机初始化狼群位置 X (Pop_size x dim)
    # X 代表不同的配方组合
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    
    convergence_curve = []
    
    # 辅助函数：修复约束 (Ensure sum constraints)
    # 这是一个关键步骤：GWO 更新后可能破坏总量约束，必须修复
    def repair_constraints(pos_vector):
        # 1. 裁剪到边界内
        pos_vector = np.clip(pos_vector, lb, ub)
        
        # 2. 归一化以满足 Total Binder Target
        # 你的配方总量目标 (通常是 100 - Gypsum)
        target_sum = fixed_params.get('total_binder', 96.0)
        
        current_sum = np.sum(pos_vector)
        if current_sum > 0:
            scale_factor = target_sum / current_sum
            pos_vector = pos_vector * scale_factor
            
        # 再次裁剪以防归一化后越界 (简单的迭代修复)
        pos_vector = np.clip(pos_vector, lb, ub)
        return pos_vector

    # 辅助函数：计算适应度 (Loss Function)
    def calculate_fitness(pos_vector):
        # 1. 构建完整输入字典
        input_data = fixed_params.copy()
        for i, var in enumerate(active_vars):
            input_data[var] = pos_vector[i]
        
        # 将未选中的 SCMs 设为 0 (如果需要)
        # 这里假设 input_data 会被 model.predict 自动处理缺失列或已有默认值
        
        # 2. 调用模型预测
        # 注意：这里需要根据你的 model_wrapper 调整格式，假设它接受 dict 或 list
        # 这里模拟将其转为 DataFrame 格式的一行进行预测
        preds = model.predict_single_sample(input_data) # 假设你封装了这个方法
        
        # 3. 计算误差 (MSE 或 MAE)
        loss = 0.0
        for key, target_val in target_dict.items():
            pred_val = preds.get(key, 0)
            # 使用相对误差平方，避免量级差异 (Cost 很小，Strength 很大)
            if target_val != 0:
                loss += ((pred_val - target_val) / target_val) ** 2
            else:
                loss += (pred_val - target_val) ** 2
        return loss

    # --- 主循环 (Main Loop) ---
    print("🐺 GWO Optimization Started...")
    
    for l in range(0, max_iter):
        # 线性衰减因子 a 从 2 降到 0
        a = 2 - l * ((2) / max_iter)
        
        for i in range(0, pop_size):
            # 1. 强制修复约束 (总量守恒)
            Positions[i, :] = repair_constraints(Positions[i, :])
            
            # 2. 计算适应度
            fitness = calculate_fitness(Positions[i, :])
            
            # 3. 更新 Alpha, Beta, Delta
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()
        
        # 4. 更新狼群位置 (狩猎行为)
        for i in range(0, pop_size):
            for j in range(0, dim):
                r1 = np.random.random()
                r2 = np.random.random()
                
                # Towards Alpha
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha
                
                # Towards Beta
                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta
                
                # Towards Delta
                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta
                
                # 平均位置
                Positions[i, j] = (X1 + X2 + X3) / 3
        
        convergence_curve.append(Alpha_score)
        
        # 简单的早停 (可选)
        if Alpha_score < 1e-4:
            break

    # --- 结束 ---
    print(f"✅ GWO Finished. Best Loss: {Alpha_score:.5f}")
    
    # 构建最佳结果字典
    best_recipe = {}
    for i, var in enumerate(active_vars):
        best_recipe[var] = Alpha_pos[i]
        
    return best_recipe, Alpha_score, convergence_curve