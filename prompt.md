为了完成论文中关于 L2 范数攻击（C&W 和 DeepFool）的鲁棒性评估，我们需要对标之前 L-inf 的多参数实验逻辑。评估在这两种算法的不同“攻击强度”和“置信度要求”下，病灶区域与非病灶区域的脆弱性差异是否具有普遍性。

请帮我编写一个完整的 L2 鲁棒性评估脚本（例如命名为 `evaluate_l2_robustness.py`），完成以下任务：

### 任务 0：C&W 步数先导收敛性测试（Pilot Study - 动态优化耗时）
C&W 的 `steps=1000` 可能会导致 200 个患者的测试耗时过长。为了科学严谨地节省时间，请在进入主实验循环前，先执行一个自动的“步数消融”先导实验：
- 仅随机抽取 3 个能够成功生成 random_patch 掩码的患者。
- 使用代表性参数（c=50.0, kappa=0.0, lr=0.05），对这 3 个患者分别测试 `steps_list = [100, 250, 500, 1000]`。
- 检查收敛性：比较不同 steps 下最终生成的 L2_norm。如果某个较小的 steps（如 250 或 500）与 1000 步得出的 L2_norm 差异极小（例如相对误差 < 2%），则将全局变量 `optimal_cw_steps` 设置为该较小值。如果不满足，则保守兜底使用 1000。
- 将先导测试的详细对比结果（每个 steps 的平均 L2 耗能）打印到控制台，并保存到名为 `cw_convergence_pilot_log.txt` 的文件中，作为后续论文 Method 章节的证据支撑。

### 任务 1：C&W 算法的多强度与置信度评估
- 权重参数（控制攻击力度）：c_values = [1.0, 10.0, 50.0, 100.0]
- 置信度参数（控制欺骗深度）：kappa_values = [0.0, 1.0, 10.0]
- 固定参数：steps = optimal_cw_steps (由任务 0 决定), lr = 0.05。
- 请使用嵌套循环或组合列表来遍历这些参数，并严格复用 `unified_attack_framework_fixed.py` 中带有正确 Mask 逻辑的 `cw_attack_unified` 函数。

### 任务 2：DeepFool 算法的过冲系数评估
- 过冲参数（控制跨越边界的激进程度）：overshoot_values = [0.01, 0.02, 0.05, 0.1]
- 固定参数：steps = 150 (DeepFool收敛极快，作为防死循环上限即可)。
- 必须严格复用带有正确 Mask 逻辑的 `deepfool_attack_unified` 函数。

### 任务 3：极其严格的实验控制变量与数据集约束（核心要求！）
1. 数据集：必须遍历完整的 200 个患者数据集。
2. 模式对比：每个患者的每次参数组合，必须分别测试 'lesion' 和 'random_patch' 模式（'full' 模式可选，为节省时间可略过）。
3. L0 面积绝对对齐：必须使用 `multi_metric_attack_framework.py` 中的 `generate_equivalent_random_mask` 来生成 random_patch。
4. 消除幸存者偏差：如果因为大病灶导致某患者的 random_patch 生成失败，请捕获该异常，并直接跳过该患者当前的所有测试，以确保基线绝对公平。

### 任务 4：数据记录
- 将所有跑出的结果（包括 patient_id, 算法, 模式, c值, kappa值, overshoot值, success, l2_norm, confidence_drop 等）实时追加（append）保存到 `l2_robustness_evaluation.csv` 中。防止中断导致数据丢失。

### 任务 5：生成可视化对比图 (Visualization)
- 请在脚本中配置：测试过程中，随机挑选 10 个患者案例（仅在第一次遇到他们时触发画图逻辑）。
- 针对这 10 个患者，生成“攻击前后对比图”（包含：Clean Image, Adversarial Image, 以及放大/归一化后的 Perturbation Noise）。
- 仅挑选代表性参数（如 c=50, kappa=0.0, overshoot=0.05）保存这 10 张图到 `results/l2_visualizations/` 文件夹中即可，避免生成过多图片。

请先简述你的脚本编写思路，确认无误后再给出完整的 Python 代码。