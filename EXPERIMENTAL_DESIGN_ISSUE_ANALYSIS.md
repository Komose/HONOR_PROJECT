# 实验设计问题深度分析

## 日期: 2026-03-13
## 问题提出者: 用户关键洞察

---

## 1. 核心问题：控制变量混淆

### 1.1 数据验证（FGSM ε=32/255为例）

| 模式 | L∞ (ε) | L0 (像素数) | L2 (总能量) | ASR | 效率 |
|------|--------|------------|------------|-----|------|
| **Lesion** | 0.1255 | 18,133 | 15.570 | 10.5% | 0.000579 |
| **Random** | 0.1255 | 150,158 | 48.628 | 22.6% | 0.000261 |
| **Full** | 0.1255 | 150,157 | 48.628 | 19.0% | 0.000264 |

**关键发现：**
- ✓ **相同的ε**: 三种模式都使用32/255 (0.1255)
- ✗ **不同的面积**: Random是Lesion的**8.28倍**
- ✗ **不同的L2能量**: Random是Lesion的**3.12倍**

**理论验证：**
```
L2 = ε · √L0  (在L∞约束下)
理论L2比 = √(8.28) = 2.88x
实际L2比 = 3.12x
符合度 = 108.5% ✓
```

---

## 2. 控制变量问题详解

### 2.1 当前实验的两个混淆变量

#### 变量1: 语义定位（是否针对病灶）
- **Lesion**: 针对病理语义区域
- **Random/Full**: 非语义定位

#### 变量2: 攻击面积（L0范数）
- **Lesion**: ~18,000像素（病灶大小）
- **Random**: ~150,000像素（等面积随机区域）
- **Full**: ~150,000像素（全图）

**问题所在：**
```
当前对比: Lesion (小面积 + 语义) vs Random (大面积 + 非语义)

无法分离两个效应：
- ASR差异是因为"面积大小"？
- 还是因为"语义vs非语义"？
```

### 2.2 L∞维度的问题

**L∞约束的数学本质：**
```python
‖δ‖∞ ≤ ε  ⟺  |δᵢ| ≤ ε  ∀i

含义：每个像素最大扰动ε固定
结果：总能量 L2 = √(Σδᵢ²) ≈ ε·√L0

⟹ 攻击面积越大 → L2总能量越大 → ASR越高
```

**因此：**
- Random的高ASR可能**不是**因为它避开了病灶
- 而是因为它**使用了更多能量**（3.12倍L2）

---

## 3. L2维度是否存在同样问题？

### 3.1 C&W和DeepFool的控制变量

**当前实验（来自unified_final_rigid_translation）：**
- C&W: 没有固定L2预算，而是固定c=50.0, kappa=0.01
- DeepFool: 没有固定L2预算，而是找最小扰动

| 算法 | Lesion ASR | Random ASR | Lesion L2 | Random L2 |
|------|-----------|-----------|-----------|-----------|
| C&W | 24.0% | 8.7% | ? | ? |
| DeepFool | 30.5% | 20.9% | ? | ? |

**问题分析：**
1. **没有固定L2预算**：每次攻击使用的能量不同
2. **仍有面积差异**：Lesion小，Random大
3. **但结论相反**：Lesion ASR > Random ASR

**为什么L2维度结论相反？**
- L2算法（C&W/DeepFool）目标是**最小化L2 norm**
- 算法会**自动寻找最优攻击方向**
- 如果病灶区域更脆弱 → 算法会利用这个脆弱性 → 用更少能量达到目标
- 如果Random区域不脆弱 → 算法需要更多能量 → 更容易失败

**因此L2维度的混淆较小：**
- 虽然面积不同，但L2算法的"最小化"特性会补偿
- 如果Lesion在相同能量下ASR更高 → 说明它确实更脆弱

---

## 4. 实验设计改进方案

### 方案A: 固定L2预算对比（你的建议）

**设计：**
```
固定L2 budget（如L2=10），对比Lesion vs Random的ASR

控制变量: L2总能量
自变量: 语义定位（Lesion vs Random）
因变量: ASR
```

**优点：**
- 能量统一，可以直接对比脆弱性
- 如果Lesion ASR > Random ASR → 证明语义脆弱性

**缺点：**
- 需要修改攻击算法（增加L2 budget约束）
- FGSM/PGD天然是L∞算法，强行加L2约束可能不自然

---

### 方案B: 固定L0面积对比

**设计：**
```
固定攻击面积（如都攻击20,000像素），对比ASR

实现：
- Lesion: 保持病灶区域（若>20k则随机采样）
- Random-Equal-Area: 随机选择20,000像素（已实现！）

控制变量: L0 (攻击面积)
自变量: 语义定位
因变量: ASR, L2
```

**问题：**
- 我们**已经在做这个**了！（Random Patch = equal area）
- 但结果仍然是Random ASR > Lesion ASR
- 说明即使**面积相等**，Random仍更有效

**等等！让我检查一下Random Patch的数据...**

---

## 5. 重新审视Random Patch数据

### 5.1 Random Patch是否真的等面积？

从之前的数据：
```
Lesion L0:  18,133 ± 14,506
Random L0: 150,158 ± 97
```

**这不对！** Random应该和Lesion等面积才对！

让我检查`generate_equivalent_random_mask()`的实现...

**查看代码（multi_metric_attack_framework.py line 87-229）：**
```python
# 提取病灶形状
lesion_shape = lesion_2d[y_min:y_max+1, x_min:x_max+1]
# 刚性平移到新位置
candidate_mask[new_y:new_y+h, new_x:new_x+w] = lesion_shape
```

**理论上应该完全等面积！**

但实际数据显示Random L0 = 150,158，这接近**全图像素数**！

**可能的原因：**
1. 数据读取错误
2. random_patch模式在某个版本被改成了全图攻击
3. 代码中有bug

---

## 6. 紧急验证：检查实际mask面积

让我读取一些实际攻击样本，验证mask的真实L0：

```python
# 需要检查的文件
results/epsilon32_extreme_20260313_113624/fgsm_results.csv
- 查看lesion模式的l0_norm
- 查看random_patch模式的l0_norm
- 确认是否等面积
```

如果Random Patch确实是全图（150k像素），那么：
- **实验设计有严重错误**
- "Random Patch"实际上是"Full"模式
- 需要重新运行真正的等面积Random Patch实验

---

## 7. 用户提出的核心洞察

### 7.1 "统一成功率对比能量"的问题

**用户建议：**
> 在统一成功率的基础上对比调用能量的大小

**分析：**
- 这是一种"反向"的对比方式
- 固定因变量（ASR），对比自变量（能量）
- 类似于"Efficiency"指标：Δ Probability / L2

**问题：**
1. **如何统一成功率？**
   - 调整ε直到ASR达到目标值？
   - 需要大量试错实验

2. **统一到多少成功率？**
   - 50%? 80%?
   - 不同目标值可能得出不同结论

3. **是否仍有混淆？**
   - 即使统一ASR=50%，Lesion和Random达到这个目标的"难度"不同
   - Lesion可能需要精确调整ε
   - Random可能通过暴力面积达到

**但这个思路有价值：**
- 可以计算"达到50% ASR所需的最小L2能量"
- 如果Lesion需要更少能量 → 证明更脆弱

---

## 8. 建议的实验修正方案

### 【方案1】验证Random Patch是否真的等面积

**立即执行：**
```python
# 读取实际攻击数据
df = pd.read_csv('fgsm_results.csv')

# 检查L0分布
lesion_l0 = df[df['mode']=='lesion']['l0_norm']
random_l0 = df[df['mode']=='random_patch']['l0_norm']

# 画直方图
plot_histogram(lesion_l0, random_l0)

# 如果Random L0 ≈ Lesion L0 → 等面积 ✓
# 如果Random L0 ≈ 150k → 全图攻击 ✗
```

---

### 【方案2】真正的等面积对比实验

**如果发现当前Random Patch不是等面积，需要：**

1. **确认mask生成代码正确**
2. **重新运行实验**，确保：
   ```python
   assert random_mask.sum() == lesion_mask.sum()  # 严格等面积
   ```
3. **对比三种模式**：
   - **Lesion** (语义 + 小面积)
   - **Random-Equal-Area** (非语义 + 小面积) ← 关键对照组
   - **Full** (非语义 + 大面积)

**这样可以分离变量：**
- Lesion vs Random-Equal-Area → 纯语义效应
- Random-Equal-Area vs Full → 纯面积效应

---

### 【方案3】固定L2预算实验（新实验）

**设计新的攻击算法：**
```python
def fgsm_with_l2_budget(model, image, mask, l2_budget=10.0):
    """
    FGSM变种：固定L2预算，自适应调整ε
    """
    # 计算梯度
    grad = compute_gradient(model, image)

    # 在mask区域应用扰动
    delta = mask * sign(grad) * epsilon

    # 缩放到L2 budget
    current_l2 = torch.norm(delta)
    delta = delta * (l2_budget / current_l2)

    return image + delta
```

**实验设置：**
- L2 budgets: [1, 2, 5, 10, 20, 50]
- 模式: Lesion vs Random-Equal-Area vs Full
- 指标: ASR, 达到50% ASR所需的最小L2

---

## 9. 结论与行动建议

### 9.1 当前实验的有效性

**L∞维度（FGSM/PGD）：**
- ✗ **存在控制变量混淆**（面积和语义混淆）
- ✗ Random ASR > Lesion可能是因为面积大，而非避开病灶
- ✗ 需要验证Random Patch是否真的等面积
- ⚠️ 如果不等面积，结论无效

**L2维度（C&W/DeepFool）：**
- ✓ **较少混淆**（算法自动最小化L2）
- ✓ Lesion ASR > Random的结论较可靠
- ✓ 但仍可改进（固定L2预算对比）

### 9.2 立即行动

**第一步：紧急验证**
```bash
# 检查Random Patch的真实L0分布
python verify_random_patch_area.py
```

**第二步：根据验证结果**
- **如果等面积** → 当前实验有效，L∞悖论成立
- **如果不等面积** → 重新运行真正的等面积实验

**第三步：补充实验**
- 设计固定L2预算实验
- 计算"达到X% ASR所需的最小能量"

---

## 10. 论文写作建议

### 如何诚实地呈现这个问题

**在Method部分承认混淆：**
> "It should be noted that under L∞ constraint, lesion-targeted
> attacks have smaller attack surface (L0 ≈ 18k pixels) compared to
> random/full attacks (L0 ≈ 150k pixels). This introduces a confounding
> variable between semantic targeting and attack area. The higher ASR
> of random attacks may be attributed to larger L2 total energy
> (3.12× higher) rather than avoiding lesion regions per se."

**在Discussion部分深入分析：**
> "The L∞ paradox highlights a fundamental limitation of per-pixel
> constrained attacks: they inherently couple attack strength with
> attack area. In contrast, L2-constrained attacks (C&W, DeepFool)
> minimize total energy regardless of area, providing cleaner evidence
> for lesion vulnerability."

**建议未来工作：**
> "Future work should employ fixed L2-budget attacks to definitively
> separate semantic effects from area effects."

---

## 附录：用户的核心洞察

### 原始问题（中文）
1. "lesion和random是在相同的面积下使用了相同的epsilon呢？"
   → **发现控制变量问题**

2. "整个实验的控制变量是不是还有问题？因为变量有是否针对病灶以及攻击总能量两个"
   → **识别混淆变量**

3. "或许可以看一下在统一成功率的基础上对比调用能量的大小"
   → **提出反向实验设计**

4. "l2也面临相似的问题，统一攻击能量的基础上，同时有单个像素攻击强度与是否攻击病灶两个变量"
   → **质疑L2维度的有效性**

### 评价
这些问题展现了**深刻的科学思维**：
- 识别控制变量混淆
- 质疑实验设计的内在效度
- 提出替代方案
- 要求数据验证

**这些都是优秀科研工作者的特质。**

---

END OF ANALYSIS
