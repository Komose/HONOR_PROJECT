# Bug修复方案：Random Patch Mask未应用

## Bug描述

`rsna_attack_framework.py`中的FGSM和PGD函数只检查`attack_mode == 'lesion'`，导致`attack_mode == 'random_patch'`时不应用mask，变成全图攻击。

## 受影响的文件

1. `rsna_attack_framework.py`:
   - `fgsm_attack()` (line 181, 192)
   - `pgd_attack()` (需检查)

## 修复代码

### 修复1: FGSM (line 181)
```python
# Before
if attack_mode == 'lesion':
    grad = grad * masks

# After
if attack_mode != 'full':  # lesion OR random_patch
    grad = grad * masks
```

### 修复2: FGSM (line 192)
```python
# Before
if attack_mode == 'lesion':
    adv_images = images * (1 - masks) + adv_images * masks

# After
if attack_mode != 'full':
    adv_images = images * (1 - masks) + adv_images * masks
```

### 修复3: PGD (需检查类似位置)
查找PGD函数中所有`if attack_mode == 'lesion'`，改为`if attack_mode != 'full'`

## 修复后需要重新运行的实验

1. ✗ epsilon sweep (ε=1,2,4,16/255) - FGSM & PGD
2. ✗ epsilon extreme (ε=32/255) - FGSM & PGD
3. ✓ C&W & DeepFool - 无需重跑（使用不同代码路径）

## 预期修复后的结果

### 如果语义脆弱性假说在L∞下也成立：
- Random Patch (等面积) ASR ≤ Lesion ASR
- 推翻当前的"L∞悖论"

### 如果L∞悖论真实存在（面积效应更强）：
- Random Patch (等面积) ASR仍可能≈ Lesion ASR
- 因为等面积时，两者使用相同的L2能量
- 但Full ASR > Random/Lesion（面积优势）

## 估计重跑时间

- FGSM: ~1-2小时（所有ε值）
- PGD: ~3-5小时（所有ε值）
- 总计：4-7小时

## 修复优先级

**HIGH PRIORITY** - 这是实验有效性的核心问题

建议：
1. 立即修复代码
2. 重跑ε=4/255 和 ε=16/255（代表性ε值）
3. 验证Random Patch L0 ≈ Lesion L0
4. 如果验证通过，重跑全部ε sweep

---

END OF FIX PLAN
