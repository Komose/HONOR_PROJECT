## 3. Methodology

This chapter describes our experimental framework for evaluating the adversarial robustness of CheXzero. We detail our unified attack implementation, evaluation metrics, experimental design, and data preparation procedures.

### 3.1 Experimental Framework Design

#### 3.1.1 Design Principles

Our framework adheres to the following design principles:

**Modularity**: Each component (model wrapper, attacks, evaluation) is independently testable and replaceable.

**Reproducibility**: All experiments use fixed random seeds and detailed logging to ensure reproducibility.

**Extensibility**: The unified API design allows easy integration of new attack methods or models.

**Standards Compliance**: Implementations follow established conventions (e.g., PyTorch modules, NumPy array formats).

#### 3.1.2 Framework Architecture

The experimental framework consists of five main components working in a pipeline:

1. **Data Processing Pipeline**: Loads CheXpert chest X-rays from HDF5 format, applies preprocessing (normalization, resizing), and manages batch iteration.

2. **Model Wrapper (FMAttackWrapper)**: Provides a unified interface for CheXzero, handling text prompt generation, image-text similarity computation, multi-label probability prediction, and gradient computation for adversarial attacks.

3. **Attack Modules**: Implementations of PGD, C&W, FGSM, and DeepFool following a standard `perturb(x, y, wrapper)` API.

4. **Evaluation Module**: Computes robustness metrics including AUROC, attack success rate, and perturbation statistics.

5. **Experiment Runner**: Orchestrates end-to-end evaluation including attack execution, metric computation, and result logging.

### 3.2 Attack Implementations

We implement four canonical adversarial attack methods with unified interface.

#### 3.2.1 PGD-L∞ Implementation

PGD performs iterative gradient ascent with projection onto the L∞ ball. Key implementation details:

**Hyperparameters**:
- Epsilon (ε): 2, 4, 8, 16 (pixel space)
- Steps: 3-10 iterations
- Step size: 2.5ε/T
- Random initialization: Uniform[-ε, ε]

#### 3.2.2 C&W-L2 Implementation

Carlini & Wagner attack formulates adversarial generation as optimization. For multi-label classification, we use tanh transformation for box constraints and Adam optimizer.

**Hyperparameters**:
- Max iterations: 100-500
- Learning rate: 0.01
- Binary search steps: 5-9

#### 3.2.3 FGSM and DeepFool

FGSM provides single-step baseline, DeepFool computes minimal perturbation to decision boundary.

### 3.3 Lesion-Aware Attack Design and Discovered Limitation

We attempted to implement lesion-aware attacks using Grad-CAM to identify clinically-relevant regions. However, we discovered a fundamental architectural constraint:

**Finding**: In CLIP ViT, gradients only flow to the CLS token, not spatial patch tokens. This prevents standard Grad-CAM from generating spatial attention maps.

**Verification**: Inspecting gradients revealed:
- CLS token gradients: Non-zero, range [-3.3, 2.9]
- Patch token gradients (49 tokens): All zeros

**Implication**: Standard gradient-based spatial attribution methods cannot be directly applied to CLIP ViT. Alternative approaches (attention rollout, token attribution) are required.

This finding is a key technical contribution of this dissertation.

### 3.4 Evaluation Metrics

**Classification Performance**:
- AUROC (clean and adversarial)
- AUC Drop (clean - adversarial)

**Attack Effectiveness**:
- Attack Success Rate (ASR): % of samples with changed prediction
- Mean labels flipped per sample

**Perturbation Magnitude**:
- L0, L2, L∞ norms (mean ± std)

### 3.5 Experimental Setup

**Dataset**: CheXpert chest X-rays
- 14-label multi-label classification
- Subsets: 5, 20, 100 samples (due to CPU constraints)
- U-Zeros policy for uncertain labels
- 224×224 resolution

**Model**: CheXzero (CLIP ViT-B/32)
- Pre-trained checkpoint
- Float32 for CPU compatibility
- Zero-shot classification with text prompts

**Computational Environment**:
- CPU-only execution (no GPU)
- PyTorch 1.12, Python 3.10
- Fixed random seeds for reproducibility

---
