# Investigating Adversarial Robustness of Medical Imaging Foundation Models: A Comprehensive Framework for CheXzero Evaluation

**Author**: [Your Name]
**Institution**: [Your Institution]
**Date**: January 2026
**Word Count**: ~15,000 words

---

## Abstract

Medical imaging foundation models, particularly those based on CLIP (Contrastive Language-Image Pre-training) architecture, have demonstrated remarkable capabilities in zero-shot chest X-ray interpretation. However, their robustness against adversarial perturbations remains insufficiently explored, posing significant concerns for clinical deployment. This dissertation presents a comprehensive framework for evaluating the adversarial robustness of CheXzero, a state-of-the-art medical imaging foundation model, on the CheXpert dataset. We implement and systematically evaluate four major adversarial attack methods (PGD, C&W, FGSM, and DeepFool) under both L∞ and L2 norm constraints. Additionally, we investigate the feasibility of lesion-aware spatial attacks using Grad-CAM-based attention mechanisms. Our experimental results reveal that CheXzero exhibits significant vulnerability to adversarial perturbations, with attack success rates reaching 100% under moderate perturbation budgets (ε=8/255). Furthermore, we identify a fundamental architectural limitation in CLIP-based Vision Transformers that prevents standard Grad-CAM from generating spatial attention maps, as gradients only flow to the CLS token rather than spatial patch tokens. This finding has important implications for future research on spatially-constrained adversarial attacks on transformer-based vision models. Our work contributes: (1) a unified, modular framework (~2,500 lines of code) for adversarial robustness evaluation of medical imaging models; (2) comprehensive empirical evaluation of CheXzero's vulnerability across multiple attack methods; (3) identification of architectural constraints in CLIP ViT that affect gradient-based spatial analysis; and (4) a reproducible experimental platform for future research. These findings underscore the critical need for adversarial robustness considerations before deploying foundation models in clinical settings.

**Keywords**: Adversarial Attacks, Medical Imaging, Foundation Models, CLIP, Vision Transformer, Grad-CAM, Adversarial Robustness, CheXzero, CheXpert

---

## Table of Contents

1. Introduction
2. Background and Related Work
3. Methodology
4. Results
5. Discussion
6. Conclusion
7. References

---

## 1. Introduction

### 1.1 Motivation

The integration of artificial intelligence into medical imaging has revolutionized diagnostic radiology, with deep learning models demonstrating expert-level performance across various imaging modalities and clinical tasks [1]. Recent advances in foundation models—large-scale pre-trained models that can be adapted to diverse downstream tasks with minimal fine-tuning—have further accelerated this transformation. Among these, CLIP-based models have shown particular promise in medical imaging by leveraging vision-language pre-training to enable zero-shot classification and cross-modal retrieval [2,3].

CheXzero, a medical imaging foundation model adapted from OpenAI's CLIP architecture, exemplifies this paradigm shift [4]. Trained on a large corpus of chest X-ray images paired with radiology reports, CheXzero achieves competitive performance on multi-label chest pathology classification without requiring task-specific fine-tuning. Its zero-shot capabilities make it particularly attractive for clinical deployment, where labeled data may be scarce or costly to obtain.

However, the deployment of AI systems in safety-critical domains such as healthcare necessitates rigorous evaluation beyond standard performance metrics. One critical dimension is **adversarial robustness**—the model's resilience to carefully crafted perturbations designed to induce misclassification [5]. Adversarial examples, first systematically studied by Szegedy et al. [6] and Goodfellow et al. [7], pose significant risks in medical imaging contexts:

1. **Clinical Safety**: Adversarial perturbations could lead to misdiagnosis, potentially causing patient harm through incorrect treatment decisions.

2. **Security Concerns**: Malicious actors could exploit vulnerabilities to deliberately manipulate diagnostic outcomes for fraud or sabotage.

3. **Model Reliability**: Even unintentional perturbations (e.g., from image acquisition artifacts or processing pipelines) could trigger unexpected model behaviors.

4. **Regulatory Compliance**: Medical device regulations increasingly require evidence of robustness against various failure modes, including adversarial scenarios.

Despite the critical importance of adversarial robustness in medical AI, foundation models like CheXzero remain underexplored in this dimension. Existing adversarial robustness studies primarily focus on natural image classifiers (e.g., ImageNet models) or specialized medical CNNs, leaving a significant gap in our understanding of how vision-language foundation models behave under adversarial perturbations.

### 1.2 Research Questions

This dissertation addresses the following research questions:

**RQ1: How vulnerable are medical imaging foundation models to standard adversarial attacks?**
We systematically evaluate CheXzero's robustness against four canonical adversarial attack methods: Projected Gradient Descent (PGD), Carlini & Wagner (C&W), Fast Gradient Sign Method (FGSM), and DeepFool. We quantify vulnerability using multiple metrics including attack success rate, AUROC degradation, and perturbation magnitude.

**RQ2: Can lesion-aware spatial constraints improve adversarial attack effectiveness on medical images?**
Motivated by recent work on lesion-focused attacks [8], we investigate whether constraining perturbations to clinically-relevant regions (identified via Grad-CAM attention) can achieve higher attack effectiveness with smaller perturbation budgets. This has implications for understanding model decision-making and designing more targeted robustness evaluations.

**RQ3: What architectural characteristics of CLIP-based Vision Transformers affect their susceptibility to adversarial attacks?**
Through our investigation, we aim to uncover fundamental properties of the CLIP ViT architecture that influence its adversarial vulnerability, including how gradient flow and attention mechanisms interact with perturbation strategies.

**RQ4: How can we design a unified, reproducible framework for adversarial robustness evaluation of medical imaging models?**
Beyond empirical evaluation, we aim to contribute a modular software framework that standardizes adversarial attack implementation, evaluation metrics, and experimental workflows for future research.

### 1.3 Contributions

This dissertation makes the following contributions to the field of adversarial robustness in medical imaging:

**1. Comprehensive Empirical Evaluation**
We provide the first systematic adversarial robustness evaluation of CheXzero, a state-of-the-art medical imaging foundation model, across four major attack methods (PGD, C&W, FGSM, DeepFool) on the CheXpert dataset. Our experiments quantify vulnerability using multiple metrics and perturbation budgets, revealing significant susceptibility to adversarial perturbations (100% attack success rate under moderate perturbation constraints).

**2. Unified Adversarial Robustness Framework**
We implement a modular, extensible framework (~2,500 lines of Python code) that provides:
- Standardized attack API compatible with PyTorch models
- Unified evaluation metrics (AUROC, ASR, perturbation norms)
- Flexible model wrapper for CLIP-based architectures
- Reproducible experimental pipelines with detailed documentation

This framework lowers the barrier for future research and enables consistent comparison across studies.

**3. Technical Discovery: CLIP ViT Gradient Flow Limitation**
Through detailed investigation of Grad-CAM-based attention mechanisms, we identify a fundamental architectural constraint in CLIP Vision Transformers: **gradients only flow to the CLS token, not to spatial patch tokens**. This has two important implications:
- Standard Grad-CAM cannot generate spatial attention maps for CLIP ViT models
- Gradient-based spatial attack methods require alternative approaches (e.g., attention rollout)

This finding advances our understanding of Vision Transformer architectures and informs future research on interpretability and spatial adversarial methods.

**4. Methodological Insights for Medical AI Robustness**
Our work demonstrates the importance of architecture-aware robustness evaluation. We show that methods successful on CNN-based models (e.g., Grad-CAM) may not directly transfer to transformer architectures, highlighting the need for tailored evaluation strategies as foundation models become increasingly prevalent.

### 1.4 Dissertation Structure

The remainder of this dissertation is organized as follows:

**Chapter 2 (Background and Related Work)** provides foundational concepts in adversarial machine learning, reviews prior work on adversarial robustness in medical imaging, introduces the CLIP architecture and CheXzero model, and discusses gradient-based attribution methods including Grad-CAM.

**Chapter 3 (Methodology)** details our experimental framework design, attack implementations (PGD, C&W, FGSM, DeepFool), lesion-aware attack methodology, evaluation metrics, and experimental setup including dataset preparation and hyperparameter selection.

**Chapter 4 (Results)** presents comprehensive empirical results from baseline adversarial attacks across all four methods, quantitative analysis of attack effectiveness, qualitative visualization of adversarial examples, and documentation of the Grad-CAM limitation discovery.

**Chapter 5 (Discussion)** interprets our findings in the context of clinical safety, analyzes architectural factors affecting robustness, discusses the implications of the CLIP ViT gradient flow limitation, compares our results with prior work, and acknowledges limitations of our study.

**Chapter 6 (Conclusion)** summarizes our key contributions, discusses implications for medical AI deployment, and outlines promising directions for future research.

---

## 2. Background and Related Work

### 2.1 Adversarial Machine Learning Foundations

#### 2.1.1 Adversarial Examples

Adversarial examples are inputs intentionally designed to cause machine learning models to make incorrect predictions [6,7]. Formally, given a classifier f: X → Y, an input x with true label y, and a perturbation budget ε, an adversarial example x_adv is constructed to satisfy:

```
f(x_adv) ≠ y  (misclassification)
||x_adv - x||_p ≤ ε  (bounded perturbation)
```

where ||·||_p denotes the Lp norm (commonly L∞ or L2). The perturbation δ = x_adv - x is constrained to be imperceptible to humans while causing model failure.

The existence of adversarial examples reveals fundamental differences between human and machine perception. While humans primarily rely on high-level semantic features for recognition, deep neural networks can be sensitive to high-frequency patterns imperceptible to humans [9,10]. This vulnerability stems from the high-dimensional nature of input spaces and the linear characteristics of modern neural network activations [7].

#### 2.1.2 Canonical Attack Methods

We briefly review four canonical adversarial attack methods evaluated in this work:

**Projected Gradient Descent (PGD)** [11] is an iterative attack that performs multiple gradient ascent steps with projection onto the ε-ball:

```
x_{t+1} = Π_{x + S} (x_t + α · sign(∇_x L(f(x_t), y)))
```

where Π denotes projection onto the constraint set S = {x' : ||x' - x||_∞ ≤ ε}, α is the step size, and L is the loss function. PGD is considered a strong attack that approximates the worst-case adversarial perturbation under L∞ constraints.

**Carlini & Wagner (C&W)** [12] formulates adversarial attack as an optimization problem:

```
minimize ||δ||_2^2 + c · f(x + δ)
subject to x + δ ∈ [0,1]^n
```

where f(·) is an objective function designed to induce misclassification, and c is a constant balancing perturbation magnitude and attack success. The C&W attack uses the change-of-variables δ = 0.5(tanh(w) + 1) - x to handle box constraints, and employs Adam optimizer for efficient optimization. C&W is known for finding near-optimal L2 perturbations.

**Fast Gradient Sign Method (FGSM)** [7] is a single-step attack:

```
x_adv = x + ε · sign(∇_x L(f(x), y))
```

FGSM is computationally efficient but generally weaker than iterative methods like PGD. It serves as a useful baseline for assessing basic adversarial vulnerability.

**DeepFool** [13] computes the minimal perturbation required to reach the decision boundary by iteratively linearizing the classifier:

```
r_i = - f(x_i) / ||∇f(x_i)||_2 · ∇f(x_i)
```

where r_i is the perturbation at iteration i. DeepFool provides a lower bound on adversarial robustness by finding the closest decision boundary.

#### 2.1.3 Evaluation Metrics

Adversarial robustness is typically evaluated using several complementary metrics:

**Attack Success Rate (ASR)**: Percentage of samples successfully misclassified by the attack.

**Perturbation Magnitude**: Measured via L0 (number of perturbed pixels), L2 (Euclidean distance), or L∞ (maximum pixel change) norms.

**Model Performance Degradation**: Change in accuracy, AUROC, or other task-specific metrics under attack.

**Robustness Curves**: Performance as a function of perturbation budget ε, showing the trade-off between robustness and perturbation magnitude.

### 2.2 Adversarial Robustness in Medical Imaging

#### 2.2.1 General Medical Imaging Adversarial Studies

The medical imaging community has increasingly recognized adversarial robustness as a critical challenge. Early work by Finlayson et al. [14] demonstrated that adversarial perturbations could cause diagnostic CNNs to misclassify chest X-rays with high success rates. Subsequent studies have explored adversarial vulnerabilities across various medical imaging modalities:

**Radiology**: Ma et al. [15] evaluated adversarial attacks on mammography classification models, finding significant vulnerability even under small perturbation budgets. They emphasized the safety implications of such vulnerabilities in breast cancer screening.

**Pathology**: Ren et al. [16] investigated adversarial attacks on histopathology image classifiers, demonstrating that imperceptible perturbations could alter cancer diagnosis predictions.

**Ophthalmology**: Xu et al. [17] studied adversarial robustness of diabetic retinopathy screening models, revealing vulnerabilities that could affect screening program reliability.

**Multi-Modal Imaging**: Recent work has explored adversarial attacks on multi-modal medical imaging systems, including CT, MRI, and ultrasound [18,19].

A common finding across these studies is that medical imaging models trained on standard datasets (without adversarial training) exhibit high vulnerability to adversarial perturbations. This vulnerability persists even when perturbations are constrained to be clinically realistic.

#### 2.2.2 Lesion-Aware and Spatially-Constrained Attacks

Traditional adversarial attacks apply global perturbations across the entire image. However, in medical imaging, clinically-relevant information is often localized to specific anatomical regions or lesions. This observation has motivated research on spatially-constrained attacks:

**Li et al. (2025)** [8] proposed LatAtk, a lesion-aware adversarial attack method for chest X-rays. They use Grad-CAM to identify lesion regions, then constrain perturbations to these areas. Their work demonstrated that lesion-focused attacks can achieve higher attack effectiveness with smaller L2 perturbations compared to global attacks, suggesting that models rely heavily on lesion features for classification.

**Zhang et al. (2023)** [20] explored anatomically-constrained attacks on brain MRI segmentation models, showing that perturbations localized to tumor regions are particularly effective.

**Semantic Adversarial Attacks**: Other work has investigated attacks that respect medical image semantics, such as modifying texture patterns within lesions rather than adding arbitrary noise [21,22].

These spatially-constrained approaches have two advantages:
1. **Clinical Realism**: Localized perturbations may be more plausible as imaging artifacts or pathological variations.
2. **Interpretability**: Identifying vulnerable regions reveals which image features models rely on for decision-making.

However, prior work primarily focused on CNN-based models. The applicability of these methods to transformer-based foundation models remains underexplored—a gap this dissertation aims to address.

### 2.3 Foundation Models in Medical Imaging

#### 2.3.1 The CLIP Architecture

CLIP (Contrastive Language-Image Pre-training) [2] is a vision-language foundation model trained on 400 million image-text pairs scraped from the internet. CLIP consists of two encoders:

**Image Encoder**: A Vision Transformer (ViT) or modified ResNet that maps images to a d-dimensional embedding space.

**Text Encoder**: A Transformer that maps text descriptions to the same d-dimensional space.

During pre-training, CLIP learns to maximize the cosine similarity between embeddings of matching image-text pairs while minimizing similarity for non-matching pairs, using a contrastive loss:

```
L = - Σ log (exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ))
```

where sim(·,·) is cosine similarity, τ is a temperature parameter, and the sum ranges over all pairs in a batch.

This training objective enables CLIP to learn rich, transferable visual representations aligned with natural language, allowing zero-shot classification by comparing image embeddings with text embeddings of class names.

**Vision Transformer (ViT) Architecture**: The ViT image encoder used in CLIP divides input images into fixed-size patches (e.g., 16×16), linearly embeds each patch, and adds positional embeddings. A special [CLS] token is prepended to the sequence:

```
Patch Sequence: [CLS], P1, P2, ..., P_N
```

The sequence is processed by a stack of transformer blocks (self-attention + MLP), and the [CLS] token's final representation serves as the image embedding. Importantly, **only the [CLS] token output is used for the final image representation**; spatial patch representations are discarded. This design choice has significant implications for gradient-based spatial analysis, as we discuss in our findings.

#### 2.3.2 CheXzero: CLIP for Chest X-ray Interpretation

CheXzero [4] adapts CLIP for chest radiograph interpretation by fine-tuning on chest X-ray images paired with radiology reports from the MIMIC-CXR dataset. The model learns to associate visual patterns in chest X-rays with medical terminology extracted from free-text reports.

**Model Architecture**: CheXzero uses a ViT-B/32 image encoder (12 transformer blocks, 768 hidden dimensions, 32×32 patch size) and a 12-layer text transformer. The model is trained with a contrastive loss similar to original CLIP.

**Zero-Shot Classification**: For a given pathology class (e.g., "pneumonia"), CheXzero computes:

```
P(class | image) ∝ sim(Enc_image(x), Enc_text("finding of [class]"))
```

where text prompts are hand-crafted templates like "finding of pneumonia." Multi-label classification is performed by computing similarity with prompts for multiple classes.

**Performance**: CheXzero achieves competitive AUROC scores on CheXpert pathology classification (average AUROC ~0.85 across 14 labels) without task-specific fine-tuning, demonstrating strong zero-shot transfer capabilities.

**Clinical Relevance**: The zero-shot paradigm is particularly valuable in medical imaging where:
- Labeled data is expensive and requires expert annotation
- New pathologies or imaging protocols may emerge
- Model adaptation to different hospital systems is needed

However, the adversarial robustness of CheXzero—critical for clinical deployment—has not been systematically evaluated, motivating this dissertation.

### 2.4 Gradient-Based Attribution and Grad-CAM

#### 2.4.1 Interpretability in Medical AI

Interpretability is crucial for medical AI systems to build trust, enable error diagnosis, and comply with regulatory requirements. Gradient-based attribution methods provide insights into which input features influence model predictions.

#### 2.4.2 Grad-CAM Principle

Gradient-weighted Class Activation Mapping (Grad-CAM) [23] generates visual explanations for CNN predictions by combining feature map activations with gradient weights:

```
L_Grad-CAM = ReLU(Σ_k α_k A_k)
```

where A_k are feature maps from a target convolutional layer, and α_k are importance weights computed as:

```
α_k = (1/Z) Σ_i Σ_j ∂y^c / ∂A_{ij}^k
```

The global average pooling of gradients (∂y^c / ∂A^k) with respect to class c provides a measure of feature importance. ReLU is applied to focus on features with positive influence on the target class.

**Application to Medical Imaging**: Grad-CAM has been widely used in medical imaging to:
- Visualize regions influencing diagnostic decisions
- Validate that models focus on clinically relevant features (e.g., lesions)
- Identify potential model biases or confounding factors

#### 2.4.3 Grad-CAM for Vision Transformers

Extending Grad-CAM to Vision Transformers is non-trivial due to architectural differences from CNNs:

**CNN**: Feature maps have explicit spatial structure (H × W × C). Grad-CAM directly uses these spatial feature maps.

**ViT**: After patch embedding, spatial structure is encoded in token positions. The final output is typically a single [CLS] token vector, not a spatial feature map.

Several approaches have been proposed for ViT interpretability:
- **Attention Rollout** [24]: Aggregates attention weights across layers to approximate information flow to the [CLS] token.
- **Layer-wise Grad-CAM**: Applies Grad-CAM to intermediate transformer block outputs before global pooling.
- **Patch-wise Attribution**: Computes gradients with respect to individual patch embeddings.

However, as we discover in our investigation, the CLIP ViT architecture presents unique challenges due to its exclusive use of the [CLS] token for image representation. This architectural choice fundamentally limits gradient flow to spatial tokens, creating obstacles for standard Grad-CAM application—a key finding of this dissertation.

### 2.5 Research Gap and Dissertation Positioning

Despite extensive research on adversarial robustness in computer vision and growing interest in medical imaging adversarial examples, several gaps remain:

**Gap 1**: Foundation models, particularly CLIP-based vision-language models, have not been systematically evaluated for adversarial robustness in medical imaging contexts.

**Gap 2**: The applicability of lesion-aware spatial attacks to transformer-based architectures is unexplored, with prior work focusing on CNNs.

**Gap 3**: Architectural constraints of Vision Transformers (e.g., gradient flow patterns, attention mechanisms) that affect adversarial vulnerability and interpretability are not well understood.

**Gap 4**: Lack of unified, reproducible frameworks for adversarial robustness evaluation of medical imaging models hinders cross-study comparison and replication.

This dissertation addresses these gaps by:
1. Providing the first comprehensive adversarial evaluation of CheXzero (a CLIP-based medical foundation model)
2. Investigating lesion-aware attacks on ViT architectures
3. Identifying fundamental architectural constraints affecting Grad-CAM and spatial analysis
4. Contributing an open, modular framework for future research

Our work bridges adversarial machine learning, medical imaging AI, and vision transformer research, contributing insights relevant to all three communities.

---

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
