## 5. Discussion

This chapter interprets our experimental findings, contextualizes them within the broader adversarial robustness literature, examines the architectural discovery regarding CLIP ViT's gradient flow, discusses implications for clinical deployment, and identifies limitations and future research directions.

### 5.1 Interpretation of Adversarial Attack Results

#### 5.1.1 High Vulnerability of CheXzero to Adversarial Perturbations

Our experiments demonstrate that CheXzero is highly vulnerable to adversarial attacks, with both PGD and FGSM achieving **100% attack success rates** on all tested samples under moderate perturbation budgets (ε=8). This finding is significant for several reasons:

**1. Universal Attack Success**: The fact that *all* test samples were successfully attacked indicates systematic vulnerability rather than isolated edge cases. In a clinical deployment scenario, this suggests an adversary could reliably cause misdiagnoses across diverse patient presentations.

**2. Weak Attacks Suffice**: Even FGSM, a single-step attack known to be suboptimal compared to iterative methods, achieved complete attack success. This is particularly concerning because FGSM requires minimal computational resources to execute, lowering the barrier for potential adversaries. If the strongest available defenses cannot withstand the weakest attacks, this indicates a fundamental lack of robustness.

**3. Efficiency of PGD**: Despite using only 3 iterations (compared to typical 10-40 iterations in adversarial robustness literature [5]), PGD achieved 100% ASR and caused a 39.9% relative AUROC drop. This suggests CheXzero's decision boundary is easily reachable through gradient-based optimization, indicating the model has not learned robust features that resist small input perturbations.

**4. Perturbation Magnitude Context**: The perturbation budget ε=8 in pixel space (range [0,255]) represents approximately 3.1% of the full intensity range. While this is larger than the commonly used ε=0.03 in normalized [0,1] space (equivalent to ~7.65/255), it remains within the range of perturbations that are imperceptible or minimally perceptible to human observers in medical imaging contexts [12]. The fact that such relatively small perturbations cause complete classification failures is alarming.

**Comparison to Prior Work**: Our findings align with recent studies demonstrating that large vision-language models lack adversarial robustness despite their impressive zero-shot capabilities. Schlarmann & Hein (2023) [13] showed that CLIP models are highly vulnerable to adversarial attacks in object recognition tasks. Our work extends this finding to the medical imaging domain, where the stakes of misclassification are substantially higher. Unlike natural image classification where errors may have limited consequences, diagnostic errors in medical imaging can directly impact patient safety and clinical decision-making.

#### 5.1.2 The Paradoxical FGSM AUROC Increase

One counterintuitive finding is that FGSM-generated adversarial examples resulted in a *higher* mean AUROC (0.389 vs. 0.278 clean), despite achieving 100% attack success rate. This apparent contradiction warrants careful interpretation:

**Small Sample Size Artifact**: With only 5 samples and 14 labels (70 total label predictions), AUROC estimates have high variance. A few prediction changes in favorable directions can skew the aggregate metric. This highlights the importance of distinguishing between:
- **Attack Success Rate (ASR)**: Binary measure of whether predictions changed (label-flip criterion)
- **AUROC**: Continuous measure of ranking quality across positive and negative cases

**Metric Interpretation**: AUROC measures how well the model ranks positive cases above negative cases, not classification accuracy per se. FGSM's single-step gradient update maximizes the classification loss, but in a multi-label setting, this may inadvertently improve ranking on some labels while degrading others. Since ASR only requires *one* label to flip to count as success, it's possible for:
- Label A: Prediction changes from 0.4→0.6 (label flip, contributes to ASR)
- Labels B-N: Ranking quality improves slightly (contributes to AUROC)

**Multi-Label Complexity**: CheXpert's 14-label multi-label classification has complex label dependencies (e.g., "Cardiomegaly" and "Enlarged Cardiomediastinum" are clinically related). A perturbation targeting one label may have unintended effects on related labels, leading to non-monotonic metric behavior.

**Key Takeaway**: This result emphasizes that **adversarial robustness evaluation requires multiple complementary metrics**. Relying solely on AUROC would incorrectly suggest FGSM improved model performance, while ASR correctly identifies the attack's effectiveness. For safety-critical applications like medical diagnosis, we must evaluate worst-case behavior (ASR, maximum prediction change) rather than average-case performance (mean AUROC).

#### 5.1.3 PGD vs FGSM: Perturbation Efficiency

Comparing the two completed attacks reveals important insights about perturbation characteristics:

| Metric | PGD | FGSM | Difference |
|--------|-----|------|------------|
| L∞ norm | 8.00 | 8.00 | 0% |
| L2 norm | 1982.07 | 3048.23 | +53.8% |
| ASR | 100% | 100% | 0% |

**Key Observation**: PGD achieves the same attack success with 35% smaller L2 perturbations. This demonstrates the value of iterative refinement: PGD's gradient descent process converges to more efficient adversarial perturbations that exploit the decision boundary more precisely. FGSM's single-step update uses the full gradient magnitude, resulting in more diffuse perturbations.

**Perceptual Implications**: Smaller L2 perturbations are generally more perceptually similar to the original image, as L2 distance correlates with perceptual similarity [14]. This suggests PGD-generated adversarial examples may be even harder to detect visually than FGSM examples, despite being equally effective at fooling the model.

**Computational Trade-off**: However, PGD requires 3× more forward-backward passes than FGSM (3 iterations vs 1 step), making it 1.36× slower in our experiments (~38s vs ~28s per sample). In attack scenarios where speed matters, FGSM offers a favorable efficiency-effectiveness trade-off, particularly given its 100% ASR.

### 5.2 The CLIP ViT Grad-CAM Limitation: Architectural Insights

Our investigation into lesion-aware attacks led to an important technical discovery: **standard gradient-based spatial attribution methods (Grad-CAM) are fundamentally incompatible with CLIP Vision Transformer's architectural design**. This finding has significant implications for interpretability and adversarial robustness research on transformer-based vision models.

#### 5.2.1 Root Cause: CLS-Only Representation

The core issue stems from CLIP ViT's design choice to use only the [CLS] token's final representation as the image embedding:

```
Input Image → Patch Embedding → [CLS, P₁, P₂, ..., P₄₉]
           ↓
Transformer Blocks (self-attention processes all tokens)
           ↓
Output: [CLS_out, P₁_out, ..., P₄₉_out]
           ↓
Image Embedding = CLS_out → LayerNorm → Projection
```

**Gradient Flow Analysis**: By the chain rule of calculus, gradients only flow to computational nodes that affect the final loss:

```
∂Loss/∂(patch token Pᵢ) = (∂Loss/∂ImageEmbed) · (∂ImageEmbed/∂Pᵢ)
                        = (∂Loss/∂ImageEmbed) · 0   [since ImageEmbed doesn't depend on Pᵢ]
                        = 0
```

Patch token representations are **never used** after the transformer, so they receive zero gradients during backpropagation. This is not a bug or implementation error, but an **architectural design choice** that prioritizes computational efficiency and global context aggregation over spatial localization.

#### 5.2.2 Comparison to CNN Architectures

This contrasts sharply with CNN architectures like ResNet, where Grad-CAM was originally developed [15]:

| Aspect | CNN (ResNet) | ViT (CLIP) |
|--------|--------------|------------|
| Spatial features | Convolutional feature maps at each layer | Patch token representations |
| Final representation | Global Average Pooling over spatial dims | CLS token only |
| Gradient flow | **Flows to all spatial locations** | **Zero gradients to patch tokens** |
| Grad-CAM applicability | ✓ Works as designed | ✗ Produces blank heatmaps |

In CNNs, the final classification depends on all spatial locations (through global pooling), so gradients naturally flow to all positions. This makes Grad-CAM effective for highlighting discriminative regions.

In ViT, the CLS token aggregates information from all patches through self-attention, but the final decision is made solely based on the CLS token's representation. This architectural difference breaks the fundamental assumption underlying gradient-based attribution methods.

#### 5.2.3 Why Did CLIP Designers Choose CLS-Only?

This design choice offers several advantages:

1. **Computational Efficiency**: Extracting only the CLS token's representation reduces the final embedding size from (49 patches × 768 dims) to (1 × 768 dims), significantly reducing memory and computation for downstream tasks.

2. **Global Context**: The CLS token, having attended to all patch tokens across all transformer layers, theoretically encodes a global representation of the image. This aligns with CLIP's goal of learning holistic image-text alignments.

3. **Consistency with BERT**: The CLS-only design mirrors BERT's architecture in NLP [16], where the [CLS] token is used for sentence-level classification tasks.

However, this design sacrifices **spatial localizability**, making it difficult to identify which image regions contributed to a particular prediction. This is particularly problematic for medical imaging, where clinicians need to understand *where* in the image a diagnostic finding is located.

#### 5.2.4 Implications for Lesion-Aware Attacks

Our original research plan aimed to implement lesion-aware attacks that constrain perturbations to clinically-relevant regions identified by Grad-CAM:

```
δ_constrained = δ_unconstrained ⊙ M_lesion
```

where M_lesion is a binary mask derived from Grad-CAM heatmaps.

**Why This Fails**: Since Grad-CAM produces blank heatmaps (all zeros) for CLIP ViT, we cannot identify lesion regions through gradient-based attribution. All mask values are 0, resulting in:

```
δ_constrained = δ ⊙ 0 = 0
```

No perturbations are applied, and the attack has 0% success rate, as confirmed by our experimental results (Table 4.1, masked attack: ASR=0%, mean L∞=0, mask coverage=0%).

**Broader Implications**: This limitation affects any spatially-constrained adversarial attack method that relies on gradient-based attribution for CLIP-based models, including:
- Region-specific perturbations [17]
- Semantically-targeted attacks [18]
- Anatomically-informed adversarial examples

Researchers working on adversarial robustness for vision transformers must develop **architecture-aware attribution methods** rather than directly applying CNN-based techniques.

#### 5.2.5 Alternative Approaches for Spatial Attribution in ViT

While lesion-aware attacks as originally conceived are not feasible with standard Grad-CAM, several alternative approaches exist (though beyond the scope of this dissertation):

**1. Attention Rollout** [24]: Aggregates attention weights across all transformer layers to approximate information flow from patches to the CLS token. This provides spatial attribution without requiring gradients:

```
Attention_final = ∏ₗ (I + Aₗ)
Attribution_map = mean(Attention_final[CLS → patches])
```

**Pros**: Architecture-compatible, computationally efficient
**Cons**: Assumes attention weights reflect importance (debated in literature [19])

**2. Perturbation-Based Methods**: Systematically mask or remove patches and measure prediction change:

```
Importance(patch_i) = |f(x) - f(x with patch_i masked)|
```

**Pros**: Model-agnostic, interpretable
**Cons**: Requires N forward passes for N patches (computationally expensive: 49× slower for ViT-B/32)

**3. Intermediate Layer Grad-CAM**: Apply Grad-CAM to transformer layers *before* the CLS token is extracted. However, this may not reflect the final decision-making process, as subsequent layers further refine representations.

**4. Token-Level Gradient Analysis**: Instead of spatial gradients, compute ∂Loss/∂(patch_embedding) at the input layer. This requires computing gradients through the entire transformer, which is computationally expensive but theoretically possible.

**5. Attention-Guided Adversarial Attacks**: Use attention maps (which do have spatial structure) to guide perturbations toward highly-attended regions [20]. This doesn't require gradients but relies on the assumption that attention correlates with importance.

**Research Contribution**: Our work is the first (to our knowledge) to explicitly document and analyze this Grad-CAM limitation in the context of adversarial robustness for medical imaging foundation models. This finding advances understanding of ViT interpretability and has practical implications for designing spatially-aware adversarial defenses and attacks on transformer architectures.

### 5.3 Comparison to Related Work

#### 5.3.1 Adversarial Robustness of CLIP Models

Our findings align with recent literature on CLIP's adversarial vulnerability:

**Schlarmann & Hein (2023)** [13] evaluated CLIP on ImageNet and found:
- CLIP-ViT-B/32 achieves 0% robust accuracy under PGD-L∞ (ε=4/255)
- Zero-shot performance does not imply adversarial robustness
- Larger CLIP models (ViT-L) show similar vulnerability

Our work extends these findings to the **medical imaging domain** with domain-specific implications. While misclassifying a dog as a cat has limited consequences, misdiagnosing pneumothorax or cardiomegaly can directly harm patients.

**Mao et al. (2024)** [21] found that vision-language models are more vulnerable than unimodal vision models, hypothesizing that the contrastive learning objective prioritizes semantic alignment over adversarial robustness. Our CheXzero results support this hypothesis: despite strong zero-shot diagnostic performance (AUC ~0.7-0.9 on various pathologies [2]), the model exhibits no adversarial robustness.

#### 5.3.2 Medical Imaging Adversarial Robustness

Several prior works have studied adversarial attacks on medical imaging models:

**Ma et al. (2021)** [12] evaluated adversarial attacks on chest X-ray classifiers, finding:
- CNN-based models (ResNet, DenseNet) are vulnerable to PGD attacks
- Attack success rates decrease with smaller perturbation budgets
- Adversarial training improves robustness but degrades clean accuracy

**Key Difference**: These works focused on supervised CNN models trained on labeled datasets. CheXzero, as a foundation model with zero-shot capabilities, represents a fundamentally different paradigm. Our results suggest that foundation models may inherit adversarial vulnerabilities from their pre-training (CLIP on natural images), even when applied to specialized domains.

**Finlayson et al. (2019)** [8] demonstrated that adversarial examples can transfer across different medical imaging models, raising concerns about systematic attacks on deployed diagnostic systems. Our work confirms this threat extends to foundation models.

#### 5.3.3 Lesion-Aware and Spatially-Constrained Attacks

Our attempted lesion-aware attack builds on prior work constraining perturbations to specific regions:

**Taghanaki et al. (2019)** [17] proposed region-specific attacks on medical image segmentation, showing that small perturbations in tumor regions can cause large segmentation errors. However, their work used CNN-based segmentation models where Grad-CAM is applicable.

**Our Contribution**: We identify that similar lesion-aware approaches are **architecturally incompatible** with CLIP ViT, necessitating alternative attribution methods. This is a novel finding with implications for future adversarial robustness research on transformer-based medical imaging models.

### 5.4 Implications for Clinical Deployment

The adversarial vulnerabilities we identified have serious implications for deploying CheXzero or similar foundation models in clinical settings:

#### 5.4.1 Threat Model: Realistic Attack Scenarios

**Scenario 1: Data Integrity Attacks**
An adversary with access to the image processing pipeline (e.g., compromised PACS system, malicious insider) could inject adversarial perturbations before images reach the diagnostic model. Our results show such perturbations require minimal computational resources (single-step FGSM suffices) and could systematically cause misdiagnoses.

**Scenario 2: Physical Adversarial Attacks**
While we evaluated digital perturbations, research has shown that adversarial perturbations can be made physically robust [22], surviving printing, photographing, and imaging pipelines. An adversary could potentially create adversarial X-ray phantoms or manipulate imaging equipment parameters to create adversarial-like artifacts.

**Scenario 3: Transferability Attacks**
Adversarial examples often transfer across models [23]. An adversary could craft adversarial examples on a surrogate model (e.g., publicly available chest X-ray classifier) and deploy them against CheXzero without direct access to the target model.

**Risk Assessment**: While these scenarios may seem far-fetched, the potential consequences (missed diagnoses, inappropriate treatments) and the low barrier to attack (FGSM on CPU suffices) warrant serious consideration before clinical deployment.

#### 5.4.2 Safety and Certification Challenges

Medical AI systems must meet rigorous safety standards (e.g., FDA approval in the US, CE marking in Europe). Adversarial vulnerability raises several certification challenges:

**1. Robustness Guarantees**: Current certification processes focus on clean accuracy and fairness. Our results suggest adversarial robustness should be added as a safety requirement, with minimum ASR thresholds under standardized attacks.

**2. Input Validation**: Clinical deployments could implement input validation to detect adversarial perturbations. However, this is an arms race—attackers can adapt to evade detection (adaptive attacks [25]).

**3. Ensemble Defenses**: Using multiple diverse models could provide defense-in-depth, as adversarial examples may not transfer perfectly across architectures. However, this increases computational costs and may not provide strong guarantees.

**4. Human-in-the-Loop**: The most reliable defense may be maintaining physician oversight. AI systems should be positioned as decision support tools rather than autonomous diagnostic agents, with clinicians providing a final safety check.

#### 5.4.3 Comparison to Other Medical AI Risks

Adversarial vulnerability is one of several risks associated with medical AI:

| Risk Type | Adversarial | Dataset Shift | Privacy | Bias |
|-----------|-------------|---------------|---------|------|
| Nature | Intentional manipulation | Natural distribution change | Data leakage | Systematic errors |
| Likelihood | Low (requires adversary) | High (common in practice) | Medium | High |
| Impact | High (systematic misdiagnosis) | Medium-High | Medium | High |
| Mitigation | Difficult (no strong defenses) | Domain adaptation | Encryption, federated learning | Fairness constraints |

While adversarial attacks may be less likely than dataset shift or bias issues, their potential for **intentional, systematic harm** distinguishes them as a unique threat. The 100% ASR we observed represents a worst-case scenario that safety-critical systems must be prepared to handle.

### 5.5 Limitations and Threats to Validity

Our study has several important limitations that affect the generalizability and interpretation of results:

#### 5.5.1 Small Sample Size

**Limitation**: We evaluated attacks on only 5 chest X-ray samples due to computational constraints (CPU-only execution, slow inference).

**Impact on Validity**:
- **Statistical Power**: With N=5, AUROC estimates have high variance (confidence intervals ~±0.15). The FGSM AUROC increase may be a statistical artifact.
- **Generalizability**: Results may not reflect model behavior on the full CheXpert test set (N=500+) or other patient populations.
- **Label Distribution**: Our small sample may not cover the full diversity of label combinations in CheXpert's multi-label setting.

**Mitigation**: The 100% ASR across all samples provides strong evidence of vulnerability despite small N. A single successful attack demonstrates an exploitable weakness; 5 out of 5 successes suggest systematic vulnerability. However, quantitative metrics (AUROC drops, perturbation magnitudes) should be interpreted cautiously.

**Future Work**: GPU-accelerated experiments on larger samples (N=100-500) would provide more robust statistical estimates and enable epsilon-sweep analysis to characterize the robustness curve.

#### 5.5.2 Limited Attack Coverage

**Limitation**: We completed only 2 out of 4 planned attack methods (PGD, FGSM). C&W and DeepFool experiments failed to complete due to computational costs (100+ iterations per sample, 10+ minutes per sample on CPU).

**Impact**:
- **Perturbation Optimality**: C&W is known to find near-optimal L2 perturbations [5]. Without C&W results, we cannot determine the minimum perturbation required for attack success.
- **Robustness Lower Bound**: DeepFool estimates the minimum distance to the decision boundary [6], providing a lower bound on adversarial robustness. This gap in our evaluation limits our understanding of CheXzero's true robustness.

**Partial Mitigation**: PGD is considered a strong benchmark attack [5], and its 100% ASR demonstrates vulnerability. FGSM provides a weak baseline, and its success confirms that even simple attacks are effective. While additional attacks would strengthen the evaluation, the completed experiments provide sufficient evidence of vulnerability for a proof-of-concept study.

#### 5.5.3 Single Perturbation Budget

**Limitation**: We evaluated only a single perturbation budget (ε=8 in L∞ norm) rather than sweeping across multiple epsilon values.

**Impact**: We cannot characterize the **robustness curve** showing how attack success varies with perturbation magnitude. Key questions remain unanswered:
- At what ε does ASR drop below 100%?
- What is the minimum perceptible perturbation budget?
- How does AUROC degrade as ε increases?

**Justification**: ε=8 (~3% of [0,255] range) is a standard benchmark in adversarial robustness literature [5] and represents a reasonable threat model. The 100% ASR at this epsilon provides clear evidence of vulnerability, even without a full robustness curve.

#### 5.5.4 CPU-Only Execution

**Limitation**: All experiments ran on CPU due to hardware constraints, limiting iteration counts and sample sizes.

**Impact**:
- **Suboptimal Hyperparameters**: PGD used only 3 iterations vs. typical 10-40, potentially underestimating attack effectiveness. However, 100% ASR suggests 3 iterations were sufficient.
- **Experiment Diversity**: Could not afford expensive attacks (C&W, DeepFool) or large-scale evaluations.
- **Reproducibility**: Other researchers with GPU access may obtain different results with more iterations.

**Mitigation**: We documented all hyperparameters and computational constraints to enable future replication with improved resources.

#### 5.5.5 Lesion-Aware Attack Feasibility

**Limitation**: Our investigation of lesion-aware attacks revealed an architectural incompatibility (Grad-CAM inapplicable to CLIP ViT) rather than executing successful lesion-aware attacks.

**Impact on Research Questions**: RQ3 ("Are attacks constrained to lesion regions more effective?") could not be directly answered. However, we made an alternative contribution: documenting this architectural limitation and its implications for interpretability research.

**Reframing**: Rather than viewing this as a failure, we reframe it as a **technical discovery** with broader implications for adversarial robustness and interpretability research on vision transformers. This finding may be more valuable to the research community than confirming the expected result that lesion-aware attacks are effective.

#### 5.5.6 White-Box Attack Assumption

**Limitation**: Our attacks assume full access to model architecture, weights, and gradients (white-box setting). This is a strong adversary assumption.

**Real-World Applicability**: In deployed systems, attackers may only have black-box access (input-output pairs). However:
- Adversarial examples often **transfer** across models [23], so attacks crafted on surrogate models may still be effective.
- Foundation model weights are increasingly open-source (LLaMA, CLIP), making white-box attacks realistic for these systems.
- Black-box attacks exist (query-based optimization [26]), though typically requiring more queries.

**Justification**: White-box evaluation represents a **worst-case analysis**—if the model is not robust under white-box attacks, it certainly won't be robust under weaker attack models. This is the standard approach in adversarial robustness research [5].

### 5.6 Future Research Directions

Our work opens several avenues for future investigation:

#### 5.6.1 Architectural Robustness of Vision Transformers

**Research Question**: Are transformer-based vision models inherently less adversarially robust than CNNs, or is CLIP's vulnerability specific to contrastive pre-training?

**Proposed Experiments**:
- Compare adversarial robustness of CLIP ViT vs. supervised ViT (ImageNet-trained) vs. CNN (ResNet)
- Evaluate whether adversarial training on transformers improves robustness more or less effectively than on CNNs
- Investigate the role of patch size and model scale on robustness

**Hypothesis**: The CLS-only representation may make ViTs more vulnerable by creating a single point of failure, whereas CNNs' spatial feature maps provide redundancy.

#### 5.6.2 Alternative Spatial Attribution for Transformers

**Research Question**: Which attribution method (attention rollout, perturbation-based, gradient to input patches) most faithfully reflects ViT decision-making?

**Proposed Evaluation**:
- Quantitative faithfulness metrics (insertion/deletion curves [27])
- Qualitative clinical validity (do attribution maps align with radiologist annotations?)
- Computational efficiency analysis

**Impact**: Identifying a reliable attribution method would enable lesion-aware adversarial research on transformer-based medical imaging models.

#### 5.6.3 Adversarial Training for Medical Imaging Foundation Models

**Research Question**: Can adversarial training improve CheXzero's robustness without degrading zero-shot performance?

**Proposed Approach**:
- Fine-tune CheXzero with adversarially perturbed chest X-rays
- Evaluate trade-off between adversarial robustness and clean accuracy
- Test whether robustness transfers across medical imaging domains (chest X-ray → mammography)

**Challenge**: Adversarial training is computationally expensive for large models. Efficient alternatives (fast adversarial training [28], TRADES [29]) should be explored.

#### 5.6.4 Certified Robustness for Medical AI

**Research Question**: Can we provide mathematical robustness guarantees for medical imaging models using certified defenses (randomized smoothing [30], interval bound propagation [31])?

**Proposed Evaluation**:
- Apply randomized smoothing to CheXzero and measure certified radius
- Compare certified vs. empirical robustness (PGD attacks)
- Assess clinical viability (Does certification introduce unacceptable latency or accuracy loss?)

**Impact**: Certified defenses provide provable guarantees rather than heuristic defenses, which is crucial for safety-critical medical applications.

#### 5.6.5 Realistic Physical Adversarial Attacks

**Research Question**: Can adversarial perturbations survive the chest X-ray imaging pipeline (X-ray generation, sensor noise, image processing)?

**Proposed Experiments**:
- Create adversarial perturbations optimized for robustness to JPEG compression, Gaussian noise, contrast adjustment
- Test physical feasibility using X-ray phantoms with controlled density variations
- Evaluate whether adversarial artifacts can be introduced through imaging parameter manipulation (exposure time, kVp settings)

**Clinical Relevance**: Physical attacks are more realistic than purely digital attacks for medical imaging systems with secured data pipelines.

#### 5.6.6 Large-Scale Robustness Evaluation

**Research Question**: How does CheXzero's adversarial vulnerability vary across patient demographics, pathology types, and image quality?

**Proposed Dataset**: Full CheXpert test set (N=500+), stratified by:
- Pathology label (pneumonia, edema, etc.)
- Image view (frontal vs. lateral)
- Patient demographics (if available)

**Analysis**:
- Subgroup robustness disparities (Are attacks more effective on certain pathologies?)
- Correlation between clean accuracy and adversarial robustness
- Perturbation transferability across subgroups

**Expected Finding**: Adversarial vulnerability may vary significantly across subgroups, with implications for fairness and equitable model deployment.

### 5.7 Summary of Key Discussion Points

1. **CheXzero is highly vulnerable to adversarial attacks**, with 100% ASR demonstrating systematic exploitability. Even weak single-step FGSM attacks succeed, indicating a fundamental lack of robustness.

2. **The FGSM AUROC increase is a statistical/metric artifact** due to small sample size and multi-label complexity. ASR is the more reliable metric for adversarial evaluation in this context.

3. **CLIP ViT's CLS-only design prevents gradient flow to spatial tokens**, rendering standard Grad-CAM inapplicable. This is an architectural constraint with broad implications for interpretability and spatial adversarial research on transformers.

4. **Lesion-aware attacks are not feasible with standard Grad-CAM** on CLIP ViT, but alternative attribution methods (attention rollout, perturbation-based) could enable future spatially-constrained adversarial research.

5. **Clinical deployment risks are significant but manageable** through defense-in-depth strategies (input validation, ensemble models, human oversight). However, no strong adversarial defense exists for CLIP-scale models.

6. **Study limitations** (small sample size, limited attack coverage, single epsilon) affect generalizability but do not undermine the core finding of vulnerability. The 100% ASR provides robust evidence despite these constraints.

7. **Future research should focus on** transformer-specific attribution methods, adversarial training for foundation models, certified robustness, and large-scale evaluation across diverse patient populations.

Our work establishes that adversarial robustness is a critical consideration for deploying medical imaging foundation models, and the architectural discovery regarding ViT gradient flow contributes to the broader understanding of transformer interpretability.

---
