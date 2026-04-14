## Key Formulas for Paper

### 3. Methodology (Formulas)

#### 3.1 Structured Causal Primitives (H1)

**Primitive Structure:**
Each evidence primitive $p_i$ is defined as:
$$p_i = (I_i, C_i, O_i, \delta_i, D_i, P_i)$$

Where:
- $I_i$: Intervention (treatment)
- $C_i$: Comparator (control)
- $O_i$: Outcome
- $\delta_i$: Effect size
- $D_i$: Study design (RCT, observational, meta-analysis)
- $P_i$: Population

**Primitive Encoding:**
The primitive embedding $e(p_i)$ is computed as:
$$e(p_i) = \text{Normalize}([\mathbb{1}(D_i=\text{RCT}), |\delta_i|/10, \phi(P_i), \phi(I_i), \phi(C_i), \phi(O_i)])$$

Where $\phi(\cdot)$ is a hash-based encoding function.

**Primitive Retrieval:**
Given patient features $x$, retrieve top-$k$ primitives:
$$\mathcal{P}_k = \arg\top_{p_i \in \mathcal{P}} \text{sim}(f(x), e(p_i))$$

Where $f(x)$ is the patient encoder and $\text{sim}(\cdot, \cdot)$ is cosine similarity.

**Attribution Score:**
The attribution score $\alpha_i$ for primitive $p_i$ is:
$$\alpha_i = \text{softmax}(g(h, y))_i$$

Where $h$ is the primitive representation and $y$ is the predicted outcome.

#### 3.2 PICO-Contrastive RAG

**PICO Structure:**
For query $q$, extract PICO elements:
$$\text{PICO}(q) = (\text{Pop}, \text{Int}, \text{Comp}, \text{Out})$$

**Contrastive Embedding:**
The treatment effect embedding is:
$$e_{\text{contrast}} = f(\text{Pop}, \text{Int}) - f(\text{Pop}, \text{Comp})$$

Where $f(\cdot, \cdot)$ is the encoder for population-treatment combination.

**Nuance-Weighted Retrieval:**
Retrieve documents with statistical nuance weighting:
$$\text{score}(d_j) = \text{sim}(e_{\text{contrast}}, e(d_j)) \cdot w(d_j)$$

Where $w(d_j)$ is the nuance score based on statistical quality:
$$w(d_j) = \sigma(\text{MLP}(e(d_j)))$$

**Outcome Prediction:**
The predicted outcome $\hat{y}$ is:
$$\hat{y} = \text{MLP}([e_{\text{treat}}; e_{\text{conf}}; e_{\text{ret}}])$$

Where:
- $e_{\text{treat}}$: Treatment encoding
- $e_{\text{conf}}$: Confounder encoding
- $e_{\text{ret}}$: Retrieval encoding

#### 3.3 Loss Functions

**H1 Loss:**
$$\mathcal{L}_{\text{H1}} = \mathcal{L}_{\text{pred}} + \lambda_1 \mathcal{L}_{\text{attr}}$$

Where:
- $\mathcal{L}_{\text{pred}} = \text{MSE}(y, \hat{y})$ (prediction loss)
- $\mathcal{L}_{\text{attr}} = \text{MSE}(\alpha, \hat{\alpha})$ (attribution consistency loss)
- $\lambda_1 = 0.1$ (weight)

**PICO-Contrastive RAG Loss:**
$$\mathcal{L}_{\text{PICO}} = \mathcal{L}_{\text{pred}} + \lambda_2 \mathcal{L}_{\text{contrast}}$$

Where:
- $\mathcal{L}_{\text{contrast}} = -\log(\text{sim}(e_{\text{contrast}}, e(d^+)) + \epsilon)$ (contrastive loss)
- $\lambda_2 = 0.3$ (weight)

**Baseline Loss:**
$$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{pred}} + \lambda_3 \mathcal{L}_{\text{IPW}}$$

Where:
- $\mathcal{L}_{\text{IPW}} = \frac{1}{\pi(t|x)} \cdot \mathcal{L}_{\text{pred}}$ (inverse probability weighting)
- $\lambda_3 = 0.1$ (weight)

### 4. Evaluation Metrics (Formulas)

#### 4.1 Regression Metrics

**Mean Squared Error (MSE):**
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Mean Absolute Error (MAE):**
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Pearson Correlation:**
$$r = \frac{\sum_{i=1}^{n} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{n} (\hat{y}_i - \bar{\hat{y}})^2}}$$

#### 4.2 Classification Metrics

**Precision:**
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall:**
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score:**
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**AUC-ROC:**
$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

Where:
- $\text{TPR}(t) = \text{Recall}$ at threshold $t$
- $\text{FPR}(t) = \frac{FP}{FP + TN}$ at threshold $t$

#### 4.3 Clinical Metrics

**Mortality Rate:**
$$\text{Mort} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i^{\text{mort}} = 1)$$

**30-Day Readmission Rate:**
$$\text{Readm} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i^{\text{readm}} = 1)$$

**Combined Outcome:**
$$y_i = 0.4 \cdot y_i^{\text{mort}} + 0.3 \cdot y_i^{\text{readm}} + 0.2 \cdot \min\left(\frac{\text{LOS}_i}{30}, 1\right) + 0.1 \cdot \min\left(\frac{\text{ICU}_i}{14}, 1\right)$$

### 5. Improvement Calculations

**MSE Improvement:**
$$\Delta_{\text{MSE}} = \frac{\text{MSE}_{\text{base}} - \text{MSE}_{\text{model}}}{\text{MSE}_{\text{base}}} \times 100\%$$

**F1 Improvement vs PACE-RAG:**
$$\Delta_{\text{F1}} = \frac{\text{F1}_{\text{model}} - \text{F1}_{\text{PACE}}}{\text{F1}_{\text{PACE}}} \times 100\%$$

**Correlation Improvement:**
$$\Delta_{\text{corr}} = \frac{r_{\text{model}} - r_{\text{base}}}{r_{\text{base}}} \times 100\%$$

### 6. Retrieval Metrics

**Precision@K:**
$$\text{P@K} = \frac{|\{\text{relevant docs in top-}K\}|}{K}$$

**Recall@K:**
$$\text{R@K} = \frac{|\{\text{relevant docs in top-}K\}|}{|\{\text{all relevant docs}\}|}$$

**Nuanced Precision@5:**
$$\text{NP@5} = \frac{|\{\text{high-certainty evidence in top-5}\}|}{5}$$

Where high-certainty evidence includes RCTs, systematic reviews, and meta-analyses.

### 7. Statistical Significance

**Paired t-test:**
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

Where:
- $\bar{d}$: Mean difference between model and baseline
- $s_d$: Standard deviation of differences
- $n$: Number of samples

**Effect Size (Cohen's d):**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$$

Where $s_p$ is the pooled standard deviation:
$$s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

**Confidence Interval (95%):**
$$\text{CI} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$$
