# 🎯 Black-Box Decision-Based Perturbation Optimization on Pretrained CNN (Problem 2)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![Task](https://img.shields.io/badge/Task-Adversarial%20Machine%20Learning-success)

This repository contains the complete solution for **Problem 2: Black Box Adversarial Attack on CNN**. The goal of this project is to perform a highly query-efficient, hard-label black-box attack on a Convolutional Neural Network (CNN) to generate visually imperceptible adversarial images.

Included in this repository are the finalized notebook (`p2_code.ipynb`), all 1,000 generated and verified 8-bit PNG images (`adv_images/`), this documentation, and the technical methodology report (`report.pdf`).

---

## 📊 Performance Highlights Across All 1,000 Images

Our 4-Stage Hybrid TSA-SurFree method achieved the following metrics on the final, quantized dataset:

| Evaluation Criterion | Weight / Role | Achieved Result | Benchmark Target |
| :--- | :--- | :--- | :--- |
| **Success Rate (Label Flip)** | **50% Primary** | **87.80% (878 / 1,000)** | Verified after 8-bit `.png` reload |
| **Mean PSNR (Imperceptibility)**| **50% Primary** | **50.29 dB** | $> 40.0 \text{ dB}$ (visually indistinguishable) |
| **Mean $L_2$ Perturbation Norm**| **50% Primary** | **0.2446** (Median: **0.2220**) | Measured in $[0, 1]$ RGB space |
| **Batched Oracle Inferences** | **Tie-Breaker 1** | **121 batched `model()` calls**| Active-set dynamic filtering |

---

## 📂 Repository Structure

```text
├── adv_images/               # Directory containing 1,000 generated adversarial .png files
├── p2_code.ipynb             # Jupyter Notebook with the 4-stage optimization pipeline
├── report.pdf                # Technical report on mathematical formulation and convergence
└── README.md                 # Project documentation and execution instructions
```

---

## 🧠 Method Overview: 4-Stage Hybrid TSA-SurFree Optimization

The target model operates strictly as a **hard-label black-box oracle**, returning only top-1 predicted class indices $\hat{y} \in \{0, \dots, 9\}$ without exposing network architecture, weights, or intermediate logits. Standard decision-based methods (e.g., Boundary Attack) require thousands of queries because random directions in high-dimensional space have extremely low expected cosine similarity with the true boundary normal.

Our framework resolves this through four synergistic stages:

### 1. Zero-Query MixUp Surrogate Ensemble & Neighbor Graph (0 Oracle Calls)
* Using only the clean dataset `(X, y_true)`, we train a lightweight two-network residual ensemble with inter-class MixUp regularization for 22 epochs.
* MixUp supervises linear interpolation paths between distinct classes, allowing the ensemble to approximate decision boundaries and yield runner-up margin directions with **zero calls to the target model**.
* An exact pairwise $L_2$ distance matrix identifies the $K=8$ nearest cross-class clean neighbors, providing a 100% guaranteed label-flipping fallback ray.

### 2. Batched Multi-Scale Ray Probing & Binary Search Localization
* Candidate rays from the surrogate prior and top cross-class neighbors are snapped to the 8-bit RGB grid and evaluated in a single batched oracle call.
* The lowest-$L_2$ successful ray brackets the decision boundary, and a 10-step vectorized binary search projects all 1,000 images onto the tight boundary bracket $[x_{\text{clean}}, x_{\text{bdry}}]$ with radial precision $< 10^{-3}$.

### 3. 1-Sided Tangent Sign-Aggregated SurFree (`TSA-SurFree`)
* At the boundary midpoint $x_{\text{mid}}$, we sample $M=16$ orthogonal tangent vectors using batched QR decomposition, masked to exclude saturated pixels to prevent radial clipping.
* Because $x_{\text{mid}}$ lies directly on the decision surface, a **1-sided probe** yields 1 bit of normal sign information per query, halving gradient estimation queries.
* Aggregating the signed basis produces a high-SNR tangent normal $\hat{u}_{\text{agg}}$, guiding a parallel geodesic SurFree arc search.
* Images achieving $L_2 \le 0.28$ ($\text{PSNR} \ge 46.5 \text{ dB}$) are retired early to minimize model inferences.

### 4. Confidence-Boosting Normal Push & Exact 8-Bit Quantization Lock
* To maximize prediction confidence change and prevent 8-bit quantization reversion, each boundary vector is pushed along the normal by $\eta = 0.18$ and rigorously snapped via `torch.round(x_rgb * 255.0).clamp(0, 255) / 255.0` before final verification.

---

## 🚀 Quick-Start Instructions

### Prerequisites
* Python 3.8+
* `torch`, `torchvision`
* `Pillow`, `matplotlib`, `numpy`

### Execution
1. Clone this repository and ensure your environment has the required dependencies.
2. Place the provided clean dataset (`clean_images/`) and pretrained model (`cnn_model.pt`) in your working directory.
3. Open `p2_code.ipynb` in Google Colab or your local Jupyter environment.
4. Execute the cells sequentially. 
   * The optimization block executes the black-box attack using strictly hard-label oracle calls.
   * The final block handles denormalization, exact 8-bit PNG quantization, and exports the robust adversarial images to the `./adv_images/` directory.
