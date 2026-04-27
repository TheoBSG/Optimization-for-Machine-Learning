# Optimization for Machine Learning

## Overview

This repository contains the code and analysis for three advanced projects in optimization for machine learning. The work focuses on three key areas: geometric optimization on manifolds, synchronization problems on Lie groups, and robust optimization for generalization. Each project demonstrates a deep understanding of mathematical principles, algorithmic implementation, and large-scale empirical validation.

The skills demonstrated here—including Riemannian geometry, spectral methods, second-order optimization, and sharpness-aware minimization—are directly transferable to high-dimensional signal processing, sensor networks, and robust estimation in complex systems.

---

## Project 1: Geometric Optimization on the Product of Spheres

**Project Name:** `orthogonal-manifold-optimization`

### Description

This project explores optimization on the manifold of orthogonal frames, specifically the product of spheres $OB(d, n) = (\mathbb{S}^{d-1})^n$, with the diagonal set excluded to avoid singularities. The core objective is to maximize the pairwise angular separation between $n$ points on a $d$-dimensional sphere, a problem equivalent to minimizing the sum of negative log-distances.

The work provides a complete pipeline from theoretical manifold analysis to numerical optimization, including deriving the Riemannian structure, implementing a retraction, and developing a gradient descent algorithm with backtracking line-search.

### Key Contributions

- **Manifold Geometry**: Formalized $OB(d, n)$ as an embedded submanifold of $\mathbb{R}^{d \times n}$, derived its tangent space structure, and defined the orthogonal projection onto it.
- **Retraction Map**: Implemented a computationally efficient retraction based on component-wise normalization, satisfying second-order properties.
- **Riemannian Gradient Descent**: Derived the closed-form Riemannian gradient for the angular separation objective and implemented a backtracking line-search for adaptive step-size selection.
- **Numerical Validation**: Verified gradient correctness via Taylor expansion (error slope ≈ 2) and confirmed convergence to known optimal configurations (regular polygons on $\mathbb{S}^1$, Platonic solids on $\mathbb{S}^2$).

### Results

The algorithm successfully recovers global optima for low-dimensional cases (e.g., $d=2$, $n=20$) and achieves near-optimal configurations for high-dimensional problems (e.g., $d=3$, $n=128$). The method demonstrates linear convergence of the gradient norm on a log scale, with robustness to random initialization.

### Technologies

- Python / MATLAB
- NumPy / SciPy
- Custom manifold implementations

---

## Project 2: Rotation Synchronization on the Special Orthogonal Group

**Project Name:** `spectral-rotation-synchronization`

### Description

This project addresses the problem of rotation synchronization: recovering $m$ unknown rotations $R_i \in SO(d)$ from noisy relative measurements $H_{ij} \approx R_i R_j^\top$ in a graph $G = (V, E)$. The problem is cast as maximum likelihood estimation under a Langevin mixture model, where measurements are either inliers (concentrated around the true relative rotation) or outliers (uniformly distributed).

We propose a two-stage approach: a spectral initialization via generalized eigenvalue decomposition, followed by refinement using Riemannian optimization (gradient descent and trust-region methods). The work includes derivation of the Riemannian gradient and Hessian for the log-likelihood on the product manifold $SO(d)^m$.

### Key Contributions

- **Problem Formulation**: Expressed the MLE as a minimization problem on $SO(d)^m$ with anchored rotations, handling outliers via a mixture model.
- **Spectral Initialization**: Derived a convex relaxation via the generalized eigenvalue problem $(W_1, D_1)$, where $W_1$ is the block measurement matrix and $D_1$ the degree matrix. Proved that the solution yields a near-optimal starting point.
- **Riemannian Optimization**: Implemented both Riemannian gradient descent (RGD) and Riemannian trust-region (RTR) methods, with analytic gradient and Hessian-vector products.
- **Robustness Analysis**: Systematically compared random vs. spectral initialization under varying outlier rates ($q$), graph densities ($ER_p$), and noise concentrations ($\kappa_1$). Demonstrated that spectral initialization is critical when $q$ is low (many outliers) or graph density deviates from $0.5$.

### Results

Spectral initialization significantly improves convergence and final MSE, particularly in high-outlier regimes ($q=0.1$, MSE improvement of >50%). The method achieves state-of-the-art synchronization performance on synthetic graphs with up to $m=100$ nodes and $d=3$.

### Technologies

- Python / PyManopt / Manopt
- SciPy (sparse eigen solvers)
- Custom manifold classes for $SO(d)^m$ with anchors

---

## Project 3: Sharpness-Aware Minimization for Transformers and Generative Models

**Project Name:** `flat-minima-generalization`

### Description

Sharpness-Aware Minimization (SAM) is an optimization technique that explicitly seeks flat minima of the loss landscape, which are known to correlate with better generalization and robustness. While SAM has been extensively studied for CNNs, its effectiveness on modern architectures like Vision Transformers (ViT) and generative models (diffusion models, flow matching) remains largely unexplored.

This work provides the first systematic evaluation of SAM on these architectures. We compare SAM (wrapped around Adam) against standard Adam and SGD across classification tasks (ResNet18, ViT) and generative tasks (score-based diffusion, DDPM, flow matching). We assess robustness via label noise, adversarial attacks (FFGSM), and parameter perturbations.

### Key Contributions

- **Classification Analysis**: Trained ResNet18 and ViT on CIFAR-10 and MNIST under label noise (0%, 10%, 20%). SAM-Adam consistently outperforms Adam for ViT, especially under noise (up to +3% accuracy). Hessian spectral analysis confirms that SAM finds flatter minima (shorter right-tail eigenvalue distribution).
- **Generative Model Evaluation**: For the first time, applied SAM to score-based diffusion, DDPM, and flow matching. SAM-trained models exhibit significantly better robustness to weight perturbations: at $\epsilon=0.01$ noise, DDPM FID improves from 156.7 (Adam) to 44.7 (SAM).
- **Adversarial Robustness**: SAM-Adam improves adversarial accuracy on CIFAR-10 (ResNet18: +3.1% at $P=0$), though gains are architecture-dependent.
- **Hyperparameter Analysis**: Evaluated SAM's neighborhood radius $\rho \in \{0.05, 0.15, 0.3\}$; found $\rho=0.05$ to be optimal across most settings.

### Results

SAM is particularly effective for generative models, where robustness to parameter perturbations is critical. For classification, benefits are architecture-specific: SAM-Adam + ViT is a winning combination, while SAM-SGD underperforms relative to vanilla SGD.

### Technologies

- Python / PyTorch
- Diffusers (Hugging Face)
- Custom SAM wrapper for Adam/SGD
- PyHessian for eigenvalue spectral density estimation

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- NumPy / SciPy
- PyManopt (for Project 2)
- Matplotlib / Seaborn

### Installation

```bash
git clone https://github.com/your-username/optimization-for-machine-learning.git
cd optimization-for-machine-learning
pip install -r requirements.txt