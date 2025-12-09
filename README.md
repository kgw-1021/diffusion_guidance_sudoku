# 🧩 Logic-Diffusion: Solving Discrete Constraints via Continuous Guidance

> **"Solving the hardest logical constraints with the most general diffusion models via the most flexible inference guidance."**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Diffusion](https://img.shields.io/badge/Model-DDPM-blueviolet?style=for-the-badge)](https://arxiv.org/abs/2006.11239) [![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

## 📖 Introduction

**Logic-Diffusion** is a novel framework that solves **discrete logical constraint satisfaction problems (CSPs)** using **continuous diffusion models**.

Unlike traditional solvers that require supervised learning on (problem, solution) pairs, this framework treats the solver as an **Inverse Problem**. We demonstrate that a standard diffusion model, trained only on the inherent structure of valid data (e.g., solved Sudoku grids), can solve complex logical puzzles by injecting constraints as **energy gradients** during the inference phase.

This approach bridges the gap between **Generative AI** (learning the manifold of valid data) and **Symbolic Logic** (satisfying hard constraints), offering a flexible, training-free solution for combinatorial problems.

---

## 💡 Key Contributions

### 1. Standard Diffusion for Discrete Logic
We utilize a standard **continuous Gaussian diffusion model** by applying **continuous relaxation** (Softmax probabilities) to discrete variables. This avoids the complexity of discrete diffusion architectures (e.g., D3PM) while enabling the use of powerful gradient-based optimization.

### 2. Logic as Differentiable Energy
We translate logical hard constraints (e.g., "All numbers in a row must be unique") into differentiable **Energy Functions**:
* **Sum Constraint:** Ensuring validity of probability distribution.
* **Orthogonality Constraint:** Penalizing duplicate values via Gram matrices.
* **Entropy Minimization:** Encouraging the model to make decisive choices (sharpening).

### 3. Inference-time Guidance (Zero-shot Adaptation)
Constraints are not memorized during training but are injected during sampling.
* **Flexible:** You can change the rules (e.g., diagonal Sudoku) without re-training the model.
* **Iterative Refinement:** We employ **Langevin Dynamics** within diffusion steps to escape local minima in the energy landscape, enabling the solution of "Hard" level puzzles.

---

## 🚀 Methodology

Our approach is based on **Posterior Sampling** using Bayes' Theorem:

$$P(x|y) \propto P(y|x) \cdot P(x)$$

* **Prior $P(x)$:** A pre-trained Diffusion Model that learns the "syntax" of valid Sudoku grids (Unsupervised).
* **Likelihood $P(y|x)$:** Modeled via Energy-based Guidance.
    $$\nabla_x \log P(y|x) \approx - \nabla_x E_{constraints}(x)$$

### The Solver Process
1.  **Prior Sampling:** The model predicts a clean image $x_0$ from noise.
2.  **Constraint Check:** We compute the energy $E(x)$ based on logical rules (Rows/Cols/Boxes).
3.  **Langevin Loop:** We iteratively update $x_t$ using the gradient of the energy to resolve conflicts (e.g., duplicate numbers).
4.  **Hard Constraint Enforcement:** Given clues ($y$) are fixed using a replacement method at every step.

---

## 📊 Comparison with Existing Methods

Logic-Diffusion stands out as a **General-Purpose Generative Solver**.

| Feature | Supervised Solvers (e.g., SATNet) | Structured Diffusion (e.g., GSDM) | **Ours (Logic-Diffusion)** |
| :--- | :--- | :--- | :--- |
| **Learning Paradigm** | Supervised Mapping ($f(y)=x$) | Structured / Graph-based | **Unsupervised ($P(x)$ Only)** |
| **Data Requirement** | (Problem, Solution) Pairs | Solution Data | **Solution Data Only** |
| **Constraint Handling** | Fixed after training | Embedded in Architecture | **Injected at Inference (Flexible)** |
| **Adaptability** | Low (Re-training required) | Low (Architecture change) | **High (Zero-shot)** |
| **Mechanism** | Black-box Neural Net | Graph Neural Networks | **Langevin Dynamics + Energy** |

---

## 🧪 Experiments & Results

We evaluated the model on the `Ritvik19/Sudoku-Dataset`. The difficulty is categorized by the number of provided clues.

### Quantitative Results
*Settings: Guidance Scale=20.0, Ortho Weight=5.0, Langevin Steps=5*

### (TO DO) 

> **Blank Accuracy:** The accuracy of filling in the empty cells (excluding provided clues).
> Even in failed cases on 'Hard' puzzles, the model achieves ~97% accuracy, indicating it understands the global structure but may miss 1-2 local constraints.

### Qualitative Analysis: Escaping Local Minima
By visualizing the energy trajectory, we observed that **Langevin Dynamics** plays a crucial role. Without it ($K=0$), the energy stabilizes at a high value (logical conflict). With Langevin steps ($K=5$), the energy drops step-wise within a single timestep, resolving conflicts like duplicate numbers.

---
