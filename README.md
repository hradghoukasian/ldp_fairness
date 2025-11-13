# ldp_fairness
Code and experiments for "Optimal Fairness under Local Differential Privacy"

# Optimal Fairness under Local Differential Privacy
**Authors:** Hrad Ghoukasian, Shahab Asoodeh  
**Status:** Manuscript in preparation

This repository contains the full implementation and experimental pipeline for our forthcoming paper **Optimal Fairness under Local Differential Privacy**. The project introduces **OPT**, the optimal locally differentially private mechanism designed to reduce data unfairness while preserving utility. We compare OPT against existing LDP mechanisms and state-of-the-art fairness interventions across multiple datasets and fairness metrics.




## 🔧 Installation

Create a minimal Python environment:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn
pip install jupyter
```

For FairProjection and FairBalance:
```bash
pip install cvxpy
pip install aif360
```

All experiments run on CPU; no GPU is required.


## 🔍 1. OPT vs Existing LDP Mechanisms  
*(Figures 1–5)*

### Steps:

1. Run `Experiment_OPT.ipynb`  
   - Import `dataset_results_functions.zip`  
   - Select dataset and protected attribute

2. Run `Experiment_LDPmechanisms.ipynb` for RR, GRR, SS  

3. Use `Plot_LDPmechanismsVsOPT.ipynb` to generate Figures 1–5

---

### Dataset–Attribute Mapping:

| Fig | Dataset | Protected Attribute |
|-----|---------|---------------------|
| 1   | Adult   | gender              |
| 2   | LSAC    | gender              |
| 3   | Adult   | race                |
| 4   | Adult   | family income       |
| 5   | Adult   | race-gender         |


