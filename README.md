# Optimal Fairness under Local Differential Privacy
**Authors:** Hrad Ghoukasian, Shahab Asoodeh  
**Status:** Manuscript in preparation

This repository contains the full implementation and experimental pipeline for our forthcoming paper **Optimal Fairness under Local Differential Privacy**. The project introduces **OPT**, the optimal locally differentially private mechanism designed to minimize data unfairness while satisfying non-trivial utility. We compare OPT against existing LDP mechanisms and state-of-the-art fairness interventions across multiple datasets and fairness metrics.



---
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

---
## 🔍 1. OPT vs Existing LDP Mechanisms  
*(Figures 1–5)*

### Steps:

1. Run `OPTvsLDPmechanisms/Experiment_OPT.ipynb`  
   - Import `OPTvsLDPmechanisms/dataset_results_functions.zip`  
   - Select dataset and protected attribute

2. Run `OPTvsLDPmechanisms/Experiment_LDPmechanisms.ipynb` for RR, GRR, SS  

3. Use `OPTvsLDPmechanisms/Plot_LDPmechanismsVsOPT.ipynb` to generate Figures 1–5


### Dataset–Attribute Mapping:

| Fig | Dataset | Protected Attribute |
|-----|---------|---------------------|
| 1   | Adult   | gender              |
| 2   | LSAC    | gender              |
| 3   | Adult   | race                |
| 4   | LSAC  | family income       |
| 5   | Adult   | race-gender         |

---
## 🔍 2. OPT vs RR under Mozannar et al. Framework  
*(Figure 6)*

**Run:**

```bash
FairLearning/Experiment_FairLearning.ipynb
```
Produces accuracy–equalized odds plots.

---
## 🔍 3. OPT vs FairProjection  
*(Table 1)*

**Run in:**

```bash
FairProjection/fair-projection/adult-compas/
```
For Adult:

```bash
python3 run-mp-1.py --dataset adult
```

For COMPAS:
```bash
python3 run-mp-2.py --dataset compas
```

This generates:
adult-mp.pkl
compas-mp.pkl
Then, reproduce Table 1 using:

```bash
Plot_FairProjection.ipynb
```

---
## 🔍 4. OPT vs FairBalance  
*(Table 2)*

**Run:**

```bash
python3 FairBalance/src/main.py
```
Results appear in `FairBalance/results/test/`.


---
## 📚 Citation

```bibtex
@article{ghoukasian2025optimal,
  title   = {Optimal Fairness under Local Differential Privacy},
  author  = {Ghoukasian, Hrad and Asoodeh, Shahab},
  journal = {arXiv preprint arXiv:2511.16377},
  year    = {2025}
}
```
---
## 📬 Contact

For questions, feel free to reach out:
hradghoukasian@gmail.com
