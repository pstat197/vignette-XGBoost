# XGBoost Vignette

### Overview

Vignette on implementing XGBoost using the Iris Species dataset from Kaggle; created as a class project for PSTAT197A in Fall 2025.

### Contributors

Brooks Piper, Jay Leung, Coraline Zhu, Shahil Patel, Taneesha Panda

### Abstract

This vignette demonstrates how to use the `xgboost` library in Python to build a multi-class
classifier on the classic iris dataset. The goal is to predict the species of a flower from
four numeric features (sepal length, sepal width, petal length, and petal width). We show
how to load the data, split it into training and test sets with `scikit-learn`, fit an
`XGBClassifier`, and evaluate performance using classification accuracy. The accompanying
notebook includes a short summary of what the model does and presents the main results in a
way that another PSTAT197 student can easily follow and adapt to their own data.

## Repository Contents

- `data/`  
  - contains the example dataset used in the vignette (iris data in CSV form).

- `scripts/`  
  - `drafts/vignette-script.py`: Python script that loads the dataset, trains the XGBoost
    classifier, makes predictions, and reports accuracy. This script mirrors the main steps
    in the notebook, with line-by-line comments for reproducibility.

- Notebook file (Jupyter)  
  - main vignette notebook that combines narrative text, code cells, and output.  
    It explains what the model does, walks through the analysis, and shows the final results.
    (Update this line with the exact filename, e.g. `xgboost_iris_vignette.ipynb`.)

- `img/` (optional)  
  - figures generated from the analysis, such as feature-importance plots or confusion
    matrices, if used in the vignette.

- `README.md`  
  - overview of the project, repository structure, and references.

- `.gitignore`, `LICENSE`  
  - standard project files created when the repository was initialized.

## References

1. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.**  
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

2. XGBoost Developers. **XGBoost Documentation.**  
   Available at: https://xgboost.readthedocs.io

3. Fisher, R. A. (1936). **The Use of Multiple Measurements in Taxonomic Problems.**  
   *Annals of Eugenics*, 7(2), 179–188. (Original source of the iris dataset.)
