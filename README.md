# XGBoost Vignette

### Overview

Vignette on implementing XGBoost using the Iris Species dataset from Kaggle; created as a class project for PSTAT197A in Fall 2025.

### Contributors

Brooks Piper, Jay Leung, Coraline Zhu, Shahil Patel, Taneesha Panda

### Abstract

This vignette demonstrates how to use the `xgboost` library in Python to build a multi-class classifier on the classic iris dataset. The goal is to predict the species of a flower from four numeric features (sepal length, sepal width, petal length, and petal width). We show how to load the data, split it into training and test sets with `scikit-learn`, fit an `XGBClassifier`, and evaluate performance using classification accuracy. The accompanying notebook includes a summary of what the model does and presents the main results in a way that another PSTAT197 student can easily follow and adapt to their own data.

## Repository Contents

- `data/`  
  - `Iris.csv`: contains the example dataset used in the vignette (iris data in CSV form).
  - `Iris_processed.csv`: contains the example dataset and predicted class labels.
 
- `img/`
  - `XGBoost-breakdown.jpg`: JPG image of the XGBoost breakdown found in `vignette.ipynb` and its associated renders.

- `scripts/`  
  - `vignette-script.py`: Python script that follows the complete vignette implementation: loads the dataset, trains the XGBoost classifier, makes predictions, and reports accuracy. This script mirrors the main steps in the notebook, with line-by-line comments for reproducibility.
  - `exploratory-analysis.py`: Python script that conducts exploratory analysis on the dataset.
  - `model-fitting.py`: Python script that fits the XGBoost multi-class classifier.
  - `visualization.py`: Python script that creates model testing visualizations.

- `vignette.ipynb`  
  - main vignette notebook that combines narrative text, code cells, and output. It explains what the model does, walks through the analysis, and shows the final results.

- `vignette.html`
  - HTML render of `vignette.ipynb`.

- `vignette.pdf`
  - PDF render of `vignette.ipynb`.

## References

1. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.**  
   *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

2. XGBoost Developers. **XGBoost Documentation.**  
   Available at: https://xgboost.readthedocs.io

3. Fisher, R. A. (1936). **The Use of Multiple Measurements in Taxonomic Problems.**  
   *Annals of Eugenics*, 7(2), 179–188. (Original source of the iris dataset.)
