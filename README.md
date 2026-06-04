# Breast Cancer Detection
**Wisconsin Diagnostic Breast Cancer Dataset · Support Vector Machines · scikit-learn**
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## What this is

This project trains a binary classifier to distinguish malignant from benign breast tumors using physical measurements extracted from cell nucleus images. The dataset is the well-known Wisconsin Diagnostic Breast Cancer dataset, with 569 patients and 30 numerical features.

The main goal isn't just to get a high F1 score — it's to understand *why* SVMs behave the way they do under different conditions. The notebook walks through a deliberate sequence of experiments: stripping the model down to two features, adding scaling, switching kernels, and pushing gamma to its extremes to see what breaks and why.

---

## Results summary

| Configuration | Features | F1 Score |
|---|---|---|
| Linear SVM, no scaling | 2 | 0.872 |
| Linear SVM, with scaling | 2 | 0.886 |
| Linear SVM, with scaling | 29 | **0.988** |
| PolynomialFeatures + LinearSCV | 2 | 0.925 |
| Polynomial kernel (degree 3) | 29 | 0.976 |
| RBF kernel, gamma=0.01 | 29 | 0.976 |
| RBF kernel, gamma=0.1 | 29 | 0.965 |
| RBF kernel, gamma=1 | 29 | 0.091 |
| RBF kernel, gamma=10 | 29 | 0.0 |

The linear kernel on the full scaled dataset came out on top. The RBF results at gamma=1 and gamma=10 are not mistakes — they're the point.

---

## Why F1 Score

In a medical classification task, raw accuracy is misleading. A model that predicts "benign" for every patient would be 63% accurate just by following the class distribution. What matters is the balance between:

- **False negatives:** a malignant tumor classified as benign → delayed treatment, worse prognosis
- **False positives:** a benign tumor classified as malignant → unnecessary procedures, patient anxiety

F1 Score balances Precision and Recall, making it a more honest metric for this kind of problem.

---

## What each experiment shows

**2 features, no scaling**
Starting with only `concavity_mean` and `concave points_mean` allows the decision boundary to be plotted directly. The model works, but the boundary sits in an overlapping region where the classes mix — and the F1 Score reflects that.

**Adding StandardScaler**
SVM optimizes margins using Euclidean distance. If one feature has values in the thousands and another in the hundredths, the large-scale feature dominates the geometry regardless of how informative it actually is. Scaling brings everything to the same unit space. For distance-based algorithms, this isn't optional.

**All 29 features**
Going from 2 to 29 features with proper scaling pushes the F1 from 0.886 to 0.988. The extra features each contribute a small piece of information that the two-feature model can't recover.

**Polynomial kernel**
Two approaches are compared: explicit feature expansion with `PolynomialFeatures + LinearSVC`, and the implicit kernel trick with `SVC(kernel='poly')`. Both produce non-linear boundaries and score around 0.91–0.97. The kernel trick is computationally cleaner; the explicit expansion is more interpretable.

**RBF kernel — gamma sensitivity**
This is where the bias-variance tradeoff becomes concrete. As gamma increases, each support vector's influence shrinks, and the boundary wraps more tightly around the training data:

- gamma=0.01 → smooth boundary, F1=0.976
- gamma=0.1 → still reasonable, F1=0.965
- gamma=1 → the model starts memorizing, F1 collapses to 0.091
- gamma=10 → complete overfitting, F1=0.0 (predicts one class for everything)

Visualizing the decision boundaries at each gamma value makes this progression clear in a way that a table alone doesn't capture.

---

## Project structure

```
Breast_cancer.ipynb     # Full notebook with visualizations and commentary
breast-cancer.csv       # Wisconsin Diagnostic Breast Cancer dataset
```

---

## Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`
