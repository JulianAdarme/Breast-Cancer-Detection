# Breast Cancer Detection – English
**Wisconsin Diagnostic Breast Cancer Dataset · Support Vector Machines · scikit-learn**

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## Table of Contents

- [What this is](#what-this-is)
- [Results Summary](#results-summary)
- [Why F1 Score](#why-f1-score)
- [What Each Experiment Shows](#what-each-experiment-shows)
- [Project Structure](#project-structure)
- [Stack](#stack)

---

## What this is

This project trains a binary classifier to distinguish malignant from benign breast tumors using physical measurements extracted from cell nucleus images. The dataset is the well-known Wisconsin Diagnostic Breast Cancer dataset, with 569 patients and 30 numerical features.

The main goal isn't just to get a high F1 score — it's to understand *why* SVMs behave the way they do under different conditions. The notebook walks through a deliberate sequence of experiments: stripping the model down to two features, adding scaling, switching kernels, and pushing gamma to its extremes to see what breaks and why.

## Results Summary

| Configuration | Features | F1 Score |
|---|---|---|
| Linear SVM, no scaling | 2 | 0.864 |
| Linear SVM, with scaling | 2 | 0.886 |
| Linear SVM, with scaling | 29 | **0.988** |
| PolynomialFeatures + LinearSVC | 2 | 0.925 |
| Polynomial kernel (degree 3) | 29 | 0.976 |
| RBF kernel, gamma=0.01 | 29 | 0.976 |
| RBF kernel, gamma=0.1 | 29 | 0.965 |
| RBF kernel, gamma=1 | 29 | 0.091 |
| RBF kernel, gamma=10 | 29 | 0.0 |

The linear kernel on the full scaled dataset came out on top. The RBF results at gamma=1 and gamma=10 are not mistakes — they're the point.

## Why F1 Score

In a medical classification task, raw accuracy is misleading. A model that predicts "benign" for every patient would be ~63% accurate just by following the class distribution. What matters is the balance between:

- **False negatives:** a malignant tumor classified as benign → delayed treatment, worse prognosis
- **False positives:** a benign tumor classified as malignant → unnecessary procedures, patient anxiety

F1 Score balances Precision and Recall, making it a more honest metric for this kind of problem.

## What Each Experiment Shows

**2 features, no scaling**
Starting with only `concavity_mean` and `concave points_mean` allows the decision boundary to be plotted directly. The model works, but the boundary sits in an overlapping region where the classes mix — and the F1 Score reflects that.

**Adding StandardScaler**
SVM optimizes margins using Euclidean distance. If one feature has values in the thousands and another in the hundredths, the large-scale feature dominates the geometry regardless of how informative it actually is. Scaling brings everything to the same unit space. For distance-based algorithms, this isn't optional.

**All 29 features**
Going from 2 to 29 features with proper scaling pushes the F1 from 0.886 to 0.988. The extra features each contribute a small piece of information that the two-feature model can't recover.

**Polynomial kernel**
Two approaches are compared: explicit feature expansion with `PolynomialFeatures + LinearSVC`, and the implicit kernel trick with `SVC(kernel='poly')`. Both produce non-linear boundaries and score around 0.91–0.98. The kernel trick is computationally cleaner; the explicit expansion is more interpretable.

**RBF kernel — gamma sensitivity**
This is where the bias-variance tradeoff becomes concrete. As gamma increases, each support vector's influence shrinks, and the boundary wraps more tightly around the training data:

- gamma=0.01 → smooth boundary, F1=0.976
- gamma=0.1 → still reasonable, F1=0.965
- gamma=1 → the model starts memorizing, F1 collapses to 0.091
- gamma=10 → complete overfitting, F1=0.0 (predicts one class for everything)

Visualizing the decision boundaries at each gamma value makes this progression clear in a way that a table alone doesn't capture.

## Project Structure

```
Breast_cancer.ipynb     # Full notebook with visualizations and commentary
breast-cancer.csv       # Wisconsin Diagnostic Breast Cancer dataset
README.md
```

## Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

---
---

# Detección de Cáncer de Mama – Español
**Wisconsin Diagnostic Breast Cancer Dataset · Support Vector Machines · scikit-learn**

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## Tabla de Contenidos

- [De qué se trata](#de-qué-se-trata)
- [Resumen de Resultados](#resumen-de-resultados)
- [Por qué F1 Score](#por-qué-f1-score)
- [Qué muestra cada experimento](#qué-muestra-cada-experimento)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Stack](#stack-1)

---

## De qué se trata

Este proyecto entrena un clasificador binario para distinguir tumores malignos de benignos a partir de mediciones físicas extraídas de imágenes del núcleo celular. El dataset es el conocido Wisconsin Diagnostic Breast Cancer, con 569 pacientes y 30 variables numéricas.

El objetivo principal no es solo obtener un F1 score alto, sino entender *por qué* las SVM se comportan de determinada manera bajo distintas condiciones. El notebook recorre una secuencia deliberada de experimentos: reducir el modelo a dos variables, agregar escalado, cambiar de kernel y llevar gamma a sus valores extremos para ver qué se rompe y por qué.

## Resumen de Resultados

| Configuración | Variables | F1 Score |
|---|---|---|
| SVM lineal, sin escalado | 2 | 0.864 |
| SVM lineal, con escalado | 2 | 0.886 |
| SVM lineal, con escalado | 29 | **0.988** |
| PolynomialFeatures + LinearSVC | 2 | 0.925 |
| Kernel polinomial (grado 3) | 29 | 0.976 |
| Kernel RBF, gamma=0.01 | 29 | 0.976 |
| Kernel RBF, gamma=0.1 | 29 | 0.965 |
| Kernel RBF, gamma=1 | 29 | 0.091 |
| Kernel RBF, gamma=10 | 29 | 0.0 |

El kernel lineal sobre el dataset completo y escalado obtuvo el mejor resultado. Los resultados del RBF con gamma=1 y gamma=10 no son errores: son justamente el punto que se busca demostrar.

## Por qué F1 Score

En una tarea de clasificación médica, la exactitud (*accuracy*) por sí sola es engañosa. Un modelo que prediga "benigno" para todos los pacientes tendría ~63% de exactitud solo por seguir la distribución de clases. Lo que importa es el balance entre:

- **Falsos negativos:** un tumor maligno clasificado como benigno → tratamiento retrasado, peor pronóstico
- **Falsos positivos:** un tumor benigno clasificado como maligno → procedimientos innecesarios, ansiedad del paciente

El F1 Score equilibra Precisión y Recall, lo que lo convierte en una métrica más honesta para este tipo de problema.

## Qué muestra cada experimento

**2 variables, sin escalado**
Comenzar con solo `concavity_mean` y `concave points_mean` permite graficar la frontera de decisión directamente. El modelo funciona, pero la frontera queda en una región donde las clases se superponen, y eso se refleja en el F1 Score.

**Agregando StandardScaler**
La SVM optimiza los márgenes usando distancia euclidiana. Si una variable tiene valores del orden de los miles y otra del orden de las centésimas, la variable de mayor escala domina la geometría del problema, sin importar qué tan informativa sea realmente. El escalado lleva todas las variables al mismo espacio de unidades. Para algoritmos basados en distancia, esto no es opcional.

**Las 29 variables**
Pasar de 2 a 29 variables con el escalado correcto lleva el F1 de 0.886 a 0.988. Cada variable adicional aporta una pequeña porción de información que el modelo de dos variables no puede recuperar.

**Kernel polinomial**
Se comparan dos enfoques: la expansión explícita de variables con `PolynomialFeatures + LinearSVC`, y el *kernel trick* implícito con `SVC(kernel='poly')`. Ambos producen fronteras no lineales y obtienen puntajes entre 0.91 y 0.98. El *kernel trick* es computacionalmente más eficiente; la expansión explícita es más interpretable.

**Kernel RBF — sensibilidad a gamma**
Aquí es donde el trade-off entre sesgo y varianza se vuelve concreto. A medida que gamma aumenta, la influencia de cada vector de soporte se reduce, y la frontera se ajusta cada vez más estrechamente a los datos de entrenamiento:

- gamma=0.01 → frontera suave, F1=0.976
- gamma=0.1 → todavía razonable, F1=0.965
- gamma=1 → el modelo empieza a memorizar, el F1 cae a 0.091
- gamma=10 → sobreajuste total, F1=0.0 (predice una sola clase para todo)

Visualizar las fronteras de decisión para cada valor de gamma hace esta progresión mucho más clara de lo que una tabla por sí sola puede mostrar.

## Estructura del Proyecto

```
Breast_cancer.ipynb     # Notebook completo con visualizaciones y comentarios
breast-cancer.csv       # Dataset Wisconsin Diagnostic Breast Cancer
README.md
```

## Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`
