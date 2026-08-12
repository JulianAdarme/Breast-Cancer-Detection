# Credit Card Default Prediction

A binary classification project that predicts whether a credit card customer will default on their next monthly payment, using the UCI *Default of Credit Card Clients* dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Business Problem & Evaluation Metric](#business-problem--evaluation-metric)
- [Project Workflow](#project-workflow)
- [Key Results](#key-results)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations & Next Steps](#limitations--next-steps)

---

## Overview

Credit default prediction is one of the most important applications of machine learning in retail banking. Accurately identifying customers likely to default enables financial institutions to reduce credit risk, optimize lending decisions, and improve portfolio profitability.

This project builds a binary classification model using demographic information, credit limits, billing statements, payment history, and engineered financial indicators. Beyond producing a predictive model, it also investigates:

- **Feature engineering:** Do business-oriented features improve predictive performance?
- **Feature aggregation:** Can monthly billing and payment histories be summarized without losing important information?
- **Model comparison:** Logistic Regression vs. Random Forest, evaluated under the same experimental framework.
- **Model optimization:** Cross-validation, hyperparameter tuning, and decision-threshold optimization to maximize the F1-score.

## Dataset

Source: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

- **30,000** credit card clients
- **23** predictor variables (demographics, credit limit, 6 months of billing statements, 6 months of payment history)
- **1** binary target: default on next monthly payment

| Class | Label | Count | Proportion |
|---|---|---|---|
| Non-Default | 0 | 23,364 | 77.88% |
| Default | 1 | 6,636 | 22.12% |

## Business Problem & Evaluation Metric

In credit risk, both types of classification error carry financial consequences:

- A **false negative** (predicting no default when the customer actually defaults) increases expected credit losses.
- A **false positive** (predicting default for a customer who would have repaid) risks rejecting a creditworthy customer or imposing unnecessarily restrictive credit terms.

Since both error types are costly, **F1-score** was chosen as the primary evaluation metric, as it balances precision and recall rather than favoring one at the expense of the other.

## Project Workflow

1. Dataset & Business Understanding
2. Exploratory Data Analysis (EDA) — SQL-based exploration + statistical/visual analysis
3. Feature Engineering
4. Data Preprocessing
5. Baseline Models
6. Logistic Regression (across 3 engineered feature sets)
7. Random Forest (across 3 engineered feature sets)
8. Feature Importance / Reduced Feature Set
9. Cross-Validation
10. Hyperparameter Optimization (Grid Search)
11. Decision Threshold Optimization
12. Final Evaluation on the Test Set
13. Conclusions

Three progressively engineered feature sets were compared throughout the notebook:

| Dataset | Description |
|---|---|
| Baseline | Raw dataset (with `EDUCATION` / `MARRIAGE` label corrections only) |
| Dataset 1 | Baseline + `Utilization_ratio` and `Delinq_month_count` |
| Dataset 2 | Dataset 1 + `AVG_payment` and `AVG_BILL` (aggregated monthly features) |
| Dataset 3 | Dataset 2 with raw monthly `BILL_AMT1–6` / `PAY_AMT1–6` columns removed |

## Key Results

| Model | Cross-Validation F1 (default) | Cross-Validation F1 (tuned) | Test F1 (optimized threshold) |
|---|---|---|---|
| Logistic Regression | 0.523 | 0.524 | **0.536** |
| Random Forest | 0.442 | 0.542 | **0.555** |

- **Random Forest** achieved the highest F1-score on the held-out test set (**0.555**), after hyperparameter tuning and decision-threshold optimization.
- **Logistic Regression** showed very consistent performance between cross-validation and test (0.524 → 0.536), indicating reliable generalization.
- Random Forest's ranking relative to Logistic Regression flipped **during hyperparameter tuning** (before ever touching the test set) — its default-hyperparameter CV score was lower than Logistic Regression's, but its tuned CV score was higher.

## Key Findings

- The dataset is moderately imbalanced (≈22% default cases).
- Customers with lower credit limits tend to exhibit higher default rates (confirmed with a Mann–Whitney U test, p < 0.001).
- Payment history variables (`PAY_0`, `PAY_2`–`PAY_6`) show the strongest relationship with the target.
- Business-oriented engineered features (`Utilization_ratio`, `Delinq_month_count`) improved performance; aggregated monthly averages (`AVG_BILL`, `AVG_payment`) did not — the detailed monthly history carries temporal information that simple averages don't capture.
- Restricting the model to the top 13 most important features produced nearly identical cross-validation performance to the full feature set, suggesting predictive information is concentrated in a relatively small subset of variables.

## Tech Stack

- Python (pandas, NumPy)
- scikit-learn (Logistic Regression, Random Forest, `Pipeline`, `GridSearchCV`, `StratifiedKFold`)
- SciPy (Mann–Whitney U test)
- SQLite (SQL-based exploratory analysis)
- Matplotlib, Seaborn

## Repository Structure

```
.
├── credit_card_default.ipynb   # Main analysis notebook
├── credit_default.csv          # Dataset (UCI Credit Card Default)
└── README.md
```

## How to Run

```bash
git clone <repo-url>
cd <repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook credit_card_default.ipynb
```

## Limitations & Next Steps

This project is intended as a demonstration of an end-to-end classification workflow, not a production-ready credit risk model. Before any real-world use, it would require:

- Probability calibration
- Cost-sensitive learning aligned to the bank's actual false-negative / false-positive cost ratio
- A fairness / disparate-impact audit across demographic variables (`SEX`, `AGE`, `EDUCATION`, `MARRIAGE`)
- External validation on more recent data and periodic retraining

---
---

# Predicción de Incumplimiento de Pago en Tarjetas de Crédito

Un proyecto de clasificación binaria que predice si un cliente de tarjeta de crédito incumplirá su próximo pago mensual, utilizando el dataset *Default of Credit Card Clients* de la UCI.

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Dataset](#dataset-1)
- [Problema de Negocio y Métrica de Evaluación](#problema-de-negocio-y-métrica-de-evaluación)
- [Flujo de Trabajo del Proyecto](#flujo-de-trabajo-del-proyecto)
- [Resultados Principales](#resultados-principales)
- [Hallazgos Clave](#hallazgos-clave)
- [Stack Tecnológico](#stack-tecnológico)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Cómo Ejecutarlo](#cómo-ejecutarlo)
- [Limitaciones y Próximos Pasos](#limitaciones-y-próximos-pasos)

---

## Descripción General

La predicción de incumplimiento de pago en tarjetas de crédito es una de las aplicaciones más importantes del machine learning en la banca minorista. Identificar con precisión a los clientes con mayor probabilidad de incumplir permite a las instituciones financieras reducir el riesgo crediticio, optimizar las decisiones de otorgamiento de crédito y mejorar la rentabilidad de su portafolio.

Este proyecto construye un modelo de clasificación binaria utilizando información demográfica, límites de crédito, estados de cuenta, historial de pagos e indicadores financieros construidos a partir de estas variables. Más allá de producir un modelo predictivo, el proyecto también explora:

- **Ingeniería de características:** ¿Las variables orientadas al negocio mejoran el desempeño predictivo?
- **Agregación de características:** ¿Puede resumirse el historial mensual de facturación y pagos sin perder información relevante?
- **Comparación de modelos:** Regresión Logística vs. Random Forest, evaluados bajo el mismo marco experimental.
- **Optimización de modelos:** Validación cruzada, ajuste de hiperparámetros y optimización del umbral de decisión para maximizar el F1-score.

## Dataset

Fuente: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

- **30,000** clientes de tarjetas de crédito
- **23** variables predictoras (datos demográficos, límite de crédito, 6 meses de estados de cuenta, 6 meses de historial de pagos)
- **1** variable objetivo binaria: incumplimiento del próximo pago mensual

| Clase | Etiqueta | Cantidad | Proporción |
|---|---|---|---|
| No incumplimiento | 0 | 23,364 | 77.88% |
| Incumplimiento | 1 | 6,636 | 22.12% |

## Problema de Negocio y Métrica de Evaluación

En el riesgo crediticio, ambos tipos de error de clasificación tienen consecuencias financieras:

- Un **falso negativo** (predecir que un cliente no incumplirá cuando en realidad sí lo hace) incrementa las pérdidas crediticias esperadas.
- Un **falso positivo** (predecir incumplimiento para un cliente que sí habría pagado) puede llevar a rechazar clientes confiables o a imponer condiciones de crédito innecesariamente restrictivas.

Dado que ambos tipos de error son costosos, se eligió el **F1-score** como métrica principal de evaluación, ya que equilibra precisión y sensibilidad (*recall*) en lugar de favorecer una a costa de la otra.

## Flujo de Trabajo del Proyecto

1. Comprensión del Dataset y el Negocio
2. Análisis Exploratorio de Datos (EDA) — exploración basada en SQL + análisis estadístico/visual
3. Ingeniería de Características
4. Preprocesamiento de Datos
5. Modelos Base (Baseline)
6. Regresión Logística (sobre 3 conjuntos de variables)
7. Random Forest (sobre 3 conjuntos de variables)
8. Importancia de Variables / Conjunto Reducido
9. Validación Cruzada
10. Optimización de Hiperparámetros (Grid Search)
11. Optimización del Umbral de Decisión
12. Evaluación Final en el Conjunto de Prueba
13. Conclusiones

A lo largo del notebook se compararon tres conjuntos de variables, construidos de forma progresiva:

| Dataset | Descripción |
|---|---|
| Baseline | Dataset original (solo con la corrección de etiquetas de `EDUCATION` / `MARRIAGE`) |
| Dataset 1 | Baseline + `Utilization_ratio` y `Delinq_month_count` |
| Dataset 2 | Dataset 1 + `AVG_payment` y `AVG_BILL` (variables mensuales agregadas) |
| Dataset 3 | Dataset 2 sin las columnas mensuales originales `BILL_AMT1–6` / `PAY_AMT1–6` |

## Resultados Principales

| Modelo | F1 Validación Cruzada (por defecto) | F1 Validación Cruzada (ajustado) | F1 Test (umbral optimizado) |
|---|---|---|---|
| Regresión Logística | 0.523 | 0.524 | **0.536** |
| Random Forest | 0.442 | 0.542 | **0.555** |

- **Random Forest** obtuvo el F1-score más alto en el conjunto de prueba (**0.555**), tras la optimización de hiperparámetros y del umbral de decisión.
- **Regresión Logística** mostró un desempeño muy consistente entre validación cruzada y test (0.524 → 0.536), lo que indica una buena capacidad de generalización.
- El orden entre Random Forest y Regresión Logística cambió **durante la optimización de hiperparámetros** (antes de llegar al conjunto de prueba): con hiperparámetros por defecto, el CV de Random Forest era menor que el de Regresión Logística, pero tras el ajuste pasó a ser mayor.

## Hallazgos Clave

- El dataset presenta un desbalance moderado (≈22% de casos de incumplimiento).
- Los clientes con límites de crédito más bajos tienden a presentar tasas de incumplimiento más altas (confirmado con una prueba de Mann–Whitney U, p < 0.001).
- Las variables de historial de pago (`PAY_0`, `PAY_2`–`PAY_6`) muestran la relación más fuerte con la variable objetivo.
- Las variables orientadas al negocio (`Utilization_ratio`, `Delinq_month_count`) mejoraron el desempeño; los promedios mensuales agregados (`AVG_BILL`, `AVG_payment`) no lo hicieron — el historial mensual detallado contiene información temporal que los promedios simples no capturan.
- Restringir el modelo a las 13 variables más importantes produjo un desempeño de validación cruzada prácticamente idéntico al del conjunto completo, lo que sugiere que la información predictiva se concentra en un subconjunto relativamente pequeño de variables.

## Stack Tecnológico

- Python (pandas, NumPy)
- scikit-learn (Regresión Logística, Random Forest, `Pipeline`, `GridSearchCV`, `StratifiedKFold`)
- SciPy (prueba de Mann–Whitney U)
- SQLite (análisis exploratorio basado en SQL)
- Matplotlib, Seaborn

## Estructura del Repositorio

```
.
├── credit_card_default.ipynb   # Notebook principal del análisis
├── credit_default.csv          # Dataset (UCI Credit Card Default)
└── README.md
```

## Cómo Ejecutarlo

```bash
git clone <repo-url>
cd <carpeta-del-repo>
pip install pandas numpy matplotlib seaborn scikit-learn scipy
jupyter notebook credit_card_default.ipynb
```

## Limitaciones y Próximos Pasos

Este proyecto está pensado como una demostración de un flujo de trabajo de clasificación de principio a fin, no como un modelo de riesgo crediticio listo para producción. Antes de cualquier uso real, sería necesario:

- Calibración de probabilidades
- Aprendizaje sensible al costo, alineado con el costo real de falsos negativos y falsos positivos para el banco
- Una auditoría de equidad / impacto diferencial sobre variables demográficas (`SEX`, `AGE`, `EDUCATION`, `MARRIAGE`)
- Validación externa con datos más recientes y reentrenamiento periódico
