# AI Chatbot Prediction

## To-do

### Data

- [X] Split and clean data (Yanzun)

### Model

- [ ] Organize and zip the code and data

#### Naive Bayes

- [ ] Make the function compatible for new data (Yanzun)

#### Logistic Regression

#### Random Forest

### Report

- [ ] Draft the report (Amanda)
- [ ] Check the items in the rubric
- [ ] Cite packages used
- [ ] Convert .docx to .tex (or .rmd/.qmd)

## Overview

This repo contains the prediction of different AI chatbots using various ML models.

## File Structure

The repo is structured as:
-   `code` contains the Python code used to train, validate, and test the model.
-   `data` contains the data used to train the model.
-   `group_contract` contains the files used to generate the group contract, including the LaTeX document the PDF of the group contract.
-   `project_proposal` contains the project proposal in PDF format, and the R script used to clean the dataset.
-   `report` contains the files used to generate the report, including the LaTeX document and reference bibliography file, as well as the PDF of the report.

## Grading Scheme

### (2) Data Exploration

- [ ] (0.5) Summarize the dataset: Clearly describes feature types (numerical, categorical, text), distributions, and class balance.
- [ ] (0.5) Identify and address data issues: Notes missing values, outliers, or inconsistencies, and explains how they are handled.
- [ ] (0.5) Explain preprocessing: Describes transformations (e.g., normalization, encoding, text representation) with justification.
- [ ] (0.5) Prevent data leakage: Explains how the test set was reserved and not used during exploration.

### (4) Methodology

- [ ] (0.5) Model families: States the three model families used and why they fit the dataset.
- [ ] (0.5) Optimization: Describes optimizer (e.g., SGD, Adam) and LR schedule and/or regularization/early stopping.
- [ ] (0.5) Validation method: Explains train/validation split or cross-validation.
- [ ] (0.5) Hyperparameter list: Comprehensive list for each model.
- [ ] (0.5) Hyperparameter choices: Ranges/search strategy with evidence they’re reasonable (validation results, rationale).
- [ ] (0.5) Avoid winner-only tuning: Tuned all three models before selecting the best.
- [ ] (1.0) Evaluation metrics: Describes which metrics were used. Includes at least one metric beyond accuracy and justifies why the chosen metrics are appropriate.

### (2) Results

- [ ] (1.0) Report results clearly: Uses evaluation metrics from Methodology; clear tables/plots; compares model families; states final choice.
- [ ] (0.5) Analyze errors: Identifies common misclassifications/weaknesses; includes confusion matrix or examples if helpful.
- [ ] (0.5) Test performance: Single-number estimate for best model with justification (e.g., CV stability, learning curves).
