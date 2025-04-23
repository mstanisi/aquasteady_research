# AquaSteady USDA Irrigation Analysis

## Project Overview

This repository contains my work as a research assistant for the NSF-funded AquaSteady project, analyzing 20 years of USDA census data to identify irrigation and sustainability trends among American farms.

AquaSteady is developing a seaweed-based hydrogel designed to help farms maintain soil moisture and withstand irregular weather patterns caused by climate change. My analysis focused on identifying:
- Nationwide irrigation challenges
- Geographic patterns in sustainability struggles
- Key factors driving irrigation difficulties

## Key Findings

1. **Multi-correlated failures**: Farms nationwide were consistently failing to meet irrigation standards across multiple metrics
2. **Critical metrics**: Acres irrigated provided the clearest signal among noisy data
3. **Geographic hotspots**: Cluster analysis revealed California and the Texas Gulf as significant outlier regions
4. **Growing barriers**: Linear regression identified crop yield risk and financial constraints as the fastest growing sustainability challenges

## Methodology

1. **Data Preparation**
   - Cleaned 20 years of USDA census data
   - Identified acres irrigated as the most reliable metric
   - [1. importing and cleaning.ipynb](code/1.%20importing%20and%20cleaning.ipynb)

2. **Feature Selection**
   - Implemented Random Forest Classifier to rank region significance
   - Applied Ridge Regression to identify most impactful features
   - [2. feature selection.ipynb](code/2.%20feature%20selection.ipynb)

3. **Analysis**
   - Performed k-means clustering on geographic regions
   - Conducted final linear regression on key sustainability factors
   - [3. analysis.ipynb](code/3.%20analysis.ipynb)

## Repository Structure

```
├── code/
│   ├── ipynb_checkpoints/
│   ├── 1. importing and cleaning.ipynb
│   ├── 2. feature selection.ipynb
│   └── 3. analysis.ipynb
│
└── data/
    ├── raw/                  # Original USDA data
    ├── transformed/          # Processed datasets
    ├── report/ (TBD)        # Future analysis reports
    ├── visuals/              # Generated visualizations
    │   ├── aquasteady_area.png
    │   ├── cluster.png
    │   ├── finance_regression.png
    │   ├── randomforest.png
    │   ├── ridge_regression.png
    │   └── yield_regression.png
```

## Visualizations

Key outputs included:
- Geographic cluster analysis (`cluster.png`)
- Financial constraint regression (`finance_regression.png`)
- Yield risk regression (`yield_regression.png`)
- Feature importance plots (`randomforest.png`, `ridge_regression.png`)

## Future Work

Areas for further investigation:
- Deeper spatial analysis of identified outlier regions
- Incorporation of climate data to enhance predictive models
- Development of policy recommendations based on findings

## Acknowledgments

This work was supported by the National Science Foundation through the AquaSteady project. Special thanks to my research advisors and the USDA for making this valuable data publicly available.
