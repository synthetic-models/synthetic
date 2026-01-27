# 🧪 Synthetic Package Release - Alpha Testing Now Available!

Hi everyone, 

I am excited to announce the alpha release of **Synthetic**, a virtual cell generation library for creating biological synthetic data to benchmark our computational modellng workflows. I have given numerous presentations in the past few months about the generation of synthetic models, and I am thrilled to finally share it with you all for testing and feedback.

## Unpublished work warning 

This is an alpha release of an unpublished work. Please do not share or distribute this package outside of our lab until it has been formally published. If you find any bugs or have suggestions, please report them through our lab's GitHub repository. Thanks! 

## What is Synthetic?

Synthetic generates synthetic data using ODE models based on biochemical laws common in cancer cell signaling networks. Since it is capable of generating models of arbitrary sizes, it can be used as a training ground to test the capability of computational workflows and identify current method's weaknesses. Synthetic is aimed to help on multiple fronts in computational oncology research, including:

- Benchmarking machine learning algorithms for cellular response prediction
- Testing parameter estimation and system identification methods for experimental calibration of ODE models
- Developing and validating data analysis workflows with known prior knowledge

## Installation

### For Lab Members (Alpha Testing)

Install directly from our lab's GitHub repository:

```bash
pip install git+https://github.com/IntegratedNetworkModellingLab/Synthetic.git
```

To install a specific version:

```bash
pip install git+https://github.com/IntegratedNetworkModellingLab/Synthetic.git@v0.1.0
```

For lab members with write access, use SSH:

```bash
pip install git+ssh://git@github.com/IntegratedNetworkModellingLab/Synthetic.git
```

## Quick Start

Get up and running in just a few lines:

```python
from synthetic import Builder, make_dataset_drug_response

# Create a virtual cell model
vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)

# Generate a sklearn-compatible dataset
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')

print(f"Feature matrix shape: {X.shape}")  # (1000, n_features)
print(f"Target vector shape: {y.shape}")    # (1000,)
```

## Example Workflow: Analyzing Feature Correlations

Here's a complete workflow showing how to generate a synthetic drug response dataset, analyze feature correlations, and export the data for further analysis:

```python
from synthetic import Builder, make_dataset_drug_response
import pandas as pd

# Create a virtual cell with network topology
# [3, 10, 20] means: 3 cascades of degree 1, 10 cascades of degree 2, 20 cascades of degree 3
vc = Builder.specify([3, 10, 20])

# Generate dataset with 1000 samples
X, y = make_dataset_drug_response(1000, cell_model=vc)

print("Feature data shape:", X.shape)
print("Target data shape:", y.shape)

# Display feature names (first 20 if there are many)
if X.shape[1] <= 20:
    print("Feature names:", X.columns.tolist())
else:
    print("Feature names (first 20):", X.columns.tolist()[:20])

# Calculate Pearson correlation between features and target
correlations = X.apply(lambda col: col.corr(y))
correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

print("\nFeature correlations with target (top 20):")
print(correlations.head(20))

# Export data to CSV files for further analysis
# Features file
X.to_csv('synthetic_features.csv', index=False)
print(f"\n✓ Features exported to 'synthetic_features.csv'")

# Target file
y.to_csv('synthetic_targets.csv', index=False, header=['target'])
print(f"✓ Targets exported to 'synthetic_targets.csv'")

# Combined file for convenience
combined = pd.concat([X, y], axis=1)
combined.to_csv('synthetic_dataset.csv', index=False)
print(f"✓ Combined dataset exported to 'synthetic_dataset.csv'")

# Export correlation analysis
correlations.to_csv('feature_correlations.csv', header=['correlation'])
print(f"✓ Correlations exported to 'feature_correlations.csv'")
```

```output
Feature correlations with target (first 20):
I1_1    -0.552031
I1_3    -0.460993
I1_2    -0.381278
R1_1    -0.331594
R1_3    -0.286238
I3_11   -0.130051
R3_9    -0.122845
R3_15   -0.113456
R3_10   -0.113217
I3_17   -0.111442
I2_8    -0.109473
I2_10   -0.100148
I2_1    -0.089138
I3_1    -0.087076
R2_4     0.086490
I3_4    -0.086017
I2_2    -0.077523
I3_15   -0.076388
R3_20    0.075960
R3_7     0.075231
```

## Example: Combination Therapy

One of the powerful features of Synthetic is its flexibility for testing combination therapy workflows. You can add multiple drugs, each targeting different pathways in the network. Here's an example of creating a combination therapy with two drugs targeting separate I1_x (degree 1 intermediate) pathways:

```python
from synthetic import Builder, make_dataset_drug_response
import pandas as pd

# Create a virtual cell without auto-drug (we'll add drugs manually)
# [5, 10, 15] creates 5 cascades of degree 1, 10 of degree 2, 15 of degree 3
vc = Builder.specify(
    [5, 10, 15],
    auto_drug=False,  # Disable auto-generated drug
    auto_compile=False  # We'll compile after adding drugs
)

# Add Drug A: targets I1_1 and I1_2 (down-regulates these intermediates)
vc.add_drug(
    name="Drug_A",
    start_time=5000.0,
    default_value=0.0,
    regulation=["I1_1", "I1_2"],
    regulation_type=["down", "down"],
    value=100.0
)

# Add Drug B: targets I1_3 and I1_4 (down-regulates these intermediates)
vc.add_drug(
    name="Drug_B",
    start_time=5000.0,
    default_value=0.0,
    regulation=["I1_3", "I1_4"],
    regulation_type=["down", "down"],
    value=100.0
)

# List all drugs in the system
drugs = vc.list_drugs()
print("Drugs in the system:")
for drug in drugs:
    print(f"  {drug['name']}: targets {drug['targets']} ({drug['types']})")

# Compile the model with both drugs applied
vc.compile()

# Generate combination therapy dataset
X, y = make_dataset_drug_response(
    n=500,
    cell_model=vc,
    target_specie='Oa',
    seed=42
)

print(f"\nDataset shape: {X.shape} samples, {X.shape[1]} features")

# Calculate correlations to see which features are most predictive
correlations = X.apply(lambda col: col.corr(y))
correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

print("\nTop 15 features correlated with target:")
print(correlations.head(15))

# Export combination therapy data
X.to_csv('combo_therapy_features.csv', index=False)
y.to_csv('combo_therapy_targets.csv', index=False, header=['target'])
combined = pd.concat([X, y], axis=1)
combined.to_csv('combo_therapy_dataset.csv', index=False)

print("\n✓ Combination therapy data exported to CSV files")
```

### How This Works

1. **Disable auto-drug**: Set `auto_drug=False` to prevent automatic drug generation
2. **Add Drug A**: Targets I1_1 and I1_2 with down-regulation
3. **Add Drug B**: Targets I1_3 and I1_4 with down-regulation
4. **Compile**: Applies both drugs to the model
5. **Generate data**: Creates a dataset reflecting the combined effect of both drugs

This flexibility allows you to:
- Test drug combinations with different targets
- Model synergistic or antagonistic effects
- Compare single-drug vs. combination therapies
- Validate combination therapy prediction algorithms

## Feedback

As this is an alpha release, we welcome your feedback! Please report issues or suggest features through the lab's GitHub repository.

---

**Happy modeling! 🚀**