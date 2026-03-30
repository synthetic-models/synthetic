from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify([3,5])

X, y = make_dataset_drug_response(10, cell_model=vc, verbose=True, n_cores=-1)

print("Feature data shape:", X.shape)
print("Target data shape:", y.shape)

# print all feature names, perhaps limit to first 20 if too many, if only the list is long
if X.shape[1] <= 20:
    print("Feature names:", X.columns.tolist())
else:
    print("Feature names (first 20):", X.columns.tolist()[:20])
    
    
# perform pearson correlation between features and target
correlations = X.apply(lambda col: col.corr(y))
# sort correlations by absolute value descending
correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
print("Feature correlations with target (first 20):")
print(correlations.head(20))