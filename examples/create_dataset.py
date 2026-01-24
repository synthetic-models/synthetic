from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify([1,2,5])

X, y = make_dataset_drug_response(100, cell_model=vc)

print("Feature data shape:", X.shape)
print("Target data shape:", y.shape)

# print all feature names, perhaps limit to first 20 if too many, if only the list is long
if X.shape[1] <= 20:
    print("Feature names:", X.columns.tolist())
else:
    print("Feature names (first 20):", X.columns.tolist()[:20])