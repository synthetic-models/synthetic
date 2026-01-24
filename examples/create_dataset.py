from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify([1,2,5])

X, y = make_dataset_drug_response(1000, cell_model=vc)

print("Feature data shape:", X.shape)
print("Target data shape:", y.shape)

