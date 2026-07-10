import pandas as pd

# Load datasets
application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

# Create target column
def create_target(status_list):
    bad_status = ['1', '2', '3', '4', '5']

    for status in status_list:
        if status in bad_status:
            return 0      # Rejected

    return 1              # Approved

# Group credit records by customer ID
target = credit.groupby("ID")["STATUS"].apply(list).reset_index()

# Create Approved column
target["Approved"] = target["STATUS"].apply(create_target)

# Keep only ID and target
target = target[["ID", "Approved"]]

# Merge datasets
data = application.merge(target, on="ID", how="inner")

print(data.head())

print("\nShape")
print(data.shape)

print("\nTarget Distribution")
print(data["Approved"].value_counts())

# Save merged dataset
data.to_csv("dataset/final_credit_dataset.csv", index=False)

print("\nDataset saved successfully!")