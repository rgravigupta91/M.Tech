import pandas as pd

# Sample datasets
data1 = {
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 22, 27]
}

data2 = {
    'ID': [3, 4, 5, 6],
    'Name': ['Charlie', 'David', 'Eve', 'Frank'],
    'Age': [22, 27, 35, 40]
}

dataset1 = pd.DataFrame(data1)
dataset2 = pd.DataFrame(data2)

# Concatenate the two datasets
integrated_dataset = pd.concat([dataset1, dataset2], ignore_index=True)

# Display the integrated dataset before handling redundancy
print("Integrated Dataset Before Handling Redundancy:")
print(integrated_dataset)

# Handling redundancy by removing duplicate records based on 'ID'
integrated_dataset_deduplicated = integrated_dataset.drop_duplicates(subset='ID', keep='first')

# Display the integrated dataset after handling redundancy
print("\nIntegrated Dataset After Handling Redundancy:")
print(integrated_dataset_deduplicated)
