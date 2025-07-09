import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# load dataset
train_file_path = 'Train_Dataset.csv'
test_file_path = 'Test_Dataset.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# handling missing values
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# label encoding for cat values 
categorical_cols = train_data.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()

for col in categorical_cols:
    train_data[col] = label_encoder.fit_transform(train_data[col].astype(str))
    if col in test_data.columns:
        test_data[col] = label_encoder.transform(test_data[col].astype(str))

# split features and target from train data
X = train_data.drop(columns=['EmployeeID', 'Attrition'])
y = train_data['Attrition']

# split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# build the gradient boosting vlassifier
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# predictions on the validation set
y_val_pred_gb = gb_model.predict(X_val)

# accuracy for GBC
val_accuracy_gb = accuracy_score(y_val, y_val_pred_gb)

# predictions on the test set using GBC
test_features = test_data.drop(columns=['EmployeeID'])
test_predictions_gb = gb_model.predict(test_features)

# submission dataframe for GBC
submission_df_gb = pd.DataFrame({
    'EmployeeID': test_data['EmployeeID'],
    'Attrition': test_predictions_gb
})

# Save the file for GBC
submission_file_path_gb = 'submission_attrition_model_v4.csv'
submission_df_gb.to_csv(submission_file_path_gb, index=False)

val_accuracy_gb, submission_file_path_gb