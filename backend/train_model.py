import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\Batman\Desktop\diabetes\backend\diabetes.csv')
# 2. Print columns for debugging
print("Columns in dataset:", df.columns)

# 3. Check if 'diabetes' column exists (target variable)
if 'diabetes' not in df.columns:
    print("❌ Error: 'diabetes' column not found in the dataset.")
    exit()

# 4. Clean the dataset (Optional)
df = df.dropna()  # Drop any rows with missing values (optional)
df.columns = df.columns.str.strip().str.lower()  # Standardize column names

# 5. Convert categorical columns to numeric using Label Encoding
label_encoder = LabelEncoder()

# Encode categorical columns
categorical_columns = ['gender', 'smoking_history']  # Add other categorical columns as necessary

for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# 6. Prepare features and target
X = df.drop('diabetes', axis=1)  # Features
y = df['diabetes']  # Target variable (diabetes column)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model performance storage
model_names = []
model_accuracies = []

# 8. Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
joblib.dump(log_model, 'logistic_model.pkl')  # Save the trained model
print("✅ Logistic Regression model saved as logistic_model.pkl")

# Evaluate Logistic Regression model
y_pred_log = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, y_pred_log)
model_names.append('Logistic Regression')
model_accuracies.append(log_accuracy)

# Print classification report
print(f"Logistic Regression accuracy: {log_accuracy * 100:.2f}%")
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_log))  # Detailed performance

# 9. Train Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
joblib.dump(tree_model, 'decision_tree_model.pkl')  # Save the trained model
print("✅ Decision Tree model saved as decision_tree_model.pkl")

# Evaluate Decision Tree model
y_pred_tree = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)
model_names.append('Decision Tree')
model_accuracies.append(tree_accuracy)

# Print classification report
print(f"Decision Tree accuracy: {tree_accuracy * 100:.2f}%")
print("Classification Report for Decision Tree:")
print(classification_report(y_test, y_pred_tree))  # Detailed performance

# 10. Train Random Forest model
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Optionally tune hyperparameters
forest_model.fit(X_train, y_train)
joblib.dump(forest_model, 'random_forest_model.pkl')  # Save the trained model
print("✅ Random Forest model saved as random_forest_model.pkl")

# Evaluate Random Forest model
y_pred_forest = forest_model.predict(X_test)
forest_accuracy = accuracy_score(y_test, y_pred_forest)
model_names.append('Random Forest')
model_accuracies.append(forest_accuracy)

# Print classification report
print(f"Random Forest accuracy: {forest_accuracy * 100:.2f}%")
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_forest))  # Detailed performance

print("🎉 All models trained, evaluated, and saved successfully!")

# Plotting performance (Accuracy Comparison)
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=model_accuracies, palette='Blues_d')
plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Diabetic', 'Diabetic'], yticklabels=['Not Diabetic', 'Diabetic'])
    plt.title(f'Confusion Matrix: {model_name}', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrix(log_model, X_test, y_test, 'Logistic Regression')
plot_confusion_matrix(tree_model, X_test, y_test, 'Decision Tree')
plot_confusion_matrix(forest_model, X_test, y_test, 'Random Forest')
