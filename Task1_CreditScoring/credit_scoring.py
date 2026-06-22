
# CREDIT SCORING MODEL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
columns = [
    'checking_account', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'employment', 'installment_rate',
    'personal_status_sex', 'other_debtors', 'residence_since',
    'property', 'age', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'num_dependents', 'telephone',
    'foreign_worker', 'target'
]

df = pd.read_csv('data/german_credit.csv', sep='\s+', names=columns)

# Convert target: 1 = good, 2 = bad → 1 (good), 0 (bad)
df['target'] = df['target'].map({1: 1, 2: 0})

print("First 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nClass distribution:")
print(df['target'].value_counts())

# Exploratory Data Analysis
sns.countplot(x='target', data=df)
plt.title('Credit Risk Distribution (1=Good, 0=Bad)')
plt.savefig('class_distribution.png')
plt.show()

numeric_cols = ['duration', 'credit_amount', 'installment_rate',
                'residence_since', 'age', 'existing_credits', 'num_dependents']
df[numeric_cols].hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.savefig('numeric_histograms.png')
plt.show()


# Preprocessing & Train/Test Split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

X = df.drop('target', axis=1)
y = df['target']

cat_cols = [
    'checking_account', 'credit_history', 'purpose', 'savings_account',
    'employment', 'personal_status_sex', 'other_debtors', 'property',
    'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker'
]
num_cols = [
    'duration', 'credit_amount', 'installment_rate', 'residence_since',
    'age', 'existing_credits', 'num_dependents'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

#feature names for later interpretation
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(cat_feature_names)
print(f"Number of features after preprocessing: {X_train_processed.shape[1]}")


# Baseline Models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    print(f"Accuracy: {np.mean(y_pred == y_test):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'])
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'confusion_{name.replace(" ", "_").lower()}.png')
    plt.show()


# Handling Imbalance with SMOTE

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

smote_rf = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
smote_rf.fit(X_train_processed, y_train)
y_pred_smote = smote_rf.predict(X_test_processed)
y_proba_smote = smote_rf.predict_proba(X_test_processed)[:, 1]

print("\n--- Random Forest with SMOTE ---")
print(classification_report(y_test, y_pred_smote))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_smote):.3f}")

# --------------------------------
# Hyperparameter Tuning
# --------------------------------
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_processed, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_processed)
y_proba_best = best_rf.predict_proba(X_test_processed)[:, 1]
print("\n--- Best Random Forest (Tuned) ---")
print(classification_report(y_test, y_pred_best))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_best):.3f}")

# Save model and preprocessor
import joblib
joblib.dump(best_rf, 'models/best_rf_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Model and preprocessor saved to 'models/' folder.")

# --------------------------------
# Feature Importance & SHAP (optional)
# --------------------------------
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-10:]   # top 10
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# SHAP (will take a moment)
import shap
explainer = shap.TreeExplainer(best_rf)
# Use a small sample for speed
shap_values = explainer.shap_values(X_test_processed[:100])
shap.summary_plot(shap_values, X_test_processed[:100], feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot')
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.show()