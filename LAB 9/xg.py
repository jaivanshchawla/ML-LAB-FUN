import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
path = r"C:\Users\Jaivansh Chawla\Documents\COLLEGE\3 YEAR\5 sem\ML\ML LAB\LAB 9\UCI_HAR_dataset.csv"
df = pd.read_csv(path)
print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}\n")

# ============================================
# 1. EXPLORATORY DATA ANALYSIS
# ============================================

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
df['Activity'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Activity Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Activity')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Pie chart
df['Activity'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Activity Proportion', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.show()

# Feature correlation (sample of first 20 features for readability)
plt.figure(figsize=(12, 8))
correlation = df.iloc[:, :20].corr()
sns.heatmap(correlation, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Heatmap (First 20 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 2. DATA PREPARATION
# ============================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# ============================================
# 3. MODEL TRAINING
# ============================================

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("🚀 Model Training Complete!\n")

# ============================================
# 4. MODEL EVALUATION
# ============================================

# Accuracy and Classification Report
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(
    y_test, y_pred, target_names=le.classes_
))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_title("Confusion Matrix", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Normalized Confusion Matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1].set_title("Normalized Confusion Matrix", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ============================================
# 5. FEATURE IMPORTANCE
# ============================================

# Top 20 most important features
feature_importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ============================================
# 6. ROC CURVES (Multi-class)
# ============================================

# Binarize the output for ROC curve
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))
y_pred_proba = model.predict_proba(X_test)

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

for i, (color, activity) in enumerate(zip(colors, le.classes_)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{activity} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for Each Activity Class', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# 7. PER-CLASS METRICS VISUALIZATION
# ============================================

report_dict = classification_report(y_test, y_pred, 
                                    target_names=le.classes_, 
                                    output_dict=True)

metrics_df = pd.DataFrame(report_dict).transpose()[:-3]  # Exclude avg rows

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['precision', 'recall', 'f1-score']

for idx, metric in enumerate(metrics):
    axes[idx].bar(metrics_df.index, metrics_df[metric], color='teal', alpha=0.7)
    axes[idx].set_title(f'{metric.capitalize()} by Activity', fontweight='bold')
    axes[idx].set_ylabel(metric.capitalize())
    axes[idx].set_ylim([0, 1.1])
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Analysis Complete!")