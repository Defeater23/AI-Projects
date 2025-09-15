"""
Cardiovascular Disease Prediction Project
==========================================

This project performs comprehensive analysis and prediction of cardiovascular disease
using multiple machine learning algorithms including SVM, KNN, Decision Trees, 
Logistic Regression, and Random Forest.

Author: AI Assistant
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("CARDIOVASCULAR DISEASE PREDICTION PROJECT")
print("="*60)

# ================================
# STEP 1: DATA LOADING AND EXPLORATION
# ================================

print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

# Load the cardiovascular dataset
try:
    df = pd.read_csv('data/cardio_train.csv', sep=';')
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
except FileNotFoundError:
    print("❌ Dataset not found. Please ensure the dataset is in the data/ directory.")
    exit()

# Display column information
print(f"\nDataset Columns:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

# Display basic information about the dataset
print(f"\nDataset Info:")
print(f"- Number of samples: {df.shape[0]}")
print(f"- Number of features: {df.shape[1]}")
print(f"- Missing values: {df.isnull().sum().sum()}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset description:")
print(df.describe())

# ================================
# STEP 2: DATA PREPROCESSING
# ================================

print("\n\n2. DATA PREPROCESSING")
print("-" * 40)

# Remove the ID column as it's not needed for prediction
if 'id' in df.columns:
    df = df.drop('id', axis=1)
    print("✓ ID column removed")

# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"Missing values found:")
    print(missing_counts[missing_counts > 0])
    
    # Handle missing values appropriately
    df = df.dropna()
    print("✓ Missing values handled by removal")
else:
    print("✓ No missing values found")

# Data preprocessing and feature engineering
print("\nData preprocessing:")

# Convert age from days to years
df['age'] = df['age'] / 365.25
print("✓ Age converted from days to years")

# Create BMI feature
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
print("✓ BMI feature created")

# Create blood pressure categories
df['bp_category'] = 0  # Normal
df.loc[(df['ap_hi'] >= 130) | (df['ap_lo'] >= 80), 'bp_category'] = 1  # High
print("✓ Blood pressure category feature created")

# Rename target column for clarity
df = df.rename(columns={'cardio': 'target'})

# Display target distribution
print(f"\nTarget distribution (Cardiovascular Disease):")
print(df['target'].value_counts())
print(f"Percentage of positive cases: {(df['target'].sum() / len(df)) * 100:.1f}%")

# Display updated dataset info
print(f"\nFinal dataset shape: {df.shape}")
print(f"\nFeature descriptions:")
feature_descriptions = {
    'age': 'Age in years',
    'gender': 'Gender (1=female, 2=male)',
    'height': 'Height in cm',
    'weight': 'Weight in kg',
    'ap_hi': 'Systolic blood pressure',
    'ap_lo': 'Diastolic blood pressure', 
    'cholesterol': 'Cholesterol level (1=normal, 2=above normal, 3=well above normal)',
    'gluc': 'Glucose level (1=normal, 2=above normal, 3=well above normal)',
    'smoke': 'Smoking (0=no, 1=yes)',
    'alco': 'Alcohol intake (0=no, 1=yes)',
    'active': 'Physical activity (0=no, 1=yes)',
    'bmi': 'Body Mass Index (calculated)',
    'bp_category': 'Blood pressure category (0=normal, 1=high)',
    'target': 'Cardiovascular disease presence (0=no, 1=yes)'
}

for feature, description in feature_descriptions.items():
    if feature in df.columns:
        print(f"  {feature}: {description}")

# ================================
# STEP 3: EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# ================================

print("\n\n3. EXPLORATORY DATA ANALYSIS AND VISUALIZATION")
print("-" * 40)

# Create comprehensive visualizations
fig_size = (15, 12)

# 1. Distribution of target variable
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
target_counts = df['target'].value_counts()
plt.pie(target_counts.values, labels=['No CVD', 'CVD Present'], autopct='%1.1f%%', 
        colors=['lightblue', 'lightcoral'])
plt.title('Distribution of Cardiovascular Disease')

# 2. Age distribution by target
plt.subplot(2, 3, 2)
plt.hist([df[df['target']==0]['age'], df[df['target']==1]['age']], 
         bins=20, label=['No CVD', 'CVD Present'], alpha=0.7)
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.title('Age Distribution by CVD Status')
plt.legend()

# 3. Gender distribution
plt.subplot(2, 3, 3)
gender_disease = pd.crosstab(df['gender'], df['target'])
gender_disease.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
plt.xlabel('Gender (1=Female, 2=Male)')
plt.ylabel('Count')
plt.title('CVD by Gender')
plt.legend(['No CVD', 'CVD Present'])
plt.xticks(rotation=0)

# 4. BMI distribution
plt.subplot(2, 3, 4)
plt.boxplot([df[df['target']==0]['bmi'], df[df['target']==1]['bmi']], 
            labels=['No CVD', 'CVD Present'])
plt.ylabel('BMI')
plt.title('BMI Distribution by CVD Status')

# 5. Blood pressure - Systolic
plt.subplot(2, 3, 5)
plt.boxplot([df[df['target']==0]['ap_hi'], df[df['target']==1]['ap_hi']], 
            labels=['No CVD', 'CVD Present'])
plt.ylabel('Systolic BP')
plt.title('Systolic Blood Pressure by CVD Status')

# 6. Cholesterol levels
plt.subplot(2, 3, 6)
chol_disease = pd.crosstab(df['cholesterol'], df['target'])
chol_disease.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
plt.xlabel('Cholesterol Level (1=Normal, 2=Above Normal, 3=Well Above)')
plt.ylabel('Count')
plt.title('CVD by Cholesterol Level')
plt.legend(['No CVD', 'CVD Present'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('visualizations/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional detailed visualizations
plt.figure(figsize=(16, 12))

# Lifestyle factors analysis
plt.subplot(2, 3, 1)
smoke_disease = pd.crosstab(df['smoke'], df['target'])
smoke_disease.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
plt.xlabel('Smoking (0=No, 1=Yes)')
plt.ylabel('Count')
plt.title('CVD by Smoking Status')
plt.legend(['No CVD', 'CVD Present'])
plt.xticks(rotation=0)

plt.subplot(2, 3, 2)
alco_disease = pd.crosstab(df['alco'], df['target'])
alco_disease.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
plt.xlabel('Alcohol (0=No, 1=Yes)')
plt.ylabel('Count')
plt.title('CVD by Alcohol Consumption')
plt.legend(['No CVD', 'CVD Present'])
plt.xticks(rotation=0)

plt.subplot(2, 3, 3)
active_disease = pd.crosstab(df['active'], df['target'])
active_disease.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
plt.xlabel('Physical Activity (0=No, 1=Yes)')
plt.ylabel('Count')
plt.title('CVD by Physical Activity')
plt.legend(['No CVD', 'CVD Present'])
plt.xticks(rotation=0)

plt.subplot(2, 3, 4)
gluc_disease = pd.crosstab(df['gluc'], df['target'])
gluc_disease.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
plt.xlabel('Glucose Level (1=Normal, 2=Above Normal, 3=Well Above)')
plt.ylabel('Count')
plt.title('CVD by Glucose Level')
plt.legend(['No CVD', 'CVD Present'])
plt.xticks(rotation=0)

# Weight and height distributions
plt.subplot(2, 3, 5)
plt.scatter(df[df['target']==0]['height'], df[df['target']==0]['weight'], 
           alpha=0.5, label='No CVD', color='lightblue')
plt.scatter(df[df['target']==1]['height'], df[df['target']==1]['weight'], 
           alpha=0.5, label='CVD Present', color='lightcoral')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight by CVD Status')
plt.legend()

# Blood pressure scatter
plt.subplot(2, 3, 6)
plt.scatter(df[df['target']==0]['ap_hi'], df[df['target']==0]['ap_lo'], 
           alpha=0.5, label='No CVD', color='lightblue')
plt.scatter(df[df['target']==1]['ap_hi'], df[df['target']==1]['ap_lo'], 
           alpha=0.5, label='CVD Present', color='lightcoral')
plt.xlabel('Systolic BP')
plt.ylabel('Diastolic BP')
plt.title('Blood Pressure Distribution by CVD Status')
plt.legend()

plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================
# STEP 4: CORRELATION ANALYSIS
# ================================

print("\n\n4. CORRELATION ANALYSIS")
print("-" * 40)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Create correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, mask=mask)
plt.title('Correlation Matrix of Heart Disease Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Display correlations with target variable
target_correlations = correlation_matrix['target'].sort_values(key=abs, ascending=False)
print("\nCorrelations with target variable (Cardiovascular Disease):")
print(target_correlations)

# Display strongest positive and negative correlations
print("\nStrongest positive correlations with CVD:")
positive_corr = target_correlations[target_correlations > 0].head(6)
for feature, corr in positive_corr.items():
    if feature != 'target':
        print(f"  {feature}: {corr:.4f}")
        
print("\nStrongest negative correlations with CVD:")
negative_corr = target_correlations[target_correlations < 0].head(5)
for feature, corr in negative_corr.items():
    print(f"  {feature}: {corr:.4f}")

# ================================
# STEP 5: MACHINE LEARNING MODEL IMPLEMENTATION
# ================================

print("\n\n5. MACHINE LEARNING MODEL IMPLEMENTATION")
print("-" * 40)

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

# Train and evaluate models
results = {}
model_objects = {}

print("\nTraining and evaluating models...")
print("="*50)

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * len(name))
    
    # Train the model
    if name in ['K-Nearest Neighbors', 'Support Vector Machine']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'predictions': y_pred
    }
    model_objects[name] = model
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Mean Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))

# ================================
# STEP 6: MODEL COMPARISON AND SELECTION
# ================================

print("\n\n6. MODEL COMPARISON AND FINAL RESULTS")
print("-" * 40)

# Create accuracy comparison
accuracy_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'CV Mean Accuracy': [results[model]['cv_mean'] for model in results.keys()],
    'CV Std': [results[model]['cv_std'] for model in results.keys()]
})

accuracy_df = accuracy_df.sort_values('Test Accuracy', ascending=False)
print("\nModel Performance Comparison:")
print(accuracy_df.to_string(index=False))

# Visualize model performance
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.bar(accuracy_df['Model'], accuracy_df['Test Accuracy'], color='skyblue', alpha=0.8)
plt.title('Model Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(accuracy_df['Test Accuracy']):
    plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.bar(accuracy_df['Model'], accuracy_df['CV Mean Accuracy'], color='lightcoral', alpha=0.8)
plt.title('Model Cross-Validation Accuracy')
plt.ylabel('CV Accuracy')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(accuracy_df['CV Mean Accuracy']):
    plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Select best model
best_model_name = accuracy_df.iloc[0]['Model']
best_model = model_objects[best_model_name]
best_accuracy = accuracy_df.iloc[0]['Test Accuracy']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test Accuracy: {best_accuracy:.4f}")

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Decision Tree']:
    plt.figure(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, y='feature', x='importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTop 5 Important Features:")
    print(feature_importance.head())

# Confusion Matrix for best model
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
plt.show()

# ================================
# STEP 7: SAVE RESULTS AND MODEL
# ================================

print("\n\n7. SAVING RESULTS")
print("-" * 40)

# Save accuracy results to CSV
accuracy_df.to_csv('reports/model_accuracy_comparison.csv', index=False)
print("✓ Model accuracy comparison saved to reports/model_accuracy_comparison.csv")

# Save correlation matrix
correlation_matrix.to_csv('reports/correlation_matrix.csv')
print("✓ Correlation matrix saved to reports/correlation_matrix.csv")

# Save feature importance if available
if best_model_name in ['Random Forest', 'Decision Tree']:
    feature_importance.to_csv('reports/feature_importance.csv', index=False)
    print("✓ Feature importance saved to reports/feature_importance.csv")

# Create summary report
summary_report = f"""
CARDIOVASCULAR DISEASE PREDICTION - SUMMARY REPORT
=================================================

Dataset Information:
- Total samples: {df.shape[0]}
- Total features: {df.shape[1]-1}
- Disease prevalence: {(df['target'].sum() / len(df)) * 100:.1f}%

Model Performance Results:
{accuracy_df.to_string(index=False)}

Best Performing Model: {best_model_name}
- Test Accuracy: {best_accuracy:.4f}
- Cross-Validation Accuracy: {accuracy_df.iloc[0]['CV Mean Accuracy']:.4f} (+/- {accuracy_df.iloc[0]['CV Std']*2:.4f})

Key Insights:
1. The dataset contains {df.shape[0]} samples with {df.shape[1]-1} features
2. {(df['target'].sum() / len(df)) * 100:.1f}% of patients have heart disease
3. {best_model_name} achieved the highest accuracy of {best_accuracy:.4f}
4. All models performed reasonably well, indicating good predictive features

Features with Strongest Correlation to Heart Disease:
{target_correlations.head().to_string()}

Recommendations:
- The {best_model_name} model is recommended for cardiovascular disease prediction
- Further feature engineering could potentially improve performance
- Consider ensemble methods for even better results
"""

with open('reports/summary_report.txt', 'w') as f:
    f.write(summary_report)

print("✓ Summary report saved to reports/summary_report.txt")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("="*60)
print(f"Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")
print("All results, visualizations, and reports have been saved.")
print("="*60)