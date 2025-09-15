# Cardiovascular Disease Prediction Project

## Project Overview
This project implements a comprehensive cardiovascular disease prediction system using machine learning algorithms. The analysis includes data preprocessing, exploratory data analysis, feature engineering, and comparison of multiple ML algorithms to identify the best performing model.

## Dataset Information
- **Source**: `cardio_train.csv` (70,000 samples)
- **Target**: Binary classification (0 = No CVD, 1 = CVD Present)
- **Features**: 12 original features + 2 engineered features
- **Disease Prevalence**: 50.0% (perfectly balanced dataset)

### Feature Descriptions
| Feature | Description |
|---------|-------------|
| age | Age in years (converted from days) |
| gender | Gender (1=female, 2=male) |
| height | Height in cm |
| weight | Weight in kg |
| ap_hi | Systolic blood pressure |
| ap_lo | Diastolic blood pressure |
| cholesterol | Cholesterol level (1=normal, 2=above normal, 3=well above normal) |
| gluc | Glucose level (1=normal, 2=above normal, 3=well above normal) |
| smoke | Smoking status (0=no, 1=yes) |
| alco | Alcohol intake (0=no, 1=yes) |
| active | Physical activity (0=no, 1=yes) |
| bmi | Body Mass Index (calculated feature) |
| bp_category | Blood pressure category (0=normal, 1=high) |

## Project Structure
```
C:\AI Projects\Major Project\
├── cardiovascular_disease_prediction.py  # Main analysis script
├── requirements.txt                       # Dependencies
├── README.md                             # Project documentation
├── data/
│   └── cardio_train.csv                  # Dataset
├── visualizations/
│   ├── comprehensive_analysis.png        # Main EDA visualizations
│   ├── feature_distributions.png         # Feature analysis plots
│   ├── correlation_matrix.png            # Correlation heatmap
│   ├── model_comparison.png              # Model performance comparison
│   └── confusion_matrix_best_model.png   # Best model confusion matrix
├── reports/
│   ├── summary_report.txt                # Comprehensive project summary
│   ├── model_accuracy_comparison.csv     # Model performance metrics
│   └── correlation_matrix.csv            # Feature correlations
└── models/                               # Directory for saved models
```

## Key Findings

### Data Analysis Insights
1. **Age and Blood Pressure**: Strong positive correlations with CVD (0.238 each)
2. **Cholesterol**: Significant predictor with 0.221 correlation
3. **BMI and Weight**: Moderate positive correlations (0.166 and 0.182)
4. **Lifestyle Factors**: Physical activity shows protective effect (-0.036 correlation)
5. **Gender**: Minimal impact on CVD risk (0.008 correlation)

### Model Performance Results
| Model | Test Accuracy | CV Mean Accuracy | CV Std |
|-------|---------------|------------------|--------|
| **Support Vector Machine** | **72.34%** | **72.56%** | **±0.72%** |
| Random Forest | 71.17% | 71.64% | ±0.84% |
| Logistic Regression | 71.16% | 71.54% | ±1.28% |
| K-Nearest Neighbors | 64.30% | 64.34% | ±1.09% |
| Decision Tree | 62.81% | 63.40% | ±0.79% |

## Data Preprocessing Operations Performed
1. ✅ **Data Loading**: Successfully loaded 70,000 samples with 13 features
2. ✅ **Missing Value Handling**: No missing values found in the dataset
3. ✅ **Feature Engineering**:
   - Converted age from days to years
   - Created BMI feature from height and weight
   - Engineered blood pressure category feature
4. ✅ **Data Cleaning**: Removed non-predictive ID column
5. ✅ **Feature Scaling**: Applied StandardScaler for SVM and KNN models

## Comprehensive Visualizations Created
1. **Target Distribution**: Pie chart showing CVD prevalence
2. **Age Analysis**: Age distribution by CVD status
3. **Gender Analysis**: CVD distribution by gender
4. **BMI Analysis**: Box plots showing BMI differences
5. **Blood Pressure Analysis**: Systolic BP by CVD status
6. **Cholesterol Analysis**: CVD risk by cholesterol levels
7. **Lifestyle Factors**: Analysis of smoking, alcohol, and physical activity
8. **Correlation Heatmap**: Complete feature correlation matrix
9. **Model Comparison**: Performance visualization across all algorithms
10. **Confusion Matrix**: Best model classification results

## Machine Learning Implementation
- **Train/Test Split**: 70/30 split with stratification
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Feature Scaling**: Applied for distance-based algorithms
- **Model Selection**: Based on test accuracy and CV consistency

## Best Model: Support Vector Machine
- **Test Accuracy**: 72.34%
- **Cross-Validation Accuracy**: 72.56% (±0.72%)
- **Precision (No CVD)**: 71%
- **Precision (CVD Present)**: 74%
- **Recall (No CVD)**: 75%
- **Recall (CVD Present)**: 69%

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python cardiovascular_disease_prediction.py
```

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

## Results and Outputs
The script generates:
- **5 visualization files** in the `visualizations/` folder
- **3 report files** in the `reports/` folder
- **Console output** with detailed analysis steps and results

## Recommendations
1. **Model Selection**: Use Support Vector Machine for CVD prediction
2. **Key Risk Factors**: Focus on age, blood pressure, and cholesterol monitoring
3. **Feature Engineering**: The engineered BMI and BP category features provide additional predictive value
4. **Further Improvements**: 
   - Consider ensemble methods combining multiple algorithms
   - Explore advanced feature engineering techniques
   - Implement hyperparameter tuning for better performance

## Clinical Relevance
This model achieves 72.34% accuracy in predicting cardiovascular disease, which is clinically significant for:
- **Risk Assessment**: Early identification of high-risk patients
- **Preventive Care**: Targeted interventions for modifiable risk factors
- **Resource Allocation**: Efficient healthcare resource utilization
- **Population Health**: Large-scale screening programs

## Author
AI Assistant - Major Project in Cardiovascular Disease Prediction
Date: September 2025

---
*This project successfully implements all required components: data preprocessing, comprehensive visualizations, correlation analysis, multiple ML algorithms comparison, and final model selection with detailed reporting.*