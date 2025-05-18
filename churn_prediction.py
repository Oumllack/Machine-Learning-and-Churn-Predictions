import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

class ChurnPredictor:
    def __init__(self, data_path):
        """Initialize the churn predictor."""
        self.data_path = data_path
        self.data = None
        self.original_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'xgboost': XGBClassifier(random_state=42, n_estimators=100)
        }
        self.predictions = {}
        self.metrics = {}
        
    def load_data(self):
        """Load and prepare the data."""
        try:
            self.data = pd.read_csv(self.data_path)
            self.original_data = self.data.copy()
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def analyze_data(self):
        """Perform comprehensive data analysis."""
        if self.data is None:
            print("Please load data first.")
            return

        # Create analysis directory
        import os
        os.makedirs('analysis', exist_ok=True)

        # 1. Basic Statistics
        stats = self.original_data.describe()
        stats.to_csv('analysis/basic_statistics.csv')

        # 2. Churn Distribution
        plt.figure(figsize=(10, 6))
        churn_counts = self.original_data['Churn'].value_counts()
        plt.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        plt.title('Customer Churn Distribution')
        plt.savefig('analysis/churn_distribution.png')
        plt.close()

        # 3. Correlation Analysis
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
        correlation = self.original_data[numeric_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('analysis/correlation_matrix.png')
        plt.close()

        # 4. Feature Analysis
        categorical_cols = self.original_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'customerID':
                plt.figure(figsize=(12, 6))
                sns.countplot(data=self.original_data, x=col, hue='Churn')
                plt.title(f'Churn Distribution by {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'analysis/churn_by_{col}.png')
                plt.close()

        # 5. Numerical Features Analysis
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.original_data, x='Churn', y=col)
            plt.title(f'{col} Distribution by Churn Status')
            plt.savefig(f'analysis/{col}_distribution.png')
            plt.close()

        print("Data analysis completed. Results saved in 'analysis' directory.")

    def preprocess_data(self):
        """Preprocess the data for modeling."""
        if self.data is None:
            print("Please load data first.")
            return False

        # Save customer IDs
        customer_ids = self.data['customerID'].copy() if 'customerID' in self.data.columns else None

        # Remove non-relevant columns
        if 'customerID' in self.data.columns:
            self.data = self.data.drop('customerID', axis=1)

        # Convert categorical variables
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            self.data[col] = le.fit_transform(self.data[col])

        # Save feature names
        self.feature_names = self.data.drop('Churn', axis=1).columns

        # Split features and target
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']

        # Normalize numerical features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        print("Data preprocessing completed successfully.")
        return True

    def train_models(self):
        """Train models and evaluate their performance."""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name} model...")
            
            # Training
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Store predictions
            self.predictions[name] = {
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Calculate metrics
            results[name] = {
                'classification_report': classification_report(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            # Plot ROC curve
            plt.figure(figsize=(10, 6))
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {results[name]["roc_auc"]})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.savefig(f'analysis/roc_curve_{name}.png')
            plt.close()

            # Plot Precision-Recall curve
            plt.figure(figsize=(10, 6))
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            plt.plot(recall, precision, label=f'PR curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {name}')
            plt.legend()
            plt.savefig(f'analysis/pr_curve_{name}.png')
            plt.close()

            # Plot learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10))
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, label='Training score')
            plt.plot(train_sizes, test_mean, label='Cross-validation score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
            plt.xlabel('Training Examples')
            plt.ylabel('Score')
            plt.title(f'Learning Curve - {name}')
            plt.legend(loc='best')
            plt.savefig(f'analysis/learning_curve_{name}.png')
            plt.close()
            
            print(f"\nClassification Report for {name}:")
            print(results[name]['classification_report'])
            print(f"ROC AUC Score: {results[name]['roc_auc']:.3f}")
        
        self.metrics = results
        return results

    def generate_predictions_file(self, model_name='random_forest'):
        """Generate a CSV file with predictions for each customer."""
        if model_name not in self.predictions:
            print(f"Model {model_name} not found in predictions.")
            return

        # Get test set customer IDs
        test_indices = self.original_data.index[-len(self.y_test):]
        customer_ids = self.original_data.loc[test_indices, 'customerID']

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'customerID': customer_ids,
            'Churn_Predicted': self.predictions[model_name]['predictions'],
            'Churn_Probability': self.predictions[model_name]['probabilities']
        })

        # Add important customer information
        important_features = ['tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 
                            'PaymentMethod', 'InternetService', 'TechSupport']
        for feature in important_features:
            if feature in self.original_data.columns:
                predictions_df[feature] = self.original_data.loc[test_indices, feature]

        # Save predictions
        output_file = f'analysis/predictions_{model_name}.csv'
        predictions_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")

        # Identify high-risk customers
        high_risk_customers = predictions_df[
            (predictions_df['Churn_Probability'] > 0.7) & 
            (predictions_df['Churn_Predicted'] == 1)
        ]
        print(f"\nNumber of high-risk customers identified: {len(high_risk_customers)}")
        
        # Analyze high-risk customers
        self.analyze_high_risk_customers(high_risk_customers)
        
        return predictions_df

    def analyze_high_risk_customers(self, high_risk_df):
        """Analyze characteristics of high-risk customers."""
        # 1. Contract Type Distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(data=high_risk_df, x='Contract')
        plt.title('Contract Type Distribution - High Risk Customers')
        plt.xticks(rotation=45)
        plt.savefig('analysis/high_risk_contract_distribution.png')
        plt.close()

        # 2. Monthly Charges Analysis
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=high_risk_df, x='Contract', y='MonthlyCharges')
        plt.title('Monthly Charges by Contract Type - High Risk Customers')
        plt.xticks(rotation=45)
        plt.savefig('analysis/high_risk_monthly_charges.png')
        plt.close()

        # 3. Tenure Analysis
        plt.figure(figsize=(12, 6))
        sns.histplot(data=high_risk_df, x='tenure', bins=30)
        plt.title('Tenure Distribution - High Risk Customers')
        plt.savefig('analysis/high_risk_tenure_distribution.png')
        plt.close()

        # 4. Payment Method Analysis
        plt.figure(figsize=(12, 6))
        sns.countplot(data=high_risk_df, x='PaymentMethod')
        plt.title('Payment Method Distribution - High Risk Customers')
        plt.xticks(rotation=45)
        plt.savefig('analysis/high_risk_payment_method.png')
        plt.close()

        # Save high-risk customer analysis
        high_risk_df.to_csv('analysis/high_risk_customers_analysis.csv', index=False)
        print("\nHigh-risk customer analysis saved in 'analysis' directory")

    def plot_feature_importance(self, model_name='random_forest'):
        """Visualize feature importance for the specified model."""
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return

        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.tight_layout()
            plt.savefig(f'analysis/feature_importance_{model_name}.png')
            plt.close()
            print(f"\nFeature importance plot saved to analysis/feature_importance_{model_name}.png")
        else:
            print(f"Model {model_name} does not provide feature importances.")

    def generate_summary_report(self):
        """Generate a comprehensive summary report of the analysis."""
        report = []
        report.append("# Customer Churn Analysis Report\n")
        
        # 1. Dataset Overview
        report.append("## 1. Dataset Overview")
        report.append(f"- Total number of customers: {len(self.original_data)}")
        report.append(f"- Number of features: {len(self.feature_names)}")
        report.append(f"- Churn rate: {(self.original_data['Churn'].mean() * 100):.1f}%\n")
        
        # 2. Model Performance
        report.append("## 2. Model Performance")
        for name, metrics in self.metrics.items():
            report.append(f"\n### {name.upper()}")
            report.append(f"- ROC AUC Score: {metrics['roc_auc']:.3f}")
            report.append("\nClassification Report:")
            report.append("```")
            report.append(metrics['classification_report'])
            report.append("```\n")
        
        # 3. Key Findings
        report.append("## 3. Key Findings")
        report.append("\n### Customer Characteristics")
        report.append("- Most important factors influencing churn:")
        if hasattr(self.models['random_forest'], 'feature_importances_'):
            importances = self.models['random_forest'].feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [self.feature_names[i] for i in indices[:5]]
            report.append("  - " + "\n  - ".join(top_features))
        
        # 4. Recommendations
        report.append("\n## 4. Recommendations")
        report.append("1. Focus on customers with month-to-month contracts")
        report.append("2. Review pricing strategy for high monthly charges")
        report.append("3. Implement retention programs for customers with high churn probability")
        report.append("4. Monitor customer service quality, especially for customers without tech support")
        
        # Save report
        with open('analysis/summary_report.md', 'w') as f:
            f.write('\n'.join(report))
        print("\nSummary report generated: analysis/summary_report.md")

def main():
    # Initialize
    predictor = ChurnPredictor('data/Telco-Customer-Churn.csv')
    
    # Load and analyze data
    if predictor.load_data():
        predictor.analyze_data()
        
        # Preprocess and model
        if predictor.preprocess_data():
            # Train and evaluate models
            results = predictor.train_models()
            
            # Generate predictions for Random Forest
            predictions_df = predictor.generate_predictions_file('random_forest')
            
            # Plot feature importance
            predictor.plot_feature_importance('random_forest')
            
            # Generate summary report
            predictor.generate_summary_report()

if __name__ == "__main__":
    main() 