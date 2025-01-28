import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE  # To handle class imbalance
import warnings

# Suppress warnings to avoid cluttering the output
warnings.filterwarnings("ignore")  # From ChatGPT

# Load the dataset
df = pd.read_csv('Dataset.csv')

print(df.head(51))  # Display the first 51 rows of the dataframe

df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None') 

# Encode Sleep Disorder into binary format
df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: 1 if x != 'None' else 0)

# Select features and target variable for the model
variables = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 
            'Physical Activity Level', 'Stress Level', 'BMI Category', 
            'Blood Pressure', 'Heart Rate', 'Daily Steps']
X = df[variables]
y = df['Sleep Disorder']

# Encode categorical variables into dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if len(y_train.unique()) > 1:  # Ensure there are more than 1 classes
    smote = SMOTE(random_state=42)  # ChatGPT
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)  

# Logistic Regression
logistic = LogisticRegression(max_iter=1000, random_state=42) 
logistic.fit(X_train_resampled, y_train_resampled) 

# Predictions
pred_logistic = logistic.predict(X_test)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=42) 
decision_tree.fit(X_train_resampled, y_train_resampled)  

# Predictions
pred_tree = decision_tree.predict(X_test)

# Report for Logistic Regression
logistic_report = classification_report(y_test, pred_logistic, output_dict=True) 
logistic_accuracy = (pred_logistic == y_test).mean()  # Accuracy

# Report for Decision Tree
tree_report = classification_report(y_test, pred_tree, output_dict=True)
tree_accuracy = (pred_tree == y_test).mean()  # Accuracy

# Summary report of the algorithms
print(f"{'Model':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
print(f"{'Logistic Regression':<30}{logistic_accuracy:.2f}{logistic_report['1']['precision']:.2f}{logistic_report['1']['recall']:.2f}{logistic_report['1']['f1-score']:.2f}")
print(f"{'Decision Tree Classifier':<30} {tree_accuracy:.2f}{tree_report['1']['precision']:.2f}{tree_report['1']['recall']:.2f}{tree_report['1']['f1-score']:.2f}")

# Confusion Matrix for Logistic Regression
matrix_logistic = confusion_matrix(y_test, pred_logistic) 
plt.figure(figsize=(8, 6))  
sns.heatmap(matrix_logistic, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Sleep Disorder', 'Sleep Disorder'], 
            yticklabels=['No Sleep Disorder', 'Sleep Disorder']) 
plt.title('Confusion Matrix for Logistic Regression') 
plt.ylabel('Actual') 
plt.show() 

# ROC Curve for Logistic Regression
fpr, tpr, thresholds = roc_curve(y_test, logistic.predict_proba(X_test)[:, 1])  # False positive rate and true positive rate
roc_auc = auc(fpr, tpr)  # Calculate area under the ROC curve

plt.figure(figsize=(8, 6)) 
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))  # Plot ROC curve
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Plot diagonal line
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05]) 
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate') 
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')  
plt.legend(loc='lower right')  
plt.show() 

# Confusion Matrix for Decision Tree Classifier
matrix_tree = confusion_matrix(y_test, pred_tree)  
plt.figure(figsize=(8, 6))  
sns.heatmap(matrix_tree, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Sleep Disorder', 'Sleep Disorder'], 
            yticklabels=['No Sleep Disorder', 'Sleep Disorder'])  
plt.title('Confusion Matrix for Decision Tree Classifier') 
plt.xlabel('Predicted')  
plt.ylabel('Actual') 
plt.show()  

# Node information
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, 
          filled=True, 
          feature_names=X.columns,  
          class_names=['No Sleep Disorder', 'Sleep Disorder'], 
          rounded=True,  
          fontsize=12,  
          impurity=False,  
          node_ids=True, 
          proportion=True) 

for i, node in enumerate(decision_tree.tree_.children_left): 
    if node != -1:  
        n_samples = decision_tree.tree_.n_node_samples[i] 
        class_distribution = decision_tree.tree_.value[i]  
        predicted_class = np.argmax(class_distribution)  

        plt.text(decision_tree.tree_.threshold[i], 
                 i, 
                 f'Samples: {n_samples}\nClass: {predicted_class}\n{class_distribution}', 
                 fontsize=10, 
                 ha='center', 
                 va='center', 
                 color='black')

plt.title("Decision Tree for Sleep Disorder Classification", fontsize=16)  
plt.xlabel("Features", fontsize=14) 
plt.ylabel("Samples", fontsize=14)
plt.show()  # Display