import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib  # For saving the model
from itertools import combinations
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')

# Load the dataset
data = pd.read_csv('audio/merged_output.csv')

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Filter to include all rows where 'name' column starts with 'audio' (case-insensitive)
data = data[data['name'].str.lower().str.startswith('audio')]

# Define the relevant features
possible_features = [
    'nsyll', 'npause', 'dur(s)', 'phonationtime(s)',
    'speechrate(nsyll/dur)', 'articulation_rate(nsyll/phonationtime)',
    'ASD(speakingtime/nsyll)', 'nrFP', 'tFP(s)',
    'Mean Pitch (Hz)', 'Min Pitch (Hz)', 'Max Pitch (Hz)',
    'F1 (Hz)', 'F2 (Hz)', 'Mean Intensity (dB)', 'Duration (s)'
]

# Drop rows with any missing values in relevant columns
relevant_columns = ['Fluency'] + possible_features
data = data.dropna(subset=relevant_columns)

# Drop classes with fewer than 2 samples
data = data.groupby('Fluency').filter(lambda x: len(x) >= 2)

# Map the fluency levels to 3 targets: Beginner, Intermediate, Advanced
fluency_map = {
    'A1': 'Beginner', 'A2': 'Beginner',
    'B1': 'Intermediate', 'B2': 'Intermediate',
    'C1': 'Advanced', 'C2': 'Advanced'
}
data['Fluency_mapped'] = data['Fluency'].map(fluency_map)

# Encode the new fluency labels
label_encoder = LabelEncoder()
data['Fluency_encoded'] = label_encoder.fit_transform(data['Fluency_mapped'])

# Define a custom sampling strategy to oversample specific classes
sampling_strategy = {
    0: 212,  # Beginner (A1, A2, combined)
    1: 212,  # Intermediate (B1, B2, combined)
    2: 212   # Advanced (C1, C2, combined)
}

# Generate all combinations of seven features
feature_combinations = list(combinations(possible_features, 7))

# Initialize variables to store the best model
best_f1_score = 0
best_model = None
best_features = None
best_metrics = {}
results = []

# Loop over each combination of features
for idx, features in enumerate(feature_combinations):
    print(f"Processing combination {idx+1}/{len(feature_combinations)}: {features}")
    try:
        # Select the features and target variable
        X = data[list(features)]
        y = data['Fluency_encoded']

        # Split the dataset with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # **Print the size of the test set for each category**
        test_counts = y_test.value_counts()
        print("Test set size per category:")
        for class_index in sorted(test_counts.index):
            class_name = label_encoder.inverse_transform([class_index])[0]
            print(f"  {class_name}: {test_counts[class_index]}")

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Initialize the RandomForestClassifier with class weighting
        model = RandomForestClassifier(random_state=42, class_weight='balanced')

        # Train the model
        model.fit(X_train_resampled, y_train_resampled)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Evaluate the model with F1 Score and Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Cross-Validation Scores using StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cross_val_scores = cross_val_score(
            model, X_train_resampled, y_train_resampled, cv=skf, scoring='accuracy'
        )
        mean_cv_accuracy = np.mean(cross_val_scores)

        # ROC-AUC (for multi-class classification)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

        # Log Loss
        log_loss_score = log_loss(y_test, y_prob)

        # Cohen's Kappa
        cohen_kappa = cohen_kappa_score(y_test, y_pred)

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_test, y_pred)

        # Record the results
        result = {
            'features': features,
            'accuracy': accuracy,
            'f1_score': f1,
            'mean_cv_accuracy': mean_cv_accuracy,
            'roc_auc': roc_auc,
            'log_loss': log_loss_score,
            'cohen_kappa': cohen_kappa,
            'mcc': mcc
        }

        # Append to the results list
        results.append(result)

        # Save the results to CSV after each iteration
        df_results = pd.DataFrame(results)
        df_results.to_csv('model_results.csv', index=False)

        # **Print the results for this model**
        print(f"Results for features {features}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean CV Accuracy: {mean_cv_accuracy:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Log Loss: {log_loss_score:.4f}")
        print(f"Cohen's Kappa: {cohen_kappa:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}\n")

        # Update the best model if current F1 score is higher
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model
            best_features = features
            best_metrics = result
            # Save the best model
            joblib.dump(best_model, 'best_model.pkl')

            # Generate and save the confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig('best_confusion_matrix.png')
            plt.close()

            # Feature Importances
            feature_importances = pd.Series(best_model.feature_importances_, index=best_features)
            feature_importances = feature_importances.sort_values(ascending=False)

            # Plot feature importances
            plt.figure(figsize=(8, 6))
            feature_importances.plot(kind='bar')
            plt.title("Feature Importances")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig('best_feature_importances.png')
            plt.close()

            # Learning Curves
            train_sizes, train_scores, test_scores = learning_curve(
                best_model, X_train_resampled, y_train_resampled, cv=skf, n_jobs=-1
            )
            plt.figure(figsize=(8, 6))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training score")
            plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validation score")
            plt.title("Learning Curve")
            plt.xlabel("Training Size")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig('best_learning_curve.png')
            plt.close()

            # Save the train and test datasets
            train_data = X_train.copy()
            train_data['Fluency_encoded'] = y_train
            train_data.to_csv('best_train.csv', index=False)

            test_data = X_test.copy()
            test_data['Fluency_encoded'] = y_test
            test_data.to_csv('best_test.csv', index=False)

    except Exception as e:
        print(f"An error occurred with features {features}: {e}\n")
        continue

# After all combinations are processed, print out the best model details
print("\nBest Model Found:")
print(f"Features: {best_features}")
print(f"Accuracy: {best_metrics['accuracy']:.4f}")
print(f"F1 Score: {best_metrics['f1_score']:.4f}")
print(f"Mean CV Accuracy: {best_metrics['mean_cv_accuracy']:.4f}")
print(f"ROC-AUC Score: {best_metrics['roc_auc']:.4f}")
print(f"Log Loss: {best_metrics['log_loss']:.4f}")
print(f"Cohen's Kappa: {best_metrics['cohen_kappa']:.4f}")
print(f"Matthews Correlation Coefficient: {best_metrics['mcc']:.4f}")

# Print Classification Report for the best model
y_pred_best = best_model.predict(X_test)
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# =========================================
# Rebalance the test set to 10 samples per class
# =========================================

# Function to balance the test set
def balance_test_set(X, y, label_encoder, target_count=10):
    """
    Balances the test set to have an equal number of samples per class.

    Parameters:
    - X: Feature DataFrame
    - y: Target labels
    - label_encoder: Label encoder used for encoding the target labels
    - target_count: Number of samples per class in the test set

    Returns:
    - Balanced X_test and y_test
    """
    X_test_balanced = []
    y_test_balanced = []

    # Combine X and y to shuffle and split by class
    combined = pd.concat([X, y], axis=1)
    class_column = y.name

    # Iterate through each class and downsample or upsample
    for class_idx in range(len(label_encoder.classes_)):
        class_data = combined[combined[class_column] == class_idx]
        if len(class_data) >= target_count:
            # Downsample
            class_sample = class_data.sample(n=target_count, random_state=42)
        else:
            # Upsample
            class_sample = resample(
                class_data, replace=True, n_samples=target_count, random_state=42
            )
        X_test_balanced.append(class_sample.drop(columns=[class_column]))
        y_test_balanced.append(class_sample[class_column])

    # Concatenate back into DataFrame
    X_test_balanced = pd.concat(X_test_balanced).reset_index(drop=True)
    y_test_balanced = pd.concat(y_test_balanced).reset_index(drop=True)

    return X_test_balanced, y_test_balanced

# Ensure the test set is balanced
X_test_balanced, y_test_balanced = balance_test_set(X_test, y_test, label_encoder, target_count=10)

# Select the best features from the previously found model
best_features_list = list(best_features)
X_train_selected = X_train_resampled[best_features_list]
X_test_selected = X_test_balanced[best_features_list]

# Train the model using the best features
best_model_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
best_model_balanced.fit(X_train_selected, y_train_resampled)

# Predictions and evaluation on the balanced test set
y_pred_balanced = best_model_balanced.predict(X_test_selected)
y_prob_balanced = best_model_balanced.predict_proba(X_test_selected)

# Compute metrics
balanced_metrics = {
    'accuracy': accuracy_score(y_test_balanced, y_pred_balanced),
    'f1_score': f1_score(y_test_balanced, y_pred_balanced, average='weighted'),
    'roc_auc': roc_auc_score(y_test_balanced, y_prob_balanced, multi_class='ovr', average='weighted'),
    'log_loss': log_loss(y_test_balanced, y_prob_balanced),
    'cohen_kappa': cohen_kappa_score(y_test_balanced, y_pred_balanced),
    'mcc': matthews_corrcoef(y_test_balanced, y_pred_balanced)
}

# Print results
print("\nBalanced Test Set Metrics:")
for metric, value in balanced_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# Classification report for the balanced test set
print("\nClassification Report for Balanced Test Set:")
print(classification_report(y_test_balanced, y_pred_balanced, target_names=label_encoder.classes_))

# Confusion matrix for the balanced test set
conf_matrix_balanced = confusion_matrix(y_test_balanced, y_pred_balanced)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_balanced, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix (Balanced Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig('balanced_confusion_matrix.png')
plt.close()

# Feature Importances
feature_importances_balanced = pd.Series(best_model_balanced.feature_importances_, index=best_features_list)
feature_importances_balanced = feature_importances_balanced.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 6))
feature_importances_balanced.plot(kind='bar')
plt.title("Feature Importances (Balanced Test Set)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig('balanced_feature_importances.png')
plt.close()

# Save the balanced test data and metrics for later use
balanced_test_data = X_test_balanced.copy()
balanced_test_data['True_Label'] = y_test_balanced
balanced_test_data['Predicted_Label'] = y_pred_balanced
balanced_test_data.to_csv('balanced_test_data.csv', index=False)

# Save metrics to a file
balanced_metrics_df = pd.DataFrame([balanced_metrics])
balanced_metrics_df.to_csv('balanced_metrics.csv', index=False)

# Save the best model for future use
joblib.dump(best_model_balanced, 'best_model_balanced.pkl')

# Display the test set size per category after balancing
print("\nBalanced Test Set Size per Category:")
for class_index in range(len(label_encoder.classes_)):
    class_name = label_encoder.inverse_transform([class_index])[0]
    print(f"  {class_name}: {sum(y_test_balanced == class_index)}")
