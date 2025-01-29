import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("audio/merged_output.csv")

# Encode the Fluency column as numerical values
fluency_mapping = {'A1': 1, 'A2': 1, 'B1': 2, 'B2': 2, 'C1': 3, 'C2': 3}
data['Fluency_encoded'] = data['Fluency'].map(fluency_mapping)
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Select numerical features and target
numerical_features = [
    'nsyll', 'npause', 'dur(s)', 'phonationtime(s)', 'speechrate(nsyll/dur)',
    'articulation_rate(nsyll/phonationtime)', 'ASD(speakingtime/nsyll)', 'nrFP',
    'tFP(s)', 'Mean Pitch (Hz)', 'Min Pitch (Hz)', 'Max Pitch (Hz)', 'F1 (Hz)',
    'F2 (Hz)', 'Mean Intensity (dB)', 'Duration (s)'
]
X = data[numerical_features]
y = data['Fluency_encoded']

# Handle missing values (if any)
X = X.fillna(0)

# ---------------- Feature Engineering ---------------- #
# Interaction Features
data['speechrate_x_articulation'] = data['speechrate(nsyll/dur)'] * data['articulation_rate(nsyll/phonationtime)']

# Log Transformation
for feature in ['dur(s)', 'phonationtime(s)', 'tFP(s)']:
    data[f'log_{feature}'] = np.log1p(data[feature])

# Ratios
data['pause_to_duration_ratio'] = data['npause'] / data['dur(s)']
data['phonation_to_duration_ratio'] = data['phonationtime(s)'] / data['dur(s)']

# Custom Fluency Index
data['fluency_index'] = data['nsyll'] / (data['npause'] + 1)

# Update numerical features with new engineered features
engineered_features = [
    'speechrate_x_articulation', 'log_dur(s)', 'log_phonationtime(s)', 'log_tFP(s)',
    'pause_to_duration_ratio', 'phonation_to_duration_ratio', 'fluency_index'
]
numerical_features += engineered_features
X = data[numerical_features]

# ---------------- Analyze Class Overlap ---------------- #
# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Fluency Encoded')
plt.title("t-SNE Visualization of Class Overlap")
plt.show()

# Pair Plots
sns.pairplot(data, vars=['speechrate(nsyll/dur)', 'articulation_rate(nsyll/phonationtime)', 'dur(s)'], hue='Fluency_encoded')
plt.show()

# Correlation Heatmap
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation with Fluency Encoded")
plt.show()

# Distribution of Key Features Across Classes
for feature in ['speechrate(nsyll/dur)', 'articulation_rate(nsyll/phonationtime)']:
    sns.boxplot(x='Fluency_encoded', y=feature, data=data)
    plt.title(f"Distribution of {feature} Across Classes")
    plt.show()

# ---------------- Model Training ---------------- #
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features (on both training and test sets)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Perform PCA on standardized training and test sets
n_components = 6  # Based on earlier analysis
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Explained variance ratio (cumulative): {pca.explained_variance_ratio_.cumsum()}")

# Train the Random Forest classifier
clf = RandomForestClassifier(random_state=42, class_weight="balanced")
clf.fit(X_train_pca, y_train_resampled)

# Evaluate the model
y_pred = clf.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Cross-validation to check generalization
from sklearn.model_selection import cross_val_score
cross_val_scores = cross_val_score(clf, X_train_pca, y_train_resampled, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy (mean ± std): {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")
