#!/usr/bin/env python
"""
Train a BDT classifier for CC vs NC events using all 6 simulation files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle

# Start timing
start_time = time.time()

# Directory containing your files
data_dir = "output/"
# Create a results directory for saving figures
results_dir = os.path.join(data_dir, "bdt_results")
os.makedirs(results_dir, exist_ok=True)

# List of simulation output files
sim_files = [
    "10000_events_simset_1600_seed_1000_20250516_112519.parquet",
    "10000_events_simset_1601_seed_1001_20250516_112659.parquet",
    "10000_events_simset_1602_seed_1002_20250516_112840.parquet",
    "10000_events_simset_1603_seed_1003_20250516_112857.parquet",
    "10000_events_simset_1604_seed_1004_20250516_112923.parquet",
    "10000_events_simset_1605_seed_1005_20250516_112950.parquet"
]

# Configure output
log_file = os.path.join(results_dir, "bdt_log.txt")

# Function to log messages to both console and file
def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Initialize log file
with open(log_file, "w") as f:
    f.write("CC vs NC BDT Classification Log\n")
    f.write("=" * 50 + "\n\n")

# Function to extract event-level features
def extract_event_features(event_data):
    try:
        # Get interaction type (1 for CC, 2 for NC)
        interaction_type = event_data['mc_truth']['interaction']
        
        # Label: 1 for CC, 0 for NC
        label = 1 if interaction_type == 1 else 0
        
        # Get photon data
        photons = event_data['photons']
        
        # Skip events with no hits
        if len(photons['t']) == 0:
            return None
        
        # Basic features
        total_hits = len(photons['t'])
        string_ids = photons['string_id']
        sensor_ids = photons['sensor_id']
        unique_doms = len(set(zip(string_ids, sensor_ids)))
        
        # Spatial features
        pos_x = photons['sensor_pos_x']
        pos_y = photons['sensor_pos_y']
        pos_z = photons['sensor_pos_z']
        
        # Calculate maximum distance between any two hit DOMs
        positions = np.vstack((pos_x, pos_y, pos_z)).T
        max_distance = 0
        if len(positions) > 1:
            from scipy.spatial.distance import pdist
            max_distance = np.max(pdist(positions))
        
        # Calculate spatial extent
        x_span = np.max(pos_x) - np.min(pos_x)
        y_span = np.max(pos_y) - np.min(pos_y)
        z_span = np.max(pos_z) - np.min(pos_z)
        
        # Calculate standard deviations
        x_std = np.std(pos_x)
        y_std = np.std(pos_y)
        z_std = np.std(pos_z)
        
        # Timing features
        hit_times = photons['t']
        time_span = np.max(hit_times) - np.min(hit_times)
        time_mean = np.mean(hit_times)
        time_std = np.std(hit_times)
        
        # Shape features
        center_x = np.mean(pos_x)
        center_y = np.mean(pos_y)
        center_z = np.mean(pos_z)
        r_squared = (pos_x - center_x)**2 + (pos_y - center_y)**2 + (pos_z - center_z)**2
        r_mean = np.mean(np.sqrt(r_squared))
        
        # Topology features
        unique_strings = len(np.unique(string_ids))
        hits_per_string = total_hits / max(1, unique_strings)
        
        # Feature set
        features = {
            'total_hits': total_hits,
            'unique_doms': unique_doms,
            'max_distance': max_distance,
            'hits_per_dom': total_hits / max(1, unique_doms),
            'x_span': x_span,
            'y_span': y_span,
            'z_span': z_span,
            'x_std': x_std,
            'y_std': y_std,
            'z_std': z_std,
            'time_span': time_span,
            'time_mean': time_mean,
            'time_std': time_std,
            'r_mean': r_mean,
            'unique_strings': unique_strings,
            'hits_per_string': hits_per_string,
            'label': label
        }
        
        return features
    except Exception as e:
        log_message(f"Error extracting features: {e}")
        return None

# List to collect features from all files
all_features = []

# Process each file
for sim_file in sim_files:
    file_path = os.path.join(data_dir, sim_file)
    if not os.path.exists(file_path):
        log_message(f"File not found: {file_path}")
        continue
    
    log_message(f"\nProcessing {sim_file}...")
    
    # Load the file
    df = pd.read_parquet(file_path)
    log_message(f"Loaded {len(df)} events")
    
    # Get CC and NC events
    cc_count = 0
    nc_count = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        if row['mc_truth']['interaction'] == 1:
            cc_count += 1
        else:
            nc_count += 1
    
    log_message(f"File contains {cc_count} CC events and {nc_count} NC events")
    
    # Extract features from all events
    file_features = []
    for i, (idx, event_data) in enumerate(df.iterrows()):
        if i % 1000 == 0:  # Progress update
            log_message(f"Processing event {i}/{len(df)}...")
        
        features = extract_event_features(event_data)
        if features is not None:
            # Add a source file identifier
            features['file_source'] = sim_file
            file_features.append(features)
    
    log_message(f"Extracted features for {len(file_features)} events from {sim_file}")
    all_features.extend(file_features)

# Convert to DataFrame
features_df = pd.DataFrame(all_features)
log_message(f"\nCombined dataset has {len(features_df)} events")

# Save feature data
feature_file = os.path.join(results_dir, "event_features.csv")
features_df.to_csv(feature_file, index=False)
log_message(f"Saved feature data to {feature_file}")

# Check class balance
cc_count = sum(features_df['label'] == 1)
nc_count = sum(features_df['label'] == 0)
log_message(f"Class balance: {cc_count} CC events ({cc_count/len(features_df)*100:.1f}%), {nc_count} NC events ({nc_count/len(features_df)*100:.1f}%)")

# Check distribution by source file
log_message("\nEvents by source file:")
source_counts = features_df['file_source'].value_counts()
for source, count in source_counts.items():
    log_message(f"  {source}: {count} events")

# Remove file_source column for modeling
X = features_df.drop(['label', 'file_source'], axis=1)
y = features_df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
log_message(f"Training set: {X_train.shape[0]} events, Test set: {X_test.shape[0]} events")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the BDT
log_message("\nTraining BDT classifier...")
bdt = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

bdt.fit(X_train_scaled, y_train)

# Make predictions
y_pred = bdt.predict(X_test_scaled)
y_prob = bdt.predict_proba(X_test_scaled)[:, 1]  # Probability of being CC

# Evaluate the model
log_message("\nModel Evaluation:")
log_message(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
log_message(f"Precision: {precision_score(y_test, y_pred):.4f}")
log_message(f"Recall: {recall_score(y_test, y_pred):.4f}")
log_message(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NC', 'CC'], yticklabels=['NC', 'CC'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
plt.close()
log_message("Saved confusion matrix plot")

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300)
plt.close()
log_message("Saved ROC curve plot")

# Feature Importance
plt.figure(figsize=(12, 8))
feature_importance = bdt.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in CC vs NC Classification')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=300)
plt.close()
log_message("Saved feature importance plot")

# Save feature importance data
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': bdt.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)
log_message("Saved feature importance data")

# Logit score distribution
plt.figure(figsize=(10, 6))
plt.hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='NC Events', density=True)
plt.hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='CC Events', density=True)
plt.xlabel('Logit Score (Probability of being a CC Event)')
plt.ylabel('Density')
plt.title('Logit Score Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "logit_score_distribution.png"), dpi=300)
plt.close()
log_message("Saved logit score distribution plot")

# Find optimal threshold
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
log_message(f"\nOptimal classification threshold: {optimal_threshold:.4f}")
log_message(f"At this threshold - True Positive Rate: {tpr[optimal_idx]:.4f}, False Positive Rate: {fpr[optimal_idx]:.4f}")

# Calculate precision at threshold properly
precision_at_threshold = precision_score(y_test, y_prob >= optimal_threshold)
log_message(f"Precision at optimal threshold: {precision_at_threshold:.4f}")

# Save threshold metrics
threshold_df = pd.DataFrame({
    'threshold': [optimal_threshold],
    'tpr': [tpr[optimal_idx]],
    'fpr': [fpr[optimal_idx]],
    'precision': [precision_at_threshold]
})
threshold_df.to_csv(os.path.join(results_dir, "optimal_threshold.csv"), index=False)
log_message("Saved optimal threshold metrics")

# Plot distribution of important features by event type
plt.figure(figsize=(15, 12))

# Get top 4 most important features
top_features = [X.columns[idx] for idx in sorted_idx[-4:]]

for i, feature in enumerate(top_features):
    plt.subplot(2, 2, i+1)
    plt.hist(features_df[features_df['label'] == 0][feature], bins=30, alpha=0.5, 
             label='NC Events', density=True)
    plt.hist(features_df[features_df['label'] == 1][feature], bins=30, alpha=0.5, 
             label='CC Events', density=True)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'Distribution of {feature}')
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "top_feature_distributions.png"), dpi=300)
plt.close()
log_message("Saved top feature distributions plot")

# Save the model
with open(os.path.join(results_dir, "cc_nc_bdt_model.pkl"), "wb") as f:
    pickle.dump(bdt, f)
log_message("Saved BDT model")

# Save the scaler for future use
with open(os.path.join(results_dir, "feature_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
log_message("Saved feature scaler")

# End timing
end_time = time.time()
execution_time = end_time - start_time
log_message(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

log_message("\nBDT training and evaluation complete!")
log_message(f"Results saved to {results_dir}")