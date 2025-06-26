#!/usr/bin/env python
"""
Compare CC vs NC classification with different unique DOM count thresholds:
- No threshold (0)
- At least 2 unique DOMs
- At least 5 unique DOMs

For each threshold, we'll create:
1. Histograms of total hits for CC vs NC events
2. Histograms of unique DOMs for CC vs NC events
3. BDT ROC curves
4. BDT logit score distributions
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
results_dir = os.path.join(data_dir, "dom_threshold_comparison")
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
log_file = os.path.join(results_dir, "comparison_log.txt")

# Function to log messages to both console and file
def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Initialize log file
with open(log_file, "w") as f:
    f.write("DOM Threshold Comparison Log\n")
    f.write("=" * 50 + "\n\n")

# Define the DOM thresholds to compare
dom_thresholds = [0, 2, 5]

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
        
        # For unique DOM thresholding, store this explicitly
        if unique_doms < 1:  # Skip events with no unique DOMs
            return None
        
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

# First, extract all event features once (for efficiency)
log_message("Extracting features from all events...")
all_event_features = []

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
    all_event_features.extend(file_features)

# Convert to DataFrame - all events with at least 1 unique DOM
all_df = pd.DataFrame(all_event_features)
log_message(f"\nCombined dataset has {len(all_df)} events")

# Save full feature data
feature_file = os.path.join(results_dir, "all_event_features.csv")
all_df.to_csv(feature_file, index=False)
log_message(f"Saved full feature data to {feature_file}")

# Check overall class balance
cc_count = sum(all_df['label'] == 1)
nc_count = sum(all_df['label'] == 0)
log_message(f"Overall class balance: {cc_count} CC events ({cc_count/len(all_df)*100:.1f}%), {nc_count} NC events ({nc_count/len(all_df)*100:.1f}%)")

# Initialize dictionaries to store results for each threshold
threshold_results = {}

# Process each threshold
for threshold in dom_thresholds:
    log_message(f"\n{'='*50}")
    log_message(f"Processing DOM threshold: {threshold}")
    
    # Filter events based on threshold
    if threshold > 0:
        filtered_df = all_df[all_df['unique_doms'] >= threshold]
    else:
        filtered_df = all_df.copy()
    
    log_message(f"After filtering: {len(filtered_df)} events")
    
    # Check class balance after filtering
    cc_count = sum(filtered_df['label'] == 1)
    nc_count = sum(filtered_df['label'] == 0)
    log_message(f"Class balance: {cc_count} CC events ({cc_count/len(filtered_df)*100:.1f}%), {nc_count} NC events ({nc_count/len(filtered_df)*100:.1f}%)")
    
    # 1. Histogram of total hits for CC vs NC
    plt.figure(figsize=(10, 6))
    max_hits = min(500, filtered_df['total_hits'].max())  # Cap at 500 for better visibility
    bins = np.arange(0, max_hits + 10, 5)  # Adjust bin size as needed
    
    plt.hist(filtered_df[filtered_df['label'] == 1]['total_hits'], bins=bins, 
             alpha=0.5, label='CC Events', edgecolor='black')
    plt.hist(filtered_df[filtered_df['label'] == 0]['total_hits'], bins=bins, 
             alpha=0.5, label='NC Events', edgecolor='black')
    plt.xlabel('Number of Hits')
    plt.ylabel('Number of Events')
    plt.title(f'Distribution of Total Hits: CC vs NC (≥{threshold} DOMs)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"hits_distribution_threshold_{threshold}.png"), dpi=300)
    plt.close()
    
    # 2. Histogram of unique DOMs for CC vs NC
    plt.figure(figsize=(10, 6))
    max_doms = min(50, filtered_df['unique_doms'].max())  # Cap at 50 for better visibility
    bins = np.arange(threshold, max_doms + 2, 1)  # Adjust bin size as needed
    
    plt.hist(filtered_df[filtered_df['label'] == 1]['unique_doms'], bins=bins, 
             alpha=0.5, label='CC Events', edgecolor='black')
    plt.hist(filtered_df[filtered_df['label'] == 0]['unique_doms'], bins=bins, 
             alpha=0.5, label='NC Events', edgecolor='black')
    plt.xlabel('Number of Unique DOMs')
    plt.ylabel('Number of Events')
    plt.title(f'Distribution of Unique DOMs: CC vs NC (≥{threshold} DOMs)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"doms_distribution_threshold_{threshold}.png"), dpi=300)
    plt.close()
    
    # Train BDT for this threshold
    X = filtered_df.drop(['label', 'file_source'], axis=1)
    y = filtered_df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    log_message(f"Training set: {X_train.shape[0]} events, Test set: {X_test.shape[0]} events")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train BDT
    bdt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    bdt.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = bdt.predict(X_test_scaled)
    y_prob = bdt.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    log_message(f"BDT Performance:")
    log_message(f"  Accuracy: {accuracy:.4f}")
    log_message(f"  Precision: {precision:.4f}")
    log_message(f"  Recall: {recall:.4f}")
    log_message(f"  F1 Score: {f1:.4f}")
    log_message(f"  ROC AUC: {roc_auc:.4f}")
    
    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_val = thresholds_roc[optimal_idx]
    
    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (≥{threshold} DOMs)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"roc_curve_threshold_{threshold}.png"), dpi=300)
    plt.close()
    
    # 4. Logit score distribution
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob[y_test == 0], bins=50, alpha=0.5, label='NC Events', density=True)
    plt.hist(y_prob[y_test == 1], bins=50, alpha=0.5, label='CC Events', density=True)
    plt.xlabel('Logit Score (Probability of being a CC Event)')
    plt.ylabel('Density')
    plt.title(f'Logit Score Distribution (≥{threshold} DOMs)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axvline(x=optimal_threshold_val, color='red', linestyle='--', 
               label=f'Optimal threshold: {optimal_threshold_val:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"logit_distribution_threshold_{threshold}.png"), dpi=300)
    plt.close()
    
    # Store results
    threshold_results[threshold] = {
        'total_events': len(filtered_df),
        'cc_events': cc_count,
        'nc_events': nc_count,
        'cc_percentage': cc_count/len(filtered_df)*100,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'optimal_threshold': optimal_threshold_val,
        'tpr_at_optimal': tpr[optimal_idx],
        'fpr_at_optimal': fpr[optimal_idx]
    }
    
    # Save the model and scaler
    with open(os.path.join(results_dir, f"bdt_model_threshold_{threshold}.pkl"), "wb") as f:
        pickle.dump(bdt, f)
    with open(os.path.join(results_dir, f"scaler_threshold_{threshold}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

# Create comparison summary plots
# 1. Bar chart of event counts by threshold
plt.figure(figsize=(10, 6))
thresholds = list(threshold_results.keys())
total_events = [threshold_results[t]['total_events'] for t in thresholds]
cc_events = [threshold_results[t]['cc_events'] for t in thresholds]
nc_events = [threshold_results[t]['nc_events'] for t in thresholds]

x = np.arange(len(thresholds))
width = 0.35

plt.bar(x, cc_events, width, label='CC Events', color='blue', alpha=0.7)
plt.bar(x, nc_events, width, bottom=cc_events, label='NC Events', color='orange', alpha=0.7)

plt.xlabel('Unique DOM Threshold')
plt.ylabel('Number of Events')
plt.title('Event Counts by DOM Threshold')
plt.xticks(x, [f'≥{t} DOMs' for t in thresholds])
plt.legend()

# Add total count labels
for i, total in enumerate(total_events):
    plt.text(i, total + 100, f'Total: {total}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "event_counts_by_threshold.png"), dpi=300)
plt.close()

# 2. Performance metrics comparison
plt.figure(figsize=(12, 6))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_values = {m: [threshold_results[t][m] for t in thresholds] for m in metrics}

for metric in metrics:
    plt.plot(thresholds, metric_values[metric], 'o-', label=metric.capitalize())

plt.xlabel('Unique DOM Threshold')
plt.ylabel('Score')
plt.title('Classification Performance by DOM Threshold')
plt.xticks(thresholds)
plt.ylim(0.5, 1.0)
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "performance_by_threshold.png"), dpi=300)
plt.close()

# Save comparison results
results_df = pd.DataFrame(threshold_results).T
results_df.index.name = 'dom_threshold'
results_df.to_csv(os.path.join(results_dir, "threshold_comparison_results.csv"))
log_message(f"Saved comparison results to threshold_comparison_results.csv")

# End timing
end_time = time.time()
execution_time = end_time - start_time
log_message(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

log_message("\nDOM threshold comparison complete!")
log_message(f"Results saved to {results_dir}")