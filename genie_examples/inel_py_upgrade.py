#!/usr/bin/env python
"""
Improved Inelasticity Prediction using BDT Muon Confidence

Approach:
1. Train a BDT to distinguish muon hits from hadronic shower hits
2. Calculate a muon confidence score for each CC event
3. Use this score as an additional feature in inelasticity prediction models
4. Train and evaluate several regression models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns

# Start timing
start_time = time.time()

# Directory containing your files
data_dir = "output/"
# Create a results directory for saving figures
results_dir = os.path.join(data_dir, "upgrade_inelasticity_prediction_with_confidence")
os.makedirs(results_dir, exist_ok=True)

# Configure output
log_file = os.path.join(results_dir, "upgrade_inelasticity_prediction.log")

# Function to log messages to both console and file
def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Initialize log file
with open(log_file, "w") as f:
    f.write("Improved Inelasticity Prediction with BDT Muon Confidence\n")
    f.write("=" * 50 + "\n\n")

# List of all parquet files (original + additional)
sim_files = [
    "10000_events_simset_1800_seed_1000_upgrade_internal_20250516_224300.parquet",
    "10000_events_simset_1810_seed_1010_upgrade_internal_20250516_224002.parquet",
    "10000_events_simset_1801_seed_1001_upgrade_internal_20250516_224251.parquet",
    "10000_events_simset_1809_seed_1009_upgrade_internal_20250516_224018.parquet",
    "10000_events_simset_1805_seed_1005_upgrade_internal_20250516_224152.parquet",
    "10000_events_simset_1806_seed_1006_upgrade_internal_20250516_224137.parquet",
    "10000_events_simset_1807_seed_1007_upgrade_internal_20250516_224123.parquet",   
    "10000_events_simset_1804_seed_1004_upgrade_internal_20250516_224203.parquet",
    "10000_events_simset_1808_seed_1008_upgrade_internal_20250516_224038.parquet",
    "10000_events_simset_1803_seed_1003_upgrade_internal_20250516_224216.parquet",
    "10000_events_simset_1802_seed_1002_upgrade_internal_20250516_224236.parquet"
]
# Check which files exist
existing_files = []
for file in sim_files:
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        existing_files.append(file)
    else:
        log_message(f"Warning: File not found: {file}")

log_message(f"Found {len(existing_files)} existing files out of {len(sim_files)}")
sim_files = existing_files

# Function to extract hit-level features for muon vs hadron classification
def extract_hit_features(event_data):
    """Extract features for each hit to classify as muon or hadron"""
    try:
        # Skip if not a CC event
        if event_data['mc_truth']['interaction'] != 1:
            return []
        
        # Get photon data
        photons = event_data['photons']
        
        # Skip if no hits
        if len(photons['t']) == 0:
            return []
        
        # Get final state particles
        final_state_particles = event_data['mc_truth']['final_state_type']
        
        # Map hits to particle types
        id_idx_array = photons['id_idx']
        particle_types = np.zeros_like(id_idx_array)
        for i, idx in enumerate(id_idx_array):
            if idx > 0 and idx <= len(final_state_particles):
                particle_types[i] = final_state_particles[idx-1]
        
        # Label hits: 1 for muon, 0 for hadron
        labels = (particle_types == 13).astype(int)
        
        # Position and time info
        pos_x = photons['sensor_pos_x']
        pos_y = photons['sensor_pos_y']
        pos_z = photons['sensor_pos_z']
        hit_times = photons['t']
        
        # Calculate event center and mean time
        center_x = np.mean(pos_x)
        center_y = np.mean(pos_y)
        center_z = np.mean(pos_z)
        mean_time = np.mean(hit_times)
        
        # Features for each hit
        hit_features = []
        for i in range(len(hit_times)):
            # Distance from center
            r = np.sqrt((pos_x[i] - center_x)**2 + (pos_y[i] - center_y)**2 + (pos_z[i] - center_z)**2)
            
            # Time relative to mean
            rel_time = hit_times[i] - mean_time
            
            # Feature dict for this hit
            features = {
                'time': hit_times[i],
                'rel_time': rel_time,
                'pos_x': pos_x[i],
                'pos_y': pos_y[i],
                'pos_z': pos_z[i],
                'r': r,
                'string_id': photons['string_id'][i],
                'sensor_id': photons['sensor_id'][i],
                'label': labels[i]
            }
            
            # Add event ID for tracking
            features['event_id'] = event_data.get('event_id', -1)
            features['file_source'] = event_data.get('file_source', 'unknown')
            
            hit_features.append(features)
        
        return hit_features
    except Exception as e:
        log_message(f"Error extracting hit features: {e}")
        return []

# Function to calculate inelasticity
def calculate_inelasticity(event_data):
    """Calculate inelasticity for a CC event"""
    try:
        # Get neutrino energy from mc_truth
        neutrino_energy = event_data['mc_truth']['initial_state_energy']
        
        # Get final state particles
        final_state_particles = event_data['mc_truth']['final_state_type']
        final_state_energies = event_data['mc_truth']['final_state_energy']
        
        # Find muon positions (PDG code 13)
        muon_indices = np.where(np.abs(final_state_particles) == 13)[0]
        
        # If no muons, return None
        if len(muon_indices) == 0:
            return None
        
        # Get first muon energy (assuming primary muon is first)
        muon_energy = final_state_energies[muon_indices[0]]
        
        # Calculate inelasticity: y = 1 - Eμ/Eν
        inelasticity = 1.0 - (muon_energy / neutrino_energy)
        
        # Check if inelasticity is physically meaningful
        if 0 <= inelasticity <= 1:
            return inelasticity
        else:
            return None
    except Exception as e:
        return None

# Function to extract event-level features for inelasticity prediction
def extract_event_features(event_data, muon_confidence=None, include_inelasticity=True):
    """Extract features for inelasticity prediction"""
    try:
        # Only process CC events
        if event_data['mc_truth']['interaction'] != 1:  # Not a CC event
            return None
        
        # Calculate inelasticity if requested
        inelasticity = calculate_inelasticity(event_data) if include_inelasticity else None
        
        # Skip events where inelasticity can't be calculated
        if include_inelasticity and inelasticity is None:
            return None
        
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
        
        # Skip events with too few hits
        if unique_doms < 2:
            return None
        
        # Spatial features
        pos_x = photons['sensor_pos_x']
        pos_y = photons['sensor_pos_y']
        pos_z = photons['sensor_pos_z']
        
        # Timing features
        hit_times = photons['t']
        
        # Calculate features
        # 1. Basic count features
        hits_per_dom = total_hits / unique_doms
        
        # 2. Spatial features
        max_distance = 0
        if len(pos_x) > 1:
            from scipy.spatial.distance import pdist
            positions = np.vstack((pos_x, pos_y, pos_z)).T
            max_distance = np.max(pdist(positions))
        
        # 3. Time features
        time_span = np.max(hit_times) - np.min(hit_times)
        time_mean = np.mean(hit_times)
        time_std = np.std(hit_times)
        
        # 4. Position features
        center_x = np.mean(pos_x)
        center_y = np.mean(pos_y)
        center_z = np.mean(pos_z)
        r_squared = (pos_x - center_x)**2 + (pos_y - center_y)**2 + (pos_z - center_z)**2
        r_mean = np.mean(np.sqrt(r_squared))
        r_std = np.std(np.sqrt(r_squared))
        
        # 5. Correlation features (if enough hits)
        time_z_corr = 0
        time_x_corr = 0
        time_y_corr = 0
        if len(hit_times) > 5:
            # Normalize position and time
            z_norm = (pos_z - center_z) / (np.std(pos_z) + 1e-6)
            x_norm = (pos_x - center_x) / (np.std(pos_x) + 1e-6)
            y_norm = (pos_y - center_y) / (np.std(pos_y) + 1e-6)
            t_norm = (hit_times - time_mean) / (np.std(hit_times) + 1e-6)
            
            time_z_corr = np.mean(z_norm * t_norm)
            time_x_corr = np.mean(x_norm * t_norm)
            time_y_corr = np.mean(y_norm * t_norm)
        
        # 6. Distribution features
        x_span = np.max(pos_x) - np.min(pos_x)
        y_span = np.max(pos_y) - np.min(pos_y)
        z_span = np.max(pos_z) - np.min(pos_z)
        
        # 7. Topology features
        unique_strings = len(np.unique(string_ids))
        hits_per_string = total_hits / max(1, unique_strings)
        
        # 8. Advanced shape features
        x_std = np.std(pos_x)
        y_std = np.std(pos_y)
        z_std = np.std(pos_z)
        
        # Calculate elongation ratio
        stds = np.array([x_std, y_std, z_std])
        if np.mean(stds) > 0:
            elongation = np.max(stds) / np.mean(stds)
        else:
            elongation = 1.0
        
        # Feature dictionary
        features = {
            'total_hits': total_hits,
            'unique_doms': unique_doms,
            'hits_per_dom': hits_per_dom,
            'max_distance': max_distance,
            'time_span': time_span,
            'time_mean': time_mean,
            'time_std': time_std,
            'r_mean': r_mean,
            'r_std': r_std,
            'time_z_corr': time_z_corr,
            'time_x_corr': time_x_corr,
            'time_y_corr': time_y_corr,
            'z_span': z_span,
            'x_span': x_span,
            'y_span': y_span,
            'x_std': x_std,
            'y_std': y_std,
            'z_std': z_std,
            'center_z': center_z,
            'elongation': elongation,
            'unique_strings': unique_strings,
            'hits_per_string': hits_per_string
        }
        
        # Add event ID and file source for tracking
        features['event_id'] = event_data.get('event_id', -1)
        features['file_source'] = event_data.get('file_source', 'unknown')
        
        # Add muon confidence if provided
        if muon_confidence is not None:
            features['muon_confidence'] = muon_confidence
        
        # Add inelasticity if requested
        if include_inelasticity:
            features['inelasticity'] = inelasticity
        
        return features
    except Exception as e:
        log_message(f"Error extracting event features: {e}")
        return None

# PART 1: BDT for Muon vs Hadron Classification
log_message("\nPART 1: Training BDT to distinguish muon hits from hadron hits")

# Extract hit features from all files
all_hit_features = []

# Process each file
for file_idx, sim_file in enumerate(sim_files):
    file_path = os.path.join(data_dir, sim_file)
    
    log_message(f"\nProcessing {sim_file} ({file_idx+1}/{len(sim_files)})...")
    
    # Load the file
    df = pd.read_parquet(file_path)
    log_message(f"Loaded {len(df)} events")
    
    # Add event ID and file source for tracking
    for i, row in df.iterrows():
        row['event_id'] = i
        row['file_source'] = sim_file
    
    # Extract hit features for muon vs hadron classification
    file_hit_features = []
    cc_count = 0
    
    for i, (idx, event_data) in enumerate(df.iterrows()):
        if i % 1000 == 0:  # Progress update
            log_message(f"  Processing event {i}/{len(df)}...")
        
        # Skip if not a CC event
        if event_data['mc_truth']['interaction'] != 1:
            continue
        
        cc_count += 1
        hit_features = extract_hit_features(event_data)
        if hit_features:
            file_hit_features.extend(hit_features)
    
    log_message(f"  Extracted features for {len(file_hit_features)} hits from {cc_count} CC events")
    all_hit_features.extend(file_hit_features)

# Convert to DataFrame
hit_df = pd.DataFrame(all_hit_features)
log_message(f"\nTotal hits collected: {len(hit_df)}")

# Check class balance
muon_count = sum(hit_df['label'] == 1)
hadron_count = sum(hit_df['label'] == 0)
log_message(f"Class balance: {muon_count} muon hits ({muon_count/len(hit_df)*100:.1f}%), {hadron_count} hadron hits ({hadron_count/len(hit_df)*100:.1f}%)")

# Save hit statistics
hit_stats = {
    'total_hits': len(hit_df),
    'muon_hits': muon_count,
    'hadron_hits': hadron_count,
    'muon_percentage': muon_count/len(hit_df)*100
}
pd.DataFrame([hit_stats]).to_csv(os.path.join(results_dir, "hit_statistics.csv"), index=False)

# Prepare features and target for BDT
X_hit = hit_df.drop(['label', 'event_id', 'file_source'], axis=1)
y_hit = hit_df['label']

# Split the data
X_hit_train, X_hit_test, y_hit_train, y_hit_test = train_test_split(X_hit, y_hit, test_size=0.3, random_state=42, stratify=y_hit)
log_message(f"Training set: {X_hit_train.shape[0]} hits, Test set: {X_hit_test.shape[0]} hits")

# Scale features
hit_scaler = StandardScaler()
X_hit_train_scaled = hit_scaler.fit_transform(X_hit_train)
X_hit_test_scaled = hit_scaler.transform(X_hit_test)

# Train BDT classifier for muon vs hadron
log_message("\nTraining BDT classifier for muon vs hadron hits...")
hit_bdt = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

hit_bdt.fit(X_hit_train_scaled, y_hit_train)

# Make predictions
y_hit_pred = hit_bdt.predict(X_hit_test_scaled)
y_hit_prob = hit_bdt.predict_proba(X_hit_test_scaled)[:, 1]  # Probability of being a muon hit

# Evaluate model
hit_accuracy = accuracy_score(y_hit_test, y_hit_pred)
log_message(f"BDT Accuracy: {hit_accuracy:.4f}")

# Feature importance
hit_importance = hit_bdt.feature_importances_
hit_sorted_idx = np.argsort(hit_importance)
hit_feature_importance = pd.DataFrame({
    'Feature': X_hit.columns[hit_sorted_idx],
    'Importance': hit_importance[hit_sorted_idx]
}).sort_values('Importance', ascending=False)

log_message("\nTop features for muon vs hadron classification:")
for i, (feature, importance) in enumerate(zip(hit_feature_importance['Feature'][:5], hit_feature_importance['Importance'][:5])):
    log_message(f"  {i+1}. {feature}: {importance:.4f}")

# Save feature importance
hit_feature_importance.to_csv(os.path.join(results_dir, "muon_hadron_feature_importance.csv"), index=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(hit_sorted_idx)), hit_importance[hit_sorted_idx])
plt.yticks(range(len(hit_sorted_idx)), X_hit.columns[hit_sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Muon vs Hadron Classification')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "muon_hadron_feature_importance.png"), dpi=300)
plt.close()

# Plot ROC curve
from sklearn.metrics import roc_curve, auc
hit_fpr, hit_tpr, _ = roc_curve(y_hit_test, y_hit_prob)
hit_roc_auc = auc(hit_fpr, hit_tpr)

plt.figure(figsize=(8, 6))
plt.plot(hit_fpr, hit_tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {hit_roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Muon vs Hadron Classification')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "muon_hadron_roc.png"), dpi=300)
plt.close()

# Plot logit score distribution
plt.figure(figsize=(10, 6))
plt.hist(y_hit_prob[y_hit_test == 0], bins=50, alpha=0.5, label='Hadron Hits', density=True)
plt.hist(y_hit_prob[y_hit_test == 1], bins=50, alpha=0.5, label='Muon Hits', density=True)
plt.xlabel('Logit Score (Probability of being a Muon Hit)')
plt.ylabel('Density')
plt.title('Logit Score Distribution for Muon vs Hadron Hits')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "muon_hadron_logit.png"), dpi=300)
plt.close()

# Save the hit BDT model
with open(os.path.join(results_dir, "muon_hadron_bdt.pkl"), "wb") as f:
    pickle.dump({
        'model': hit_bdt,
        'scaler': hit_scaler
    }, f)
log_message("Saved muon vs hadron BDT model")

# PART 2: Calculate Event-Level Muon Confidence and Extract Features
log_message("\nPART 2: Calculating event-level muon confidence scores")

# Function to calculate muon confidence for an event
def calculate_muon_confidence(event_data, hit_bdt, hit_scaler):
    """Calculate average confidence score for muon hits in an event"""
    try:
        # Get photon data
        photons = event_data['photons']
        
        # Skip if no hits
        if len(photons['t']) == 0:
            return 0.0
        
        # Extract features for each hit
        hit_features = []
        for i in range(len(photons['t'])):
            # Distance from center
            pos_x = photons['sensor_pos_x'][i]
            pos_y = photons['sensor_pos_y'][i]
            pos_z = photons['sensor_pos_z'][i]
            
            center_x = np.mean(photons['sensor_pos_x'])
            center_y = np.mean(photons['sensor_pos_y'])
            center_z = np.mean(photons['sensor_pos_z'])
            
            r = np.sqrt((pos_x - center_x)**2 + (pos_y - center_y)**2 + (pos_z - center_z)**2)
            
            # Time relative to mean
            hit_time = photons['t'][i]
            mean_time = np.mean(photons['t'])
            rel_time = hit_time - mean_time
            
            # Feature dict for this hit
            features = {
                'time': hit_time,
                'rel_time': rel_time,
                'pos_x': pos_x,
                'pos_y': pos_y,
                'pos_z': pos_z,
                'r': r,
                'string_id': photons['string_id'][i],
                'sensor_id': photons['sensor_id'][i]
            }
            
            hit_features.append(features)
        
        # Convert to DataFrame
        hit_df = pd.DataFrame(hit_features)
        
        # Scale features
        X_hit = hit_scaler.transform(hit_df)
        
        # Get muon probabilities
        muon_probs = hit_bdt.predict_proba(X_hit)[:, 1]
        
        # Calculate confidence metrics
        mean_prob = np.mean(muon_probs)
        top_10_prob = np.mean(np.sort(muon_probs)[-min(10, len(muon_probs)):])
        
        # Return the higher of the two (more conservative)
        return max(mean_prob, top_10_prob)
    except Exception as e:
        log_message(f"Error calculating muon confidence: {e}")
        return 0.0

# Process each file to calculate muon confidence and extract event features
all_event_features = []

for file_idx, sim_file in enumerate(sim_files):
    file_path = os.path.join(data_dir, sim_file)
    
    log_message(f"\nProcessing {sim_file} for event features ({file_idx+1}/{len(sim_files)})...")
    
    # Load the file
    df = pd.read_parquet(file_path)
    
    # Add event ID and file source for tracking
    for i, row in df.iterrows():
        row['event_id'] = i
        row['file_source'] = sim_file
    
    # Process CC events
    file_event_features = []
    cc_count = 0
    
    for i, (idx, event_data) in enumerate(df.iterrows()):
        if i % 1000 == 0:  # Progress update
            log_message(f"  Processing event {i}/{len(df)}...")
        
        # Skip if not a CC event
        if event_data['mc_truth']['interaction'] != 1:
            continue
        
        cc_count += 1
        
        # Calculate muon confidence
        muon_confidence = calculate_muon_confidence(event_data, hit_bdt, hit_scaler)
        
        # Extract event features with muon confidence
        features = extract_event_features(event_data, muon_confidence)
        if features is not None:
            file_event_features.append(features)
    
    log_message(f"  Extracted features for {len(file_event_features)} out of {cc_count} CC events")
    all_event_features.extend(file_event_features)

# Convert to DataFrame
event_df = pd.DataFrame(all_event_features)
log_message(f"\nTotal CC events with valid features: {len(event_df)}")

# Save all event features
event_df.to_csv(os.path.join(results_dir, "all_cc_event_features.csv"), index=False)

# Plot distribution of muon confidence scores
plt.figure(figsize=(10, 6))
plt.hist(event_df['muon_confidence'], bins=50, edgecolor='black')
plt.xlabel('Muon Confidence Score')
plt.ylabel('Number of Events')
plt.title('Distribution of Muon Confidence Scores in CC Events')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "muon_confidence_distribution.png"), dpi=300)
plt.close()

# Plot correlation between muon confidence and inelasticity
plt.figure(figsize=(10, 6))
plt.scatter(event_df['muon_confidence'], event_df['inelasticity'], alpha=0.3, s=5)
plt.xlabel('Muon Confidence Score')
plt.ylabel('Inelasticity')
plt.title('Correlation Between Muon Confidence and Inelasticity')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "muon_confidence_vs_inelasticity.png"), dpi=300)
plt.close()

# Calculate correlation coefficient
corr = np.corrcoef(event_df['muon_confidence'], event_df['inelasticity'])[0, 1]
log_message(f"Correlation between muon confidence and inelasticity: {corr:.4f}")

# PART 3: Train Inelasticity Models with Muon Confidence as Feature
log_message("\nPART 3: Training inelasticity models with muon confidence as a feature")

# Plot inelasticity distribution
plt.figure(figsize=(10, 6))
plt.hist(event_df['inelasticity'], bins=50, edgecolor='black')
plt.xlabel('Inelasticity')
plt.ylabel('Number of Events')
plt.title('Distribution of Inelasticity in CC Events')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "inelasticity_distribution.png"), dpi=300)
plt.close()

# Compare models with and without muon confidence
model_comparisons = []

# Function to train and evaluate a model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    log_message(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_clipped = np.clip(y_pred, 0, 1)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_clipped)
    r2 = r2_score(y_test, y_pred_clipped)
    
    log_message(f"{model_name} Results:")
    log_message(f"  Mean Squared Error: {mse:.4f}")
    log_message(f"  Root Mean Squared Error: {rmse:.4f}")
    log_message(f"  Mean Absolute Error: {mae:.4f}")
    log_message(f"  R-squared: {r2:.4f}")
    
    return {
        'model': model,
        'predictions': y_pred_clipped,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Define versions to test
versions = [
    {"name": "Without Muon Confidence", "include_confidence": False},
    {"name": "With Muon Confidence", "include_confidence": True}
]

for version in versions:
    log_message(f"\nTraining models {version['name']}...")
    
    # Prepare features
    if version["include_confidence"]:
        X = event_df.drop(['inelasticity', 'event_id', 'file_source'], axis=1)
    else:
        X = event_df.drop(['inelasticity', 'muon_confidence', 'event_id', 'file_source'], axis=1)
    
    y = event_df['inelasticity']


# Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    log_message(f"Training set: {X_train.shape[0]} events, Test set: {X_test.shape[0]} events")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    version_results = {}
    
    for name, model in models.items():
        full_name = f"{name} {version['name']}"
        results = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, full_name, model)
        version_results[name] = results
        
        # Add to comparison
        model_comparisons.append({
            'Version': version['name'],
            'Model': name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'R2': results['r2']
        })
        
        # Plot predictions vs true values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, results['predictions'], alpha=0.3, s=5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('True Inelasticity')
        plt.ylabel('Predicted Inelasticity')
        plt.title(f'{full_name}: Predicted vs True Inelasticity')
        plt.text(0.05, 0.95, f'RMSE: {results["rmse"]:.4f}\nR²: {results["r2"]:.4f}', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(results_dir, f"{name.replace(' ', '_')}_{version['name'].replace(' ', '_')}_predictions.png"), dpi=300)
        plt.close()
        
        # If model has feature importances, plot them
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'{full_name}: Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{name.replace(' ', '_')}_{version['name'].replace(' ', '_')}_feature_importance.png"), dpi=300)
            plt.close()
            
            # Save feature importance data
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(results_dir, f"{name.replace(' ', '_')}_{version['name'].replace(' ', '_')}_feature_importance.csv"), index=False)
    
    # For the best model in this version (Gradient Boosting), create simple model with top features
    gb_results = version_results['Gradient Boosting']
    gb_model = gb_results['model']
    
    if hasattr(gb_model, 'feature_importances_'):
        feature_importance = gb_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        
        # Try with top 5 features
        n_features = 5
        top_features = [X.columns[i] for i in sorted_idx[-n_features:]]
        log_message(f"\nTraining simple model with top {n_features} features: {top_features}")
        
        # Extract and scale these features
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]
        
        scaler_top = StandardScaler()
        X_train_top_scaled = scaler_top.fit_transform(X_train_top)
        X_test_top_scaled = scaler_top.transform(X_test_top)
        
        # Train a simple linear model
        simple_model = LinearRegression()
        simple_model.fit(X_train_top_scaled, y_train)
        
        # Make predictions
        y_pred_simple = simple_model.predict(X_test_top_scaled)
        y_pred_simple_clipped = np.clip(y_pred_simple, 0, 1)
        
        # Evaluate model
        mse_simple = mean_squared_error(y_test, y_pred_simple_clipped)
        rmse_simple = np.sqrt(mse_simple)
        mae_simple = mean_absolute_error(y_test, y_pred_simple_clipped)
        r2_simple = r2_score(y_test, y_pred_simple_clipped)
        
        log_message(f"Simple Model (Top {n_features} Features) {version['name']} Results:")
        log_message(f"  Mean Squared Error: {mse_simple:.4f}")
        log_message(f"  Root Mean Squared Error: {rmse_simple:.4f}")
        log_message(f"  Mean Absolute Error: {mae_simple:.4f}")
        log_message(f"  R-squared: {r2_simple:.4f}")
        
        # Add to comparison
        model_comparisons.append({
            'Version': version['name'],
            'Model': f'Simple Linear (Top {n_features})',
            'RMSE': rmse_simple,
            'MAE': mae_simple,
            'R2': r2_simple
        })
        
        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_simple_clipped, alpha=0.3, s=5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('True Inelasticity')
        plt.ylabel('Predicted Inelasticity')
        plt.title(f'Simple Model (Top {n_features} Features) {version["name"]}: Predicted vs True Inelasticity')
        plt.text(0.05, 0.95, f'RMSE: {rmse_simple:.4f}\nR²: {r2_simple:.4f}', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(results_dir, f"simple_model_{n_features}_features_{version['name'].replace(' ', '_')}.png"), dpi=300)
        plt.close()
        
        # Save model coefficients
        coef_df = pd.DataFrame({
            'Feature': top_features,
            'Coefficient': simple_model.coef_
        })
        coef_df.loc[len(coef_df)] = ['Intercept', simple_model.intercept_]
        coef_df.to_csv(os.path.join(results_dir, f"simple_model_{n_features}_coefficients_{version['name'].replace(' ', '_')}.csv"), index=False)
        
        # Print formula
        formula = "Inelasticity = "
        for i, feature in enumerate(top_features):
            sign = "+" if simple_model.coef_[i] >= 0 else "-"
            formula += f" {sign} {abs(simple_model.coef_[i]):.4f} × {feature}"
        formula += f" + {simple_model.intercept_:.4f}"
        log_message(f"\nSimple Model Formula ({n_features} features) {version['name']}:")
        log_message(formula)
        
        # Store simple model results
        version_results['Simple Linear'] = {
            'model': simple_model,
            'predictions': y_pred_simple_clipped,
            'mse': mse_simple,
            'rmse': rmse_simple,
            'mae': mae_simple,
            'r2': r2_simple,
            'features': top_features,
            'formula': formula
        }

# Create comparison DataFrame
comparison_df = pd.DataFrame(model_comparisons)
comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

# Plot comparison of models with and without muon confidence
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='RMSE', hue='Version', data=comparison_df)
plt.title('RMSE by Model and Version')
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "rmse_comparison.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='R2', hue='Version', data=comparison_df)
plt.title('R² by Model and Version')
plt.xticks(rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "r2_comparison.png"), dpi=300)
plt.close()

# Find best model
best_row = comparison_df.loc[comparison_df['RMSE'].idxmin()]
best_version = best_row['Version']
best_model_name = best_row['Model']
best_rmse = best_row['RMSE']
best_r2 = best_row['R2']

log_message(f"\nBest Model: {best_model_name} {best_version}")
log_message(f"  RMSE: {best_rmse:.4f}")
log_message(f"  R²: {best_r2:.4f}")

# Save best model information
with open(os.path.join(results_dir, "best_model_info.pkl"), "wb") as f:
    pickle.dump({
        'best_version': best_version,
        'best_model_name': best_model_name,
        'best_rmse': best_rmse,
        'best_r2': best_r2,
        'hit_bdt': hit_bdt,
        'hit_scaler': hit_scaler
    }, f)
log_message(f"Saved best model information")

# End timing
end_time = time.time()
execution_time = end_time - start_time
log_message(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

log_message("\nInelasticity prediction modeling complete!")
log_message(f"Results saved to {results_dir}")