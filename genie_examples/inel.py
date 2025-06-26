#!/usr/bin/env python
"""
Inelasticity Prediction Model for CC Events in IceCube

This script:
1. Processes all parquet files (~60k events)
2. Calculates true inelasticity for CC events
3. Trains and evaluates several regression models
4. Saves results and figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import pickle

# Start timing
start_time = time.time()

# Directory containing your files
data_dir = "output/"
# Create a results directory for saving figures
results_dir = os.path.join(data_dir, "inelasticity_prediction")
os.makedirs(results_dir, exist_ok=True)

# Configure output
log_file = os.path.join(results_dir, "inelasticity_prediction.log")

# Function to log messages to both console and file
def log_message(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Initialize log file
with open(log_file, "w") as f:
    f.write("Inelasticity Prediction Log\n")
    f.write("=" * 50 + "\n\n")

# List of parquet files
sim_files = [
    "10000_events_simset_1600_seed_1000_20250516_112519.parquet",
    "10000_events_simset_1601_seed_1001_20250516_112659.parquet",
    "10000_events_simset_1602_seed_1002_20250516_112840.parquet",
    "10000_events_simset_1603_seed_1003_20250516_112857.parquet",
    "10000_events_simset_1604_seed_1004_20250516_112923.parquet",
    "10000_events_simset_1605_seed_1005_20250516_112950.parquet"
]

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

# Function to extract event features for regression
def extract_event_features(event_data, include_inelasticity=True):
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
        
        # Add file source for tracking
        features['file_source'] = event_data.get('file_source', 'unknown')
        
        # Add inelasticity if requested
        if include_inelasticity:
            features['inelasticity'] = inelasticity
        
        return features
    except Exception as e:
        log_message(f"Error extracting features: {e}")
        return None

# Extract features from all files
log_message("Extracting features from all files...")
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
    
    # Add file source for tracking
    for i, row in df.iterrows():
        row['file_source'] = sim_file
    
    # Extract features for CC events
    file_features = []
    for i, (idx, event_data) in enumerate(df.iterrows()):
        if i % 1000 == 0:  # Progress update
            log_message(f"  Processing event {i}/{len(df)}...")
        
        features = extract_event_features(event_data)
        if features is not None:
            file_features.append(features)
    
    log_message(f"  Extracted features for {len(file_features)} CC events with valid inelasticity")
    all_features.extend(file_features)

# Convert to DataFrame
features_df = pd.DataFrame(all_features)
log_message(f"\nCombined dataset has {len(features_df)} CC events with valid inelasticity")

# Save feature data
feature_file = os.path.join(results_dir, "inelasticity_features.csv")
features_df.to_csv(feature_file, index=False)
log_message(f"Saved feature data to {feature_file}")

# Plot distribution of inelasticity
plt.figure(figsize=(10, 6))
plt.hist(features_df['inelasticity'], bins=30, edgecolor='black')
plt.xlabel('Inelasticity')
plt.ylabel('Number of Events')
plt.title('Distribution of Inelasticity in CC Events')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(results_dir, "inelasticity_distribution.png"), dpi=300)
plt.close()
log_message("Saved inelasticity distribution plot")

# Define features and target
X = features_df.drop(['inelasticity', 'file_source'], axis=1)
y = features_df['inelasticity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_message(f"Training set: {X_train.shape[0]} events, Test set: {X_test.shape[0]} events")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

# Dictionary to store results
results = {}

for name, model in models.items():
    log_message(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Clip predictions to valid range [0, 1]
    y_pred_clipped = np.clip(y_pred, 0, 1)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_clipped)
    r2 = r2_score(y_test, y_pred_clipped)
    
    log_message(f"{name} Results:")
    log_message(f"  Mean Squared Error: {mse:.4f}")
    log_message(f"  Root Mean Squared Error: {rmse:.4f}")
    log_message(f"  Mean Absolute Error: {mae:.4f}")
    log_message(f"  R-squared: {r2:.4f}")
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred_clipped,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_clipped, alpha=0.3, s=5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('True Inelasticity')
    plt.ylabel('Predicted Inelasticity')
    plt.title(f'{name}: Predicted vs True Inelasticity')
    plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, f"{name.replace(' ', '_')}_predictions.png"), dpi=300)
    plt.close()
    
    # Save residuals plot
    residuals = y_test - y_pred_clipped
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_clipped, residuals, alpha=0.3, s=5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Inelasticity')
    plt.ylabel('Residuals')
    plt.title(f'{name}: Residuals Plot')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, f"{name.replace(' ', '_')}_residuals.png"), dpi=300)
    plt.close()
    
    # If model has feature importances, plot them
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'{name}: Feature Importance for Inelasticity Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{name.replace(' ', '_')}_feature_importance.png"), dpi=300)
        plt.close()
        
        # Save feature importance data
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(os.path.join(results_dir, f"{name.replace(' ', '_')}_feature_importance.csv"), index=False)

# Create comparison bar chart of model performance
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
rmse_values = [results[name]['rmse'] for name in model_names]
r2_values = [results[name]['r2'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='blue', alpha=0.7)
bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='green', alpha=0.7)

ax1.set_xlabel('Model')
ax1.set_ylabel('RMSE')
ax2.set_ylabel('R²')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=300)
plt.close()

# Save comparison results
comparison_df = pd.DataFrame({
    'Model': model_names,
    'RMSE': rmse_values,
    'MAE': [results[name]['mae'] for name in model_names],
    'R2': r2_values
})
comparison_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

# Train a simple model using only the top few features from best model
# Find the best model based on RMSE
best_model_name = min(results, key=lambda k: results[k]['rmse'])
log_message(f"\nBest model based on RMSE: {best_model_name}")

# If best model has feature importances, create a simple model using top features
if hasattr(results[best_model_name]['model'], 'feature_importances_'):
    best_model = results[best_model_name]['model']
    feature_importance = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    
    # Try different numbers of top features
    top_feature_counts = [3, 5, 10]
    simple_model_results = {}
    
    for n_features in top_feature_counts:
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
        
        log_message(f"Simple Model (Top {n_features} Features) Results:")
        log_message(f"  Mean Squared Error: {mse_simple:.4f}")
        log_message(f"  Root Mean Squared Error: {rmse_simple:.4f}")
        log_message(f"  Mean Absolute Error: {mae_simple:.4f}")
        log_message(f"  R-squared: {r2_simple:.4f}")
        
        # Store results
        simple_model_results[n_features] = {
            'model': simple_model,
            'predictions': y_pred_simple_clipped,
            'mse': mse_simple,
            'rmse': rmse_simple,
            'mae': mae_simple,
            'r2': r2_simple,
            'features': top_features,
            'coefficients': simple_model.coef_,
            'intercept': simple_model.intercept_
        }
        
        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_simple_clipped, alpha=0.3, s=5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('True Inelasticity')
        plt.ylabel('Predicted Inelasticity')
        plt.title(f'Simple Model (Top {n_features} Features): Predicted vs True Inelasticity')
        plt.text(0.05, 0.95, f'RMSE: {rmse_simple:.4f}\nR²: {r2_simple:.4f}', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(results_dir, f"simple_model_{n_features}_features.png"), dpi=300)
        plt.close()
        
        # Save model coefficients
        coef_df = pd.DataFrame({
            'Feature': top_features,
            'Coefficient': simple_model.coef_
        })
        coef_df.loc[len(coef_df)] = ['Intercept', simple_model.intercept_]
        coef_df.to_csv(os.path.join(results_dir, f"simple_model_{n_features}_coefficients.csv"), index=False)
        
        # Print formula
        formula = "Inelasticity = "
        for i, feature in enumerate(top_features):
            sign = "+" if simple_model.coef_[i] >= 0 else "-"
            formula += f" {sign} {abs(simple_model.coef_[i]):.4f} × {feature}"
        formula += f" + {simple_model.intercept_:.4f}"
        log_message(f"\nSimple Model Formula ({n_features} features):")
        log_message(formula)
    
    # Compare simple models with full models
    plt.figure(figsize=(14, 6))
    
    # Collect all model names and RMSE values
    all_model_names = list(results.keys()) + [f"Simple (Top {n})" for n in top_feature_counts]
    all_rmse_values = [results[name]['rmse'] for name in results.keys()] + [simple_model_results[n]['rmse'] for n in top_feature_counts]
    all_r2_values = [results[name]['r2'] for name in results.keys()] + [simple_model_results[n]['r2'] for n in top_feature_counts]
    
    x = np.arange(len(all_model_names))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, all_rmse_values, width, label='RMSE', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, all_r2_values, width, label='R²', color='green', alpha=0.7)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('R²')
    ax1.set_title('Model Performance Comparison (Including Simple Models)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_model_names, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_model_comparison.png"), dpi=300)
    plt.close()
    
    # Save all model comparison
    all_comparison_df = pd.DataFrame({
        'Model': all_model_names,
        'RMSE': all_rmse_values,
        'R2': all_r2_values
    })
    all_comparison_df.to_csv(os.path.join(results_dir, "all_model_comparison.csv"), index=False)
    
    # Save best simple model
    best_simple_n = min(simple_model_results, key=lambda k: simple_model_results[k]['rmse'])
    best_simple_model = simple_model_results[best_simple_n]['model']
    best_simple_features = simple_model_results[best_simple_n]['features']
    
    with open(os.path.join(results_dir, "best_simple_model.pkl"), "wb") as f:
        pickle.dump({
            'model': best_simple_model,
            'features': best_simple_features,
            'scaler': scaler_top
        }, f)
    log_message(f"Saved best simple model (Top {best_simple_n} features)")

# Save best overall model
best_model = results[best_model_name]['model']
with open(os.path.join(results_dir, "best_model.pkl"), "wb") as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler
    }, f)
log_message(f"Saved best overall model ({best_model_name})")

# End timing
end_time = time.time()
execution_time = end_time - start_time
log_message(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")

log_message("\nInelasticity prediction modeling complete!")
log_message(f"Results saved to {results_dir}")