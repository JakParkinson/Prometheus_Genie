import numpy as np
def group_hadronic_showers(prometheus_set):
    """
    Takes a prometheus_set DataFrame and replaces all hadronic shower particles 
    with a single 'Hadrons' particle with their combined energy.
    
    This should be run AFTER inject_in_cylinder and rotate_particles,
    but BEFORE passing the data to Prometheus.
    
    Parameters
    ----------
    prometheus_set : pd.DataFrame
        The DataFrame containing all final state particles
        
    Returns
    -------
    pd.DataFrame
        A modified DataFrame with hadronic showers grouped
    """
    # Deep copy to avoid modifying the original
    grouped_set = prometheus_set.copy()
    
    # PDG codes for particles we want to keep separate (neutrinos, leptons, neutral pions)
    lepton_pdgs = [11, 12, 13, 14, 15, 16, -11, -12, -13, -14, -15, -16]
    #special_pdgs = [111]  # neutral pion
    
    # Special PDG code for grouped hadrons
    HADRONS_PDG = -2000001006
    
    # Process each event individually
    for idx in grouped_set.index:
        pdg_codes = grouped_set.loc[idx, 'pdg_code']
        energies = grouped_set.loc[idx, 'e']
        thetas = grouped_set.loc[idx, 'theta']
        phis = grouped_set.loc[idx, 'phi']
        positions = grouped_set.loc[idx, 'position']
        
        # Separate particles into leptons/neutrinos/special and hadrons
        lepton_mask = np.array([abs(code) in lepton_pdgs for code in pdg_codes])
        hadron_mask = ~lepton_mask
        
        # If there are hadrons to group
        if any(hadron_mask):
            # Calculate total hadron energy
            total_hadron_energy = sum(energies[hadron_mask])
            
            # Find highest energy hadron for direction
            hadron_energies = energies[hadron_mask]
            max_energy_idx = np.argmax(hadron_energies)
            
            # Extract position and angles from max energy hadron
            hadron_thetas = thetas[hadron_mask]
            hadron_phis = phis[hadron_mask]
            hadron_positions = positions[hadron_mask]
            
            max_hadron_theta = hadron_thetas[max_energy_idx]
            max_hadron_phi = hadron_phis[max_energy_idx]
            max_hadron_position = hadron_positions[max_energy_idx]
            
            # Keep only leptons and special particles
            new_pdg_codes = np.append(pdg_codes[lepton_mask], HADRONS_PDG)
            new_energies = np.append(energies[lepton_mask], total_hadron_energy)
            new_thetas = np.append(thetas[lepton_mask], max_hadron_theta)
            new_phis = np.append(phis[lepton_mask], max_hadron_phi)
            new_positions = np.append(positions[lepton_mask], [max_hadron_position], axis=0)
            
            # Update all related columns
            grouped_set.at[idx, 'pdg_code'] = new_pdg_codes
            grouped_set.at[idx, 'e'] = new_energies
            grouped_set.at[idx, 'theta'] = new_thetas
            grouped_set.at[idx, 'phi'] = new_phis
            grouped_set.at[idx, 'position'] = new_positions
            
            # Update other position columns
            grouped_set.at[idx, 'pos_x'] = np.array([pos[0] for pos in new_positions])
            grouped_set.at[idx, 'pos_y'] = np.array([pos[1] for pos in new_positions])
            grouped_set.at[idx, 'pos_z'] = np.array([pos[2] for pos in new_positions])
            
            # Update all other array-based columns to match the new length
            for col in grouped_set.columns:
                val = grouped_set.loc[idx, col]
                if isinstance(val, np.ndarray) and len(val) != len(new_pdg_codes):
                    if col not in ['pdg_code', 'e', 'theta', 'phi', 'position', 'pos_x', 'pos_y', 'pos_z']:
                        if len(val) == len(pdg_codes):  # Only resize if it matches the original shape
                            # Keep values for leptons and add one value for hadrons (using the max energy hadron's value)
                            val_hadrons = val[hadron_mask][max_energy_idx] if any(hadron_mask) else None
                            grouped_set.at[idx, col] = np.append(val[lepton_mask], val_hadrons)
    
    return grouped_set