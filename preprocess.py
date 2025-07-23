import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from mace.calculators import mace_off
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import DataLoader
from mace.tools import torch_tools
from scm.plams import toASE, from_rdmol
import streamlit as st 

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

@st.cache_resource
def load_mace_calculator():
    model_path = './models/MACE-OFF24_medium.model'
    calculator = mace_off(model=model_path, device=device)
    return calculator

# Load MACE calculator
mace_calculator = load_mace_calculator()

def optimize_molecule_with_mmff(smiles: str):
    """
    Optimize molecule (MMFF).
    Returns RDKit mol object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    # Generate conformer
    conf_id = AllChem.EmbedMolecule(
        mol,
        randomSeed=42,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        maxAttempts=1000
    )
    
    if conf_id == -1:
        print(f"Warning: Conformer generation failed for SMILES: {smiles}")
        return None
    
    # Optimize 
    try:
        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
    except Exception as e:
        print(f"Warning: MMFF optimization failed for SMILES: {smiles}. Error: {e}")
        return None
    
    return mol

def get_mace_embedding(smiles: str, calculator):
    """
    Generates molecule embedding for SMILES string.
    """
    # Get optimized RDKit molecule
    opt_mol = optimize_molecule_with_mmff(smiles)
    if opt_mol is None:
        return None
    
    # Convert to ASE Atoms object
    try:
        plams_mol = from_rdmol(opt_mol)
        ase_atoms = toASE(plams_mol)
    except Exception as e:
        print(f"Warning: Conversion to ASE failed for SMILES: {smiles}. Error: {e}")
        return None

    # Get MACE descriptors
    try:
        atomic_descriptors = calculator.get_descriptors(ase_atoms)
    except Exception as e:
        print(f"Warning: MACE descriptor calculation failed for SMILES: {smiles}. Error: {e}")
        return None
    
    # Average embedding
    molecule_embedding = np.mean(atomic_descriptors, axis=0)
    
    return molecule_embedding

# MLP model
class SolubilityMLP(nn.Module):
    def __init__(self, input_size):
        super(SolubilityMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Load ensemble model
ensemble_states = torch.load('./models/aq_ensemble_64_models.pth')
feature_scalers = joblib.load('./models/aq_ensemble_64_feature_scalers.joblib')
target_scalers = joblib.load('./models/aq_ensemble_64_target_scalers.joblib')

def predict_with_ensemble(smiles: str):
    # Get embedding
    embedding = get_mace_embedding(smiles, mace_calculator)
    
    all_predictions = []
    X_new_tensor = torch.tensor(embedding, dtype=torch.float32).reshape(1, -1) # reshape to 2D tensor
    
    for i in range(len(ensemble_states)):
        # Get the model and scalers for this member of the ensemble
        model_state = ensemble_states[i]
        feature_scaler = feature_scalers[i]
        target_scaler = target_scalers[i]
        
        # Instantiate model and load state
        model = SolubilityMLP(256).to(device)
        model.load_state_dict(model_state)
        model.float()
        model.eval()
        
        # Scale the input and predict
        X_new_scaled = feature_scaler.transform(X_new_tensor.numpy())
        X_new_scaled_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            scaled_prediction = model(X_new_scaled_tensor)
            
        # Inverse transform the prediction
        final_prediction = target_scaler.inverse_transform(scaled_prediction.cpu().numpy())
        all_predictions.append(final_prediction[0][0])
        
    # Return the average of all predictions
    return all_predictions
