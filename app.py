import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric import DataLoader
from mace.tools import torch_tools
import matplotlib.pyplot as plt

# --- Step 1: Define the MLP Model Architecture ---
# This class must be identical to the one used for training your ensemble.
class SolubilityMLP(nn.Module):
    def __init__(self, input_size):
        super(SolubilityMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# --- Step 2: Functions to Load Models (with Caching) ---
# Streamlit's cache decorator prevents reloading the models on every user interaction.

@st.cache_resource
def load_mace_calculator():
    """Loads the MACE model for embedding generation."""
    model_path = 'models/MACE-OFF24_medium.model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = torch.load(model_path, map_location=device)
    print("MACE calculator loaded successfully.")
    return calculator

@st.cache_resource
def load_mlp_ensemble():
    """Loads the 64-model MLP ensemble for prediction."""
    path = 'models/ensemble_64_models_small.pth'
    input_size = 256  # Must match the training input size
    # Load to CPU, as the deployment environment may not have a GPU
    device = torch.device('cpu')
    state_dicts = torch.load(path, map_location=device)
    
    models = []
    for state_dict in state_dicts:
        model = SolubilityMLP(input_size)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        models.append(model)
    print(f"MLP ensemble of {len(models)} models loaded successfully.")
    return models

# --- Step 3: Helper Functions for Molecule Processing ---

def rdkit_to_ase(mol: Chem.Mol) -> Atoms:
    """Converts an RDKit molecule object to an ASE Atoms object."""
    positions = mol.GetConformer().GetPositions()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return Atoms(numbers=atomic_numbers, positions=positions)

def get_mace_embedding(smiles: str, calculator) -> np.ndarray:
    """
    Generates a MACE embedding for a given SMILES string.
    Returns a single 256-dimensional feature vector (mean of node features).
    """
    # 1. Convert SMILES to RDKit Mol and add hydrogens
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    # 2. Convert to ASE Atoms
    atoms = rdkit_to_ase(mol)

    # 3. Get MACE node features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader([atoms], batch_size=1, shuffle=False)
    for batch in data_loader:
        batch = batch.to(device)
        output = calculator(batch, compute_force=False)
        node_feats = output.get("node_feats")

    if node_feats is None:
        raise RuntimeError("Failed to generate MACE node features.")

    # 4. Average node features to get a single graph-level embedding
    graph_embedding = node_feats.mean(axis=0).detach().cpu().numpy()
    return graph_embedding


# --- Step 4: Main Application Logic ---

# Load the models once
mace_calculator = load_mace_calculator()
mlp_ensemble = load_mlp_ensemble()

# --- Streamlit User Interface ---
st.title("Solubility Predictor")
st.markdown(
    "Enter a SMILES string to predict its **log solubility (in mols/litre)**. "
    "The prediction is made by an ensemble of 64 MLP models, and the uncertainty "
    "is the standard deviation of their predictions."
)

# User input
smiles_input = st.text_input("Enter SMILES String:", "CCO") # Default to ethanol
predict_button = st.button("Predict")

# Prediction logic
if predict_button and smiles_input:
    with st.spinner("Calculating..."):
        try:
            # 1. Get MACE embedding
            mace_embedding = get_mace_embedding(smiles_input, mace_calculator)
            
            # 2. Prepare for MLP input
            X_new = torch.tensor(mace_embedding).view(1, -1) # Shape: (1, 256)

            # 3. Get predictions from the ensemble
            with torch.no_grad():
                all_preds = [model(X_new).numpy() for model in mlp_ensemble]
                all_preds = np.stack(all_preds) # Shape: (64, 1, 1)
            
            # 4. Calculate mean and standard deviation
            mean_pred = all_preds.mean()
            std_pred = all_preds.std()

            # 5. Display numerical results
            st.success(f"**Predicted Log Solubility:** `{mean_pred:.4f}`")
            st.info(f"**Prediction Uncertainty (Std Dev):** `{std_pred:.4f}`")

            # 6. Create and display histogram of predictions
            st.subheader("Distribution of Ensemble Predictions")
            fig, ax = plt.subplots()
            
            # Flatten the array for the histogram
            predictions_flat = all_preds.flatten()
            
            ax.hist(predictions_flat, bins=15, edgecolor='black', alpha=0.7)
            
            # Add a vertical line for the mean prediction
            ax.axvline(mean_pred, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_pred:.2f}')
            
            ax.set_xlabel("Predicted Log Solubility")
            ax.set_ylabel("Number of Models")
            ax.set_title("Histogram of the 64 Model Predictions")
            ax.legend()
            ax.grid(axis='y', alpha=0.5)
            
            st.pyplot(fig)

        except ValueError as e:
            st.error(f"Error: {e}")
        except RuntimeError as e:
            st.error(f"A model runtime error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")