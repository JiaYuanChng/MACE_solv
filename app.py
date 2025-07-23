import numpy as np
import streamlit as st
from preprocess import *

# --- Streamlit User Interface ---
st.title("Solubility Prediction")
st.markdown(
    "Enter a SMILES string to predict its **log solubility (in mols/litre)**. "
)

# User input
smiles_input = st.text_input("Enter SMILES String:", "CCO") # Default to ethanol
predict_button = st.button("Predict")

# Prediction logic
if predict_button and smiles_input:
    with st.spinner("Calculating..."):
        try:
            # Get predictions from ensemble models
            all_preds = predict_with_ensemble(smiles_input)
            
            mean_pred = np.mean(all_preds)
            std_pred = np.std(all_preds)

            # 5. Display numerical results
            st.success(f"**Predicted Log Solubility:** `{mean_pred:.2f}`")
            st.info(f"**Prediction Uncertainty:** `{std_pred:.2f}`")

            # Histogram of predictions
            st.subheader("Distribution of Predictions")
            fig, ax = plt.subplots()
            
            ax.hist(all_preds, bins=15, edgecolor='black', alpha=0.7)
            
            # Add a vertical line for the mean prediction
            ax.axvline(mean_pred, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_pred:.2f}')
            
            ax.set_xlabel("Predicted Log Solubility")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(axis='y', alpha=0.5)
            
            st.pyplot(fig)

        except ValueError as e:
            st.error(f"Error: {e}")
        except RuntimeError as e:
            st.error(f"A model runtime error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
