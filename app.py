import numpy as np
import matplotlib.pyplot as plt
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

            # Display results
            col1, col2 = st.columns([1,2])
            
            with col1:
                st.metric(label="Predicted Log Solubility", value=f"{mean_pred:.2f}")
                st.metric(label="Prediction Uncertainty", value=f"{std_pred:.2f}")

            # Histogram of predictions
            with col2:
                st.subheader("Distribution of Predictions")
                plt.rcParams.update({'font.size': 14})
                fig, ax = plt.subplots()
                
                ax.hist(all_preds, edgecolor='black', alpha=0.7)
                
                # Add a vertical line for the mean prediction
                ax.axvline(mean_pred, color='r', linestyle='--', linewidth=2, label=f'Mean')
                
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
