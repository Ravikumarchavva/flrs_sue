import streamlit as st
import pandas as pd

class Evaluate:
    def __init__(self, experiment, plots: list[str]):
        """
        Initialize the Evaluate with default attributes.
        """
        self.exp = experiment
        self.plots = plots
        
    def evaluate_model(self, model_name: str):
        """
        Evaluate a selected model by displaying its metrics and generating relevant plots.

        Args:
            model_name (str): The name of the model to evaluate.
        """
        combined_df = st.session_state.get("trained_model_df", pd.DataFrame())
        model_metrics = combined_df[combined_df["Model"] == model_name]
        if model_metrics.empty: 
            st.error(f"No metrics found for model '{model_name}'.")
            return
        st.dataframe(model_metrics[model_metrics["Fold"] == "Mean"])
        
        if st.button(label="Select Model", key="select_model"):
            st.session_state['selected_model'] = model_name
            st.success(f"Model '{model_name}' selected for evaluation.")

        if st.session_state.get("selected_model"):
            model_name = st.session_state.get("selected_model")

            # Check if trained models exist
            if not st.session_state.get("trained_models"):
                st.error("No trained models found. Train a model first.")
                return

            # Retrieve the trained model for evaluation
            trained_model = st.session_state.get("trained_models", {}).get(model_name)
            if not trained_model:
                st.error(f"Model '{model_name}' not found.")
                return

            st.markdown("<h4>Model Parameters</h4>", unsafe_allow_html=True)
            st.write(trained_model)

            st.markdown("<h4>Model Plots</h4>", unsafe_allow_html=True)
            for plot in self.plots:
                try:
                    st.write(f"Plotting '{plot}'")
                    # Ensure exp_setup is correctly initialized for plot_model
                    if self.exp:
                        self.exp.plot_model(estimator=trained_model, plot=plot, save=False, display_format="streamlit")
                    else:
                        st.error("Experiment setup not found. Please train a model first.")
                except Exception as e:
                    st.error(f"Error plotting '{plot}': {e}")