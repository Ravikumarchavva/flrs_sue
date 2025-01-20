import streamlit as st
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from typing import Union
from components.experiment import Experiment

class Evaluate:
    def __init__(self):
        """
        Initialize the Evaluate with default attributes.
        """
        self.exp_setup = None
        self.experiment_class = None
        self.sort_by = None
        self.plots = []

    def evaluate_model(self, model_name: str):
        """
        Evaluate a selected model by displaying its metrics and generating relevant plots.

        Args:
            model_name (str): The name of the model to evaluate.
        """
        st.markdown(f"<h3>Evaluating Model: {model_name}</h3>", unsafe_allow_html=True)
        combined_df = st.session_state.get("trained_models", pd.DataFrame())
        model_metrics = combined_df[combined_df["Model"] == model_name]
        if model_metrics.empty:
            st.error(f"No metrics found for model '{model_name}'.")
            return
        st.dataframe(model_metrics[model_metrics["Fold"] == "Mean"])

        # Retrieve the trained model for evaluation
        trained_model = st.session_state.get("trained_model_objects", {}).get(model_name)
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
                if self.exp_setup:
                    self.exp_setup.plot_model(estimator=trained_model, plot=plot, save=False, display_format="streamlit")
                else:
                    st.error("Experiment setup not found. Please train a model first.")
            except Exception as e:
                st.error(f"Error plotting '{plot}': {e}")