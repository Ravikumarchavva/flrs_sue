import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from components.experiment import Experiment

class TrainModel:
    def __init__(self, exp: Experiment):
        """
        Initialize the TrainModel with default attributes.
        """
        if exp is not None:
            self.exp = exp
        else:
            st.error("Experiment is not set up properly.")

    def train_model(self, models: List[str], sort_by: str):
        """
        Train multiple machine learning models and store their metrics.

        Args:
            models (List[str]): List of model names to train.
            sort_by (str): Metric to sort the models by.
        """
        if self.exp is None:
            st.error("Experiment is not set up properly.")
            return

        st.markdown("<h3>Starting Model Training</h3>", unsafe_allow_html=True)
        combined_df = pd.DataFrame()
        model_df_placeholder = st.empty()
        progress_bar = st.progress(0)

        trained_models: Dict[str, Any] = {}
        for idx, model_name in enumerate(models, start=1):
            try:
                # Create and store the model
                model = self.exp.create_model(model_name, cross_validation=True)
                trained_models[model_name] = model

                # Retrieve and process metrics
                results = self.exp.pull()
                results.reset_index(inplace=True)
                results.rename(columns={'index': 'Fold'}, inplace=True)
                results["Fold"] = results["Fold"].astype(str)
                results["Model"] = model_name
                combined_df = pd.concat([combined_df, results]).sort_values(by=sort_by, ascending=False)
                model_df_placeholder.dataframe(combined_df[combined_df['Fold'] == 'Mean'])
            except Exception as e:
                st.error(f"Error training model '{model_name}': {e}")
            progress_bar.progress(int((idx / len(models)) * 100))

        st.markdown("<strong>Model training complete</strong>", unsafe_allow_html=True)
        # st.write("Trained models:")
        # st.write(trained_models)
        # st.write("Combined metrics:")
        # st.write(combined_df)
        st.session_state["trained_model_df"] = combined_df
        st.session_state["trained_models"] = trained_models
        return trained_models, combined_df