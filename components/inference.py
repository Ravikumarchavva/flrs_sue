import streamlit as st
import pandas as pd

class Inference:
    def __init__(self, experiment):
        """
        Initialize the Inference with default attributes.
        """
        self.exp = experiment

    def infer_model(self, model_name: str, input_data: pd.DataFrame):
        """
        Infer a selected model by displaying its predictions on the input data.

        Args:
            model_name (str): The name of the model to infer.
            input_data (pd.DataFrame): The input data for inference.
        """
        if not st.session_state.get("selected_model"):
            st.session_state['selected_model'] = model_name
        st.success(f"Model '{model_name}' selected for inference.")

        if not st.session_state.get("trained_models"):
            st.error("No trained models available for inference.")
            return

        if st.session_state.get("selected_model"):
            model_name = st.session_state.get("selected_model")

            # Retrieve the trained model for inference
            trained_model = st.session_state.get("trained_models", {}).get(model_name)
            if not trained_model:
                st.error(f"Model '{model_name}' not found.")
                return

            st.markdown("<h4>Model Predictions</h4>", unsafe_allow_html=True)
            predictions = self.exp.predict_model(trained_model, data=input_data)
            if len(predictions) == 1:
                st.markdown(f"<center><h1>{predictions.iloc[:, -1].values[0]}</h1></center>", unsafe_allow_html=True)
            else:
                st.write(predictions)