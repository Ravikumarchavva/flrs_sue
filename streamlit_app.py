import streamlit as st
import pandas as pd
from typing import List
from components.auth import Auth
from components.train import TrainModel
from components.evaluate import Evaluate
from components.experiment import Experiment
from components.inference import Inference
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment


class FlrsSueApp:
    def __init__(self):
        """
        Initialize the FlrsSueApp with component instances.
        """
        self.auth = Auth("flrssue")
        self.ExperimentClass = None
        self.experiment = None
        self.sort_by = None
        self.plots = []

    def _authenticate(self):
        """
        Authenticate the user using the Auth component.
        """
        self.auth.authenticate()

    def define_experiment(self, experiment_class: str):
        """
        Define the experiment based on the selected task type.
        """
        if experiment_class == "Classification":
            self.ExperimentClass = ClassificationExperiment
            self.sort_by = "Accuracy"
            self.plots = ["confusion_matrix", "boundary", "auc"]
        else:
            self.ExperimentClass = RegressionExperiment
            self.sort_by = "R2"
            self.plots = ["residuals", "error", "cooks", "rfe"]

    def setup_experiment(self, df: pd.DataFrame, target_column: str):
        """
        Set up the experiment using the Experiment component.
        """
        self.experiment = Experiment(self.ExperimentClass, self.sort_by, self.plots)
        experiment_instance = self.experiment.set_experiment(df, target_column)
        if experiment_instance is None:
            raise Exception("Experiment setup failed.")
        st.session_state["experiment"] = self.experiment  # Store in session state
        return experiment_instance

    def train_model(self, models: List[str]):
        """
        Train models using the TrainModel component.
        """
        self.experiment = st.session_state.get("experiment")  # Retrieve from session state
        if self.experiment is not None:
            self.train_model_component = TrainModel(self.experiment)
            trained_models, combined_df = self.train_model_component.train_model(models, self.sort_by)
            return trained_models, combined_df
        else:
            st.error("Experiment is not set up properly.")

    def evaluate_model(self, model_name: str):
        """
        Evaluate models using the Evaluate component.
        """ 
        self.experiment = st.session_state.get("experiment")  # Retrieve from session state
        self.evaluate_component = Evaluate(self.experiment, self.plots)
        self.evaluate_component.evaluate_model(model_name)

    def infer_model(self, model_name: str, input_data: pd.DataFrame):
        """
        Infer models using the Inference component.
        """
        self.experiment = st.session_state.get("experiment")
        self.inference_component = Inference(self.experiment)
        self.inference_component.infer_model(model_name, input_data)

    def main_app_logic(self):
        """
        Main logic of the Streamlit application handling user interactions.
        """
        task_scope = st.selectbox(
            "Select your task type", 
            [None, "Classification", "Regression"],  # Include None as the first option
            index=0,  # Make None the default selection
            format_func=lambda x: "Select Task Type" if x is None else x,  # Display a friendly label for None
            key="task_scope_select",
        )
        if task_scope is None:
            st.warning("Please select a task type to proceed.")
            return

        st.markdown(f"<h3>You selected: {task_scope}</h3>", unsafe_allow_html=True)
        if task_scope:
            self.define_experiment(task_scope)
        
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader")
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return

            # Select the target column
            target_column = st.selectbox(
                "Select target column", 
                [None] + list(data.columns), 
                key="target_column_select", 
                format_func=lambda x: "Select target column" if x is None else x,
            )
            if target_column is None:
                st.warning("Please select a target column to proceed.")
                return

            # Optionally drop unnecessary columns
            columns_to_drop = st.multiselect(
                "Select columns to drop", 
                data.drop(columns=[target_column]).columns, 
                key="columns_to_drop_select"
            )
            if columns_to_drop:
                data.drop(columns=columns_to_drop, inplace=True)

            # Start setup process
            if st.button("Setup Experiment", key="setup_experiment_button"):
                with st.spinner("Setting up the experiment. Please wait..."):
                    try:
                        self.experiment = self.setup_experiment(data, target_column)
                        if self.experiment is None:
                            raise Exception("Experiment setup failed in self.setup_experiment.")
                        st.success("Experiment setup complete!")
                        st.session_state["experiment"] = self.experiment
                        st.session_state["data"] = data
                        st.session_state["target_column"] = target_column
                    except Exception as e:
                        st.error(f"Error during setup: {e}")
                        return
            # Check if experiment is set up before showing models
            if "experiment" in st.session_state:
                models = ["lr", "dt", "svm", "nb"]
                if st.button("Train Model", key="train_model_button"):
                    self.trained_models, self.combined_df = self.train_model(models)
                    st.session_state["trained_models"] = self.trained_models
                    st.session_state["trained_model_df"] = self.combined_df
            if st.session_state.get("trained_model_df") is not None:

                self.trained_models = st.session_state.get("trained_models")
                self.combined_df = st.session_state.get("trained_model_df")
                model_options = self.combined_df["Model"].unique()
                selected_model = st.selectbox("Select a model for evaluation", model_options, key="model_evaluation_select")
                if selected_model:
                    self.evaluate_model(selected_model)
                if st.button("Infer Model", key="infer_model_button"):
                    data = st.session_state.get("data")
                    input_data = data.sample(1)
                    self.infer_model(selected_model, input_data)

    def run(self):
        """
        Run the Streamlit application.
        """
        self._authenticate()
        if st.session_state.get("authenticated", False):
            self.main_app_logic()

if __name__ == "__main__":
    app = FlrsSueApp()
    app.run()