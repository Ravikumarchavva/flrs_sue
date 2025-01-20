import streamlit as st
import pandas as pd
from components.auth import Auth
from components.train import TrainModel
from components.evaluate import Evaluate
from components.experiment import Experiment
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

class FlrsSueApp:
    def __init__(self):
        """
        Initialize the FlrsSueApp with component instances.
        """
        self.auth = Auth("flrssue")
        self.evaluate_component = Evaluate()
        self.experiment = None
        self.sort_by = None
        self.plots = []

    def authenticate(self):
        """
        Authenticate the user using the Auth component.
        """
        self.auth.authenticate()

    def define_experiment(self, experiment_class: str):
        """
        Define the experiment based on the selected task type.
        """
        if experiment_class == "Classification":
            self.experiment_class = ClassificationExperiment
            self.sort_by = "Accuracy"
            self.plots = ["confusion_matrix", "boundary", "auc"]
        else:
            self.experiment_class = RegressionExperiment
            self.sort_by = "R2"
            self.plots = ["residuals", "error", "cooks", "rfe"]

    def setup_experiment(self, df: pd.DataFrame, target_column: str):
        """
        Set up the experiment using the Experiment component.
        """
        self.experiment = Experiment(self.experiment_class, self.sort_by, self.plots)
        return self.experiment.setup_experiment(df, target_column)

    def train_model(self, models: list[str]):
        """
        Train models using the TrainModel component.
        """
        if not self.experiment:
            raise ValueError("Experiment not defined. Please select a task type first.")
        
        self.train_model_component = TrainModel(self.experiment)
        self.train_model_component.train_model(models, self.sort_by)

    def evaluate_model(self, model_name: str):
        """
        Evaluate models using the Evaluate component.
        """
        self.evaluate_component.evaluate_model(model_name)

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
                        st.success("Experiment setup complete!")
                        st.session_state["experiment"] = self.experiment
                        st.session_state["data"] = data
                        st.session_state["target_column"] = target_column
                    except Exception as e:
                        st.error(f"Error during setup: {e}")
                        return

            # Check if experiment is set up before showing models
            if "experiment" in st.session_state:
                models = ["lr", "dt", "rf", "svm", "nb"]

                if st.button("Train Model", key="train_model_button"):
                    self.train_model(models)

        if "trained_models" in st.session_state:
            st.markdown("<hr/><h3>Model Results</h3>", unsafe_allow_html=True)
            trained_models = st.session_state["trained_models"]
            model_options = trained_models["Model"].unique()

            selected_model = st.selectbox("Select a model for evaluation", model_options, key="model_evaluation_select")
            if selected_model:
                self.evaluate_model(selected_model)

    def run(self):
        """
        Run the Streamlit application.
        """
        self.authenticate()
        if st.session_state.get("authenticated", False):
            self.main_app_logic()

if __name__ == "__main__":
    app = FlrsSueApp()
    app.run()