import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from typing import Union, List
import streamlit as st

class Experiment:
    def __init__(self, ExperimentClass: Union[ClassificationExperiment, RegressionExperiment], sort_by: str, plots: List[str]):
        """
        Initialize the Experiment with default attributes.

        Args:
            ExperimentClass (Union[ClassificationExperiment, RegressionExperiment]): The PyCaret experiment class to use.
            sort_by (str): The metric to sort the models by.
            plots (list[str]): List of plots to generate.
        """
        self.exp = ExperimentClass()
        self.sort_by = sort_by
        self.plots = plots

    def set_experiment(self, df: pd.DataFrame, target_column: str) -> Union[ClassificationExperiment, RegressionExperiment]:
        """
        Set up the PyCaret experiment based on the selected task type.

        Args:
            df (pd.DataFrame): The input dataset.
            target_column (str): The target column for the model.

        Returns:
            Union[ClassificationExperiment, RegressionExperiment]: Configured experiment instance.
        """
        exp_setup = self.exp.setup(
            data=df,
            target=target_column,
            index=False,
            fold_shuffle=True,
            verbose=False,
            session_id=123
        )
        if not self.exp:
            st.error("Experiment setup is incomplete. Please set up the experiment first.")
            return None

        info_df = exp_setup.pull()
        st.write("Setup complete")
        st.dataframe(info_df)
        return self.exp