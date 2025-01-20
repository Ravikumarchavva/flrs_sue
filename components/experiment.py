import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from typing import Union
import streamlit as st

class Experiment:
    def __init__(self, experiment_class: Union[ClassificationExperiment, RegressionExperiment], sort_by: str, plots: list[str]):
        """
        Initialize the Experiment with default attributes.

        Args:
            experiment_class (Union[ClassificationExperiment, RegressionExperiment]): The PyCaret experiment class to use.
            sort_by (str): The metric to sort the models by.
            plots (list[str]): List of plots to generate.
        """
        self.exp_setup = None
        self.experiment_class = experiment_class
        self.sort_by = sort_by
        self.plots = plots

    def setup_experiment(self, df: pd.DataFrame, target_column: str) -> Union[ClassificationExperiment, RegressionExperiment]:
        """
        Set up the PyCaret experiment based on the selected task type.

        Args:
            df (pd.DataFrame): The input dataset.
            target_column (str): The target column for the model.

        Returns:
            Union[ClassificationExperiment, RegressionExperiment]: Configured experiment instance.
        """
        exp = self.experiment_class()
        self.exp_setup = exp.setup(
            data=df,
            target=target_column,
            index=False,
            fold_shuffle=True,
            verbose=False
        )
        info_df = self.exp_setup.pull()
        st.write("Setup complete")
        st.dataframe(info_df)
        return exp