import streamlit as st
import pandas as pd

# Allow the user to select either classification or regression
task_scope = st.selectbox("Select your task type", ["Classification", "Regression"])
if task_scope == "Classification":
    from pycaret.classification import *
else:
    from pycaret.regression import *

def train_model(target_column):
    st.write("Initializing PyCaret setup...")
    setup(data, target=target_column, verbose=False)
    st.write("Setup complete")
    models = ["lr", "dt", "rf", "xgboost", "lightgbm"]
    all_model_data = {}
    combined_df = pd.DataFrame()
    progress_bar = st.progress(0)
    total = len(models)
    for idx, model_name in enumerate(models, start=1):
        create_model(model_name)
        metrics_df = pull().astype(str)
        all_model_data[model_name] = metrics_df
        combined_df = pd.concat([combined_df, metrics_df])
        st.dataframe(combined_df.astype(str))
        progress_bar.progress(int((idx / total) * 100))
    st.write("Comparing all models...")
    compare_models()
    final_results = pull().astype(str)
    st.write("Comparison complete:")
    st.dataframe(final_results)
    # Allow user to pick and see any single model
    chosen_model = st.selectbox("View metrics for a specific model", options=models)
    st.dataframe(all_model_data[chosen_model])

uploaded_file = st.file_uploader("Upload a CSV file")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    target_column = st.selectbox("Select target column", data.columns)
    columns_to_drop = st.multiselect("Select columns to drop", data.drop(columns=target_column).columns)
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True)
    if st.button("Train Model"):
        train_model(target_column)