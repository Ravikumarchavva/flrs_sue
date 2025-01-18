import streamlit as st
import pandas as pd

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication block
if not st.session_state.authenticated:
    password = st.text_input("Enter password", type="password")
    if st.button("Submit Password"):
        if password == "flrssue":
            st.session_state.authenticated = True
            st.success("Password correct. You can proceed.")
        else:
            st.warning("Incorrect password.")
            st.stop()
else:
    st.success("You are already authenticated.")

# Main application logic
if st.session_state.authenticated:
    # Allow the user to select either classification or regression
    task_scope = st.selectbox("Select your task type", ["Classification", "Regression"])
    if task_scope == "Classification":
        from pycaret.classification import *
        sort_by = "Accuracy"
    else:
        from pycaret.regression import *
        sort_by = "R2"

    def train_model(df, target_column):
        st.write("Initializing PyCaret setup...")
        pyc = setup(df, target=target_column, verbose=False)
        info_df = pull()
        st.dataframe(info_df)
        st.write("Setup complete")
        models = ["lr", "dt", "rf"]
        all_model_data = {}
        combined_df = pd.DataFrame()
        model_df_placeholder = st.empty()
        progress_bar = st.progress(0)
        total = len(models)
        for idx, model_name in enumerate(models, start=1):
            try:
                create_model(model_name, cross_validation=False)
                metrics_df = pull().astype(str)
                metrics_df.index = [model_name]  # Set model name as index
                all_model_data[model_name] = metrics_df
                combined_df = pd.concat([combined_df, metrics_df])
                combined_df = combined_df.sort_values(by=sort_by, ascending=False)  # Sort by Accuracy or R2
                model_df_placeholder.dataframe(combined_df)
            except Exception as e:
                st.error(f"Error creating model '{model_name}': {e}")
            progress_bar.progress(int((idx / total) * 100))
        st.write("Comparison complete:")
        st.session_state['all_model_data'] = all_model_data
        st.session_state['combined_df'] = combined_df
        st.session_state['trained'] = True

    uploaded_file = st.file_uploader("Upload a CSV file")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            data = None

        if data is not None:
            target_column = st.selectbox("Select target column", data.columns)
            columns_to_drop = st.multiselect("Select columns to drop", data.drop(columns=target_column).columns)
            if columns_to_drop:
                data.drop(columns=columns_to_drop, inplace=True)
            if st.button("Train Model"):
                train_model(data, target_column)

    if st.session_state.get('trained'):
        chosen_model = st.selectbox("View metrics for a specific model", options=list(st.session_state['all_model_data'].keys()))
        st.dataframe(st.session_state['all_model_data'][chosen_model])
