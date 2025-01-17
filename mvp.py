import gradio as gr
from pycaret.classification import *
import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv")

# Function to train and display progress
def train_model(target_column):
    yield {"progress": "Initializing PyCaret setup..."}
    clf = setup(data, target=target_column, verbose=False)
    yield {"progress": "Setup complete"}
    models = ["lr", "dt", "rf", "xgboost", "lightgbm"]
    created_models = {}
    for model_name in models:
        created_model = create_model(model_name)
        metrics = pull()
        created_models[model_name] = metrics.to_dict(orient="list")
        yield {"progress": f"{model_name} created", "partial_results": created_models}
    yield {"progress": "Comparing all models..."}
    best_model = compare_models()
    comparison_df = pull().to_dict(orient="list")
    yield {"progress": "Comparison complete", "data": comparison_df}

# Gradio Interface
iface = gr.Interface(
    fn=train_model,
    inputs=gr.Textbox(label="Target Column"),
    outputs=gr.JSON(label="Training Progress"),
    live=True,
    description="Select the target column to train the model and view the training progress.",
)

iface.launch()
