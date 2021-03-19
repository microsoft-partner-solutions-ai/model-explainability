import joblib
import json
import pandas as pd
from azureml.core.model import Model
from azureml.train.automl.runtime.automl_explain_utilities import automl_setup_model_explanations


def init():
    global model
    global scoring_explainer

    # Retrieve the path to the model file using the model name
    model_path = Model.get_model_path('<model_name>')
    scoring_explainer_path = Model.get_model_path('scoring_explainer')

    # Load model and explainer
    model = joblib.load(model_path)
    scoring_explainer = joblib.load(scoring_explainer_path)


def run(raw_data):
    data = pd.read_json(raw_data)

    # Make prediction
    pred = model.predict(data)

    # Get raw feature importance values
    automl_explainer_setup_obj = automl_setup_model_explanations(model, X_test=data, task='classification')
    raw_local_importance_values = scoring_explainer.explain(automl_explainer_setup_obj.X_test_transform, get_raw=True)

    # Combine explanations with feature names, reverse sorted by importance score
    num_records = data.shape[0]
    explanations = []
    for i in range(num_records):
        exp_dict = dict(zip(automl_explainer_setup_obj.raw_feature_names,raw_local_importance_values[i]))
        sorted_exp_dict = dict(sorted(exp_dict.items(), key=lambda item: item[1], reverse=True))
        explanations.append(sorted_exp_dict)

    return {"result": pred.tolist(), "explanations": explanations}
