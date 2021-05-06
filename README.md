### Model Explainability
Use  model interprebility in Azure ML to explain model predictions and provide feature importances at inference time.

- [create_explanations](create_explanations.ipynb) generate and pickle the explainer object for the best model of an AutoML run. 
- [deploy_explanations](deploy_explanations.ipynb) deploy a web service with the model explanations and model predictions. Also generate the scoring script that goes with the deployment. 
