def register():

        model_uri = f"runs:/{run_id}/model" # configurable?

        mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )