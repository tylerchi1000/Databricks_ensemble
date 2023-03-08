# Azure Databricks Investigation

This project investigates the use of Azure Databricks for machine learning modeling. The project includes three different regression models implemented using LightGBM, Random Forest, and XGBoost. Each model is located in its respective folder with its training files and a Conda environment file.

Basic feature engineering was done in pyspark on a databricks 11.3ml cluster. 

## Models

The following regression models were trained and implemented:

- LightGBM regression model: [Lightgbm](/Lightgbm)
- Random Forest regression model: [random_forest_regressor](/random_forest_regressor)
- XGBoost regression model: [xgboost](/xgboost)

Each model's folder contains the following:

- Trained model: [File Name](/model.pkl)
- Conda environment file: [File Name](/conda.yml)

Additionally a sample training file for the lgbm model is located at [ExampleTrainingLightgbm](/ExampleTrainingLightgbm.py)
## Usage

To use any of the models, follow the instructions below:

1. Clone the repository to your local machine.
2. Navigate to the model's folder.
3. Create the Conda environment by running the following command: `conda env create -f environment.yml`
4. Activate the Conda environment by running the following command: `conda activate <environment_name>`
5. Run the training script by running the following command: `python training.py`
6. After the training is complete, you can use the trained model for prediction.

## Contributing

Contributions to this project are welcome. To contribute, please follow the standard GitHub flow:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Create a pull request.

## License

Specify the license under which your project is distributed.
