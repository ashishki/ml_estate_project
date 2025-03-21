# ML Estate Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![codecov](https://codecov.io/gh/<your_username>/ml_estate_project/branch/main/graph/badge.svg?token=<your_token>)](https://codecov.io/gh/<your_username>/ml_estate_project)

## Overview

This project focuses on building a machine learning model for estate (housing) price prediction. It includes data generation, exploratory data analysis (EDA), model training, evaluation, interpretation, and continuous integration/continuous deployment (CI/CD) practices.

## Table of Contents

-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
-   [Running the Project](#running-the-project)
    -   [Data Generation and EDA](#data-generation-and-eda)
    -   [Model Training and Evaluation](#model-training-and-evaluation)
    -   [Model Interpretation](#model-interpretation)
-   [Running Tests and Code Coverage](#running-tests-and-code-coverage)
-   [Docker Integration](#docker-integration)
    -   [Build the Docker Image](#build-the-docker-image)
    -   [Run the Docker Container](#run-the-docker-container)
-   [CI/CD with GitHub Actions](#cicd-with-github-actions)
-   [Future Enhancements](#future-enhancements)
-   [Contributing](#contributing)
-   [License](#license)

## Getting Started

These instructions will guide you on how to set up and run the project on your local machine.

### Prerequisites

-   Python 3.9 or higher
-   `pip` package installer
-   `virtualenv` (optional but recommended)
-   Docker (optional)

### Installation

1.  **Clone the Repository**

    ```
    git clone https://github.com/your_username/ml_estate_project.git
    cd ml_estate_project
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Running the Project

### Data Generation and EDA

You can run the EDA module to generate synthetic data and visualize it:

python -m src.eda


This will display (or save, if configured) various plots showing the data distribution and correlations.  Ensure you have the necessary plotting libraries installed (e.g., matplotlib, seaborn).

### Model Training and Evaluation

To train and evaluate the model:

python -m src.modeling


The training module will:

*   Split the data into training and test sets.
*   Optionally apply SMOTE for class balancing.
*   Tune hyperparameters using GridSearchCV.
*   Train an XGBoost classifier and print evaluation metrics (ROC-AUC and F1-score).

### Model Interpretation

To generate SHAP-based interpretation plots:

python -m src.interpretation


This module computes SHAP values for the test set and generates summary and bar plots to help explain the model's predictions.

## Running Tests and Code Coverage

To run all unit tests with `pytest` and generate an HTML coverage report:

pytest --maxfail=1 --disable-warnings -v --cov=src --cov-report=html


After the tests finish, open the `htmlcov/index.html` file in your browser to view the detailed coverage report.

## Docker Integration

A `Dockerfile` is provided to containerize the project, ensuring consistent execution across different environments.

### Build the Docker Image

docker build -t ml_estate_project .


### Run the Docker Container

The container is configured to run tests by default:


docker run --rm ml_estate_project


If you want to run a specific module (e.g., start a web service or run EDA), you can override the default `CMD`:


docker run --rm ml_estate_project python -m src.eda


## CI/CD with GitHub Actions

The project includes a GitHub Actions workflow (located in `.github/workflows/tests.yml`) that:

*   Checks out the repository.
*   Sets up Python 3.9.
*   Installs dependencies.
*   Runs tests with coverage and uploads the coverage report as an artifact.

Pushes and pull requests to the `main` branch trigger the workflow automatically.  You can view the status of the workflow in the "Actions" tab of your GitHub repository.

## Future Enhancements

*   **Web Interface:** Develop a simple web interface using Flask, FastAPI, or Streamlit to demonstrate model predictions interactively.
*   **Model Deployment:** Integrate automated deployment pipelines to deploy the model as a REST API or web service.
*   **Monitoring & Logging:** Enhance logging and monitoring for model performance in production.
*   **More Models:** Add more models to compare results

## Contributing

Contributions are welcome! Please open issues or pull requests to improve the project.  Follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
