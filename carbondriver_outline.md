# Understanding the `carbondriver` Module

This document breaks down the structure and workflow of the `carbondriver` Python module. The goal of this module is to use Bayesian optimization to guide experimental design for CO2 electrolysis in a gas diffusion electrode (GDE). It does this by training a model on existing experimental data and then using that model to suggest the most promising next experiment to run.

## Core Workflow Overview

The overall process can be summarized in a few key steps:

1.  **Data Loading & Preprocessing**: Experimental data is loaded from an Excel file. This data includes input parameters for the experiments (e.g., catalyst composition, ink formulation) and output measurements (e.g., Faradaic efficiencies for different products). The data is cleaned, and new features are engineered.

2.  **Model Initialization**: A `GDEOptimizer` object is created. This object is the main interface for the user. When initializing it, the user specifies which type of predictive model to use (e.g., a purely data-driven MLP, or a physics-informed model).

3.  **Training**: The `GDEOptimizer` takes existing experimental data and uses a corresponding training function to train the chosen predictive model. The models are designed to predict experimental outcomes (Faradaic efficiencies) based on input parameters. A key feature is the use of model ensembles to estimate the uncertainty of the predictions, which is essential for Bayesian optimization.

4.  **Acquisition & Suggestion**: Once trained, the model is used to drive the optimization. An "acquisition function" (Expected Improvement is used here) evaluates how promising different potential experiments are. This function balances **exploitation** (choosing experiments predicted to have high performance) and **exploration** (choosing experiments in regions of high uncertainty where a surprise discovery might be made). The module can operate in two modes:
    * **Pool-based (`step_within_data`)**: Suggest the best experiment to run next from a predefined list of candidate experiments.
    * **Unconstrained (`step`)**: Suggest a novel set of experimental parameters from a continuous search space.

5.  **Iteration**: The user would (theoretically) perform the suggested experiment, add the new data point to their dataset, and repeat the process from Step 3, progressively building a better model and converging on optimal experimental conditions.

## Component Breakdown

The module is broken into several Python files, each with a specific role.

### `__init__.py`: The Package Interface

This file acts as the main entry point for the `carbondriver` package. It imports the most important classes and functions from the other modules so they can be easily accessed by the user.

-   **`GDEOptimizer`**: The central class that orchestrates the entire optimization process.
-   **Models (`PhModel`, `MLPModel`, etc.)**: The various predictive models that the optimizer can use.
-   **Training functions (`train_model_ens`, etc.)**: The functions responsible for training the different models.

---

### `gde_multi.py`: The Physics Engine

This file contains the core physics-based model of the GDE system.

-   **`System` Class**: This class encapsulates a complex electrochemical model. It takes numerous physical constants and parameters (diffusion coefficients, reaction kinetics, etc.) as input.
-   **`solve_current` Method**: This is the key method. Given a set of physical GDE properties (e.g., catalyst layer thickness `L`, porosity `eps`, particle radius `r`) and a target current density, it solves a system of differential equations to predict the resulting Faradaic efficiencies for CO and C2H4 (`fe_co`, `fe_c2h4`) and the required voltage (`phi_ext`).

This module is essentially a standalone GDE simulator. It doesn't know about the experimental inputs like ink composition; it only understands direct physical properties.

---

### `models.py`: The Predictive Models

This file defines the machine learning models that bridge the gap between the experimental inputs and the performance outputs.

-   **`MLPModel` (Multi-Layer Perceptron)**: A standard neural network. It's a "black-box" model that directly learns the relationship between the 5 experimental inputs (like 'AgCu Ratio', 'Naf vol (ul)', etc.) and the 2 outputs ('FE (Eth)', 'FE (CO)').

-   **`PhModel` (Physics-Informed Model)**: This is a hybrid model that connects the experimental inputs to the physics engine.
    1.  It contains a neural network (`self.net`) that takes the 5 experimental inputs.
    2.  Instead of predicting the FEs directly, this network predicts a set of latent *physical parameters* (`r`, `eps`, `L`, etc.) that are the inputs for the `gde_multi.System` model.
    3.  It then calls the `solve_current` method from the `System` class using these predicted physical parameters to get the final FE predictions.
    4.  This approach embeds physical constraints into the model, potentially allowing for better predictions and extrapolation.

-   **GP Models (`MultitaskGPModel`, `MultitaskGPhysModel`)**: These are Gaussian Process models, which are a form of Bayesian modeling well-suited for capturing uncertainty. The `MultitaskGPhysModel` is also a hybrid model that uses the `PhModel` to define its mean function, essentially using the GP to learn the errors and uncertainty of the physics-informed model.

---

### `train.py`: The Training Engine

This file contains the logic for training the various models.

-   **`train_model_ens`**: This is a key function used for training the `MLPModel` and `PhModel`. It doesn't just train one model; it trains an **ensemble** of them (50 by default). Each model in the ensemble is trained on a random subset of the data (a technique called bootstrapping). When making a prediction, the final output is the average of the ensemble's predictions, and the standard deviation across the ensemble gives a robust measure of model uncertainty.

-   **`train_GP_model` & `train_GP_Ph_model`**: These functions handle the specific training procedures for the Gaussian Process models.

---

### `loaders.py`: Data Handling

This file is responsible for getting the data ready for the models.

-   **`load_data`**: Reads the `Characterization_data.xlsx` file.
    -   It performs feature engineering, most notably calculating `Zero_eps_thickness` from catalyst mass loading and density.
    -   It cleans the data (drops NaNs).
    -   It normalizes the input features, which is a standard practice for training neural networks.
    -   It returns the data as `torch.Tensor` objects, ready for use with PyTorch models.

---

### `test_api.py`: Usage Examples and Testing

This file demonstrates how the `GDEOptimizer` is intended to be used and serves as a set of integration tests.

-   **`test_gde_optimizer_within`**: This function tests the **pool-based** optimization workflow.
    1.  It initializes `GDEOptimizer`.
    2.  It loads data and splits it into a small initial training set (`df_train`) and a larger pool of unexplored candidates (`df_explore`).
    3.  It calls **`gde.step_within_data(df_train, df_explore)`**. The optimizer trains its internal model on `df_train` and then evaluates the acquisition function on every point in `df_explore`.
    4.  It returns the index (`next_pick`) of the most promising candidate in the `df_explore` pool.
    5.  The test then simulates the experimental loop by moving the chosen candidate from the "explore" set to the "train" set and repeating the process.

-   **`test_gde_optimizer_free`**: This function tests the **unconstrained** optimization workflow.
    1.  It initializes the optimizer and trains it on a dataset (`df`).
    2.  It calls **`gde.step(df)`**.
    3.  Instead of picking from a list, this method uses an optimization algorithm (`optimize_acqf` from the BoTorch library) to search the entire continuous space of input parameters.
    4.  It returns a completely new set of parameters (`next_pick`) that it believes will yield the best results.

## Flow of Functions Diagram

Here is a simplified flowchart of the `step_within_data` (pool-based) workflow:
[Start]
|
V
[Excel Data] -> loaders_old.load_data() -> [Normalized df_train, df_explore]
|
V
test_api.py calls:
GDEOptimizer.step_within_data(df_train, df_explore)
|
V
GDEOptimizer -> initold.py -> train_old.train_model_ens(X_train, y_train)
|
V
train_model_ens trains an ensemble of models (e.g., PhModel)
|
V
PhModel (in models_old.py) uses a NN to predict physical params -> calls gde_multi_old.System.solve_current()
|
V
[Trained Model Ensemble with Uncertainty]
|
V
GDEOptimizer uses the trained ensemble to calculate Expected Improvement for each point in df_explore
|
V
[Index of best next experiment] -> Returned to user/test
|
V
[End]