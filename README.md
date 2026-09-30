# Football Match Prediction

A personal project exploring statistical models for predicting football match outcomes.

The project began with a standard **Poisson model** for football scores and is being progressively extended to more sophisticated models, including **Dixon-Coles** and time-decay approaches.

The main goal is to build a framework where different football prediction models can be fitted to historical data and compared using the same evaluation methodology.

## Models

Currently implemented:

- **Poisson Model** – models home and away goals using team attack/defence parameters and home advantage.
- **Dixon-Coles Model** – extends the Poisson model by correcting for dependence between low-scoring outcomes.
- **Time-Decay Dixon-Coles** – gives greater importance to recent matches when estimating team strength.
- **Multi-Parameter Dixon-Coles** – experimental extension of the Dixon-Coles framework.

The basic Poisson specification is

The basic Poisson specification is

$$
\lambda_{ij}^{H} = \exp(\alpha_i + \beta_j + \gamma)
$$

$$
\lambda_{ij}^{A} = \exp(\alpha_j + \beta_i)
$$

where $\alpha$ represents attacking strength, $\beta$ defensive strength,
and $\gamma$ home advantage.
Parameters are estimated by minimising the negative log-likelihood of observed match results.

## Project Structure

```text
scr/
├── models/              # Statistical prediction models
├── evaluation/          # Model evaluation framework
├── getting_data/        # Data collection and cleaning
├── runCode.py           # Main script for running/comparing models
├── paths.py             # Project paths
├── utilityFunctions.py  # General utilities
├── utilityDB.py         # Database utilities
└── save_to_db.py        # Saving fitted parameters/results

create_db.py             # Creates project database
schema.sql               # Database schema
```

`runCode.py` is the main entry point. It defines the models to be tested and passes them to `ModelEvaluator`, allowing several model specifications to be evaluated using the same data and methodology.

## Current Direction

This is an ongoing project rather than a finished prediction system.

The longer-term aim is to:

1. improve model evaluation and backtesting;
2. incorporate additional team-level statistics;
3. compare model probabilities against betting-market odds;
4. investigate whether model discrepancies can identify potentially mispriced markets.

The project is also intended as a practical environment for studying statistical modelling, optimisation and probability using football data.