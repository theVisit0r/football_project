CREATE TABLE IF NOT EXISTS team_abbreviations (
    team_short TEXT PRIMARY KEY,
    team_long TEXT
);

CREATE TABLE IF NOT EXISTS team_parameters (
    country TEXT PRIMARY KEY,
    attack REAL,
    defence REAL
);

CREATE TABLE IF NOT EXISTS poisson_params_v1 (
    team_name TEXT PRIMARY KEY,
    attack REAL,
    defence REAL
);

CREATE TABLE IF NOT EXISTS poisson_weights_v1 (
    stat_name TEXT PRIMARY KEY,
    attack_weight REAL,
    defence_weight REAL
);

CREATE TABLE IF NOT EXISTS dixon_coles_parameters (
    country TEXT PRIMARY KEY,
    home_atk REAL,
    home_def REAL,
    away_atk REAL,
    away_def REAL
);

CREATE TABLE IF NOT EXISTS dixon_coles_single_params (
    country TEXT PRIMARY KEY,
    attack REAL,
    defence REAL
);

CREATE TABLE IF NOT EXISTS time_decay_parameters (
    country TEXT PRIMARY KEY,
    attack REAL,
    defence REAL
);

CREATE TABLE IF NOT EXISTS global_variables (
    param_name TEXT PRIMARY KEY,
    param_value REAL
);

CREATE TABLE IF NOT EXISTS model_evaluation (
    model_name TEXT,
    split_id INT,
    train_group TEXT,
    test_group INT,
    accuracy REAL,
    log_loss REAL,
    brier_score REAL,
    expected_goal_error REAL,
    skipped_matches REAL,
    failed_predictions REAL,
    fitting_time REAL,
    prediction_time REAL,
    evaluation_time REAL,
    PRIMARY KEY (model_name, split_id)
);