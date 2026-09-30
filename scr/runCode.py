from models.PoissonModel_v1 import PoissonModel
from models.MultiParamDixonColes import MultiParamDixonColes
from models.DixonColesPoisson import DixonColesPoisson
from models.TimeDecayDixonColes import TimeDecayDixonColes
from getting_data.uniform_data_cleaning import load_football_data_1
from utilityFunctions import (load_data, ordered_teams, BASE_DIR, load_pl_data, load_football_data)

from evaluation.ModelEvaluator import ModelEvaluator


DB_PATH = BASE_DIR / "football_project.db"

# df = load_football_data_1("football-data")

df = load_football_data()

ordered_team = ordered_teams(df,"home_team_name", "away_team_name")



models = {
    "Poisson": PoissonModel,
    "DixonColes": DixonColesPoisson,
    "TimeDelayDC": TimeDecayDixonColes,
    "MultiDC": MultiParamDixonColes
}

inputs = {
    "ordered_teams": ordered_team,
    "home_advantage": True
}
eval1 = ModelEvaluator(models, inputs, df, save_params=True) # "season_id"
eval1.evaluate_models()
