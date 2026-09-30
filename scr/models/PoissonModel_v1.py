import sqlite3

import numpy as np
import scipy.optimize as optimize

from models.BaseFootballModel import BaseFootballModel

from utilityDB import save_global_parameter
from save_to_db import save_poisson_params
from utilityFunctions import measure_time


class PoissonModel(BaseFootballModel):
    def __init__(self, ordered_teams, home_team_var="home_team_name", away_team_var="away_team_name",
                 home_advantage=True, max_goals=7):
        """

        Parameters
        ----------
        ordered_teams: must input a list of teams in alphabetical order
        home_team_var: column name to use for home team identification
        away_team_var: column name to use for away team identification
        home_advantage: boolean variable, considering extra home advantage parameter
        max_goals: number of goals to consider when estimating probabilities
        """
        super().__init__(ordered_teams, home_advantage, max_goals)

        self.home_team_var = home_team_var
        self.away_team_var = away_team_var

        self.db_table_name = "poisson_params_v1"

    @measure_time
    def fit(self, df, save_db=False, x0=None):
        """

        Parameters
        ----------
        df: data used for fitting model
        save_db: save parameters to a database
        x0: initial start points for optimization

        Returns
        -------
        fitted model
        """
        num_params = self._get_num_params()

        if x0 is None:
            x0 = np.zeros(num_params)
        elif x0.shape[0] != num_params:
            raise ValueError(f"x0 should have length {num_params}, got {x0.shape[0]}.")

        result = optimize.minimize(
            self._negative_log_likelihood,
            x0=x0,
            args=(df,),
            method="L-BFGS-B"
        )

        self.result = result
        self.attack, self.defence, self.home_adv_param = self._unpack_params(result.x)

        # Save parameters to a database for ease of access
        if save_db:
            self._save_params()

        self.is_fitted = True

    def load_parameters(self, db_path):
        with sqlite3.connect(db_path) as connect:
            cursor = connect.cursor()

            cursor.execute(
                f"""
                SELECT team_name, attack, defence
                FROM {self.db_table_name}
                """
            )

            rows = cursor.fetchall()

            params = {}

            for team_name, attack_value, defence_value in rows:
                params[team_name] = (attack_value, defence_value)

            attack, defence = [], []

            for team_name in self.ordered_teams:
                if team_name not in params:
                    raise ValueError(f"No saved parameters found for {team_name}.")

                attack_value, defence_value = params[team_name]

                attack.append(attack_value)
                defence.append(defence_value)

            home_adv_param = 0
            if self.home_advantage: # todo: updated the database table
                cursor.execute(
                    """
                    SELECT param_value
                    FROM global_variables
                    WHERE param_name = ?
                    """,
                    ("home_advantage_poisson_v1",)
                )

                result = cursor.fetchone()

                if result is None:
                    raise ValueError("No saved home_advantage found in the database.")

                home_adv_param = result[0]

        self.attack = np.array(attack)
        self.defence = np.array(defence)
        self.home_adv_param = home_adv_param
        self.is_fitted = True

        print("Loaded Parameters Successfully.")

    def predict(self, home_team, away_team):
        self._param_checker()

        self._team_checker(home_team, away_team)

        home_index = self.team_to_index[home_team]
        away_index = self.team_to_index[away_team]

        home_lambda = np.exp(
            self.attack[home_index] + self.defence[away_index] + self.home_adv_param
        )
        away_lambda = np.exp(
            self.attack[away_index] + self.defence[home_index]
        )

        ft_goal_probs = self._full_match_goal_probabilities(home_lambda, away_lambda)

        match_results = self._match_outcome_summary(ft_goal_probs, home_team, away_team)

        match_results["home_expected_goals"] = home_lambda
        match_results["away_expected_goals"] = away_lambda

        return match_results

    def _negative_log_likelihood(self, params, df):
        total = 0

        attack, defence, home_adv = self._unpack_params(params)

        for _, row in df.iterrows():
            home_index = self.team_to_index[row[self.home_team_var]]
            away_index = self.team_to_index[row[self.away_team_var]]

            eta_home = attack[home_index] + defence[away_index] + home_adv
            eta_away = attack[away_index] + defence[home_index]

            lam = np.exp(eta_home)
            mu = np.exp(eta_away)

            x, y = row["home_team_score"], row["away_team_score"]

            add_value = (
                x * eta_home - lam +
                y * eta_away - mu
            )

            total += add_value

        return -total

    def _get_num_params(self):
        num_params = 2 * self.number_of_teams - 1

        if self.home_advantage:
            num_params += 1

        return num_params

    def _unpack_params(self, params):
        idx = 0

        # ---------------- Attack parameters ----------------
        # n-1 free parameters
        alpha_free = params[idx:idx + self.number_of_teams - 1]
        idx += self.number_of_teams - 1

        alpha_1 = -np.sum(alpha_free)
        alpha = np.insert(alpha_free, 0, alpha_1)

        # ---------------- Defence parameters ----------------
        # n free parameters
        beta = params[idx:idx + self.number_of_teams]
        idx += self.number_of_teams

        if self.home_advantage:
            gamma = params[idx]
        else:
            gamma = 0

        return alpha, beta, gamma

    def _save_params(self):
        """
        After fitting the model, save parameters.

        Returns
        -------
        nothing, just saved data in sqlite
        """
        save_global_parameter("home_advantage_poisson_v1", self.home_adv_param)
        for i in range(self.number_of_teams):
            save_poisson_params(
                self.db_table_name,
                self.ordered_teams[i],
                self.attack[i],
                self.defence[i]
            )
        print("Saved Parameters")