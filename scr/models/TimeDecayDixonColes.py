import sqlite3
import sys
from pathlib import Path

import numpy as np
import scipy.optimize as optimize
from datetime import date

from models.BaseFootballModel import BaseFootballModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilityDB import save_team_parameter, save_global_parameter
from utilityFunctions import measure_time


class TimeDecayDixonColes(BaseFootballModel):
    def __init__(self, ordered_teams, home_advantage=True, max_goals=7, current_year=None, xi_rate=0.1):
        """

        Parameters
        ----------
        ordered_teams: must input a list of teams in alphabetical order
        home_advantage: boolean variable, considering extra home advantage parameter
        max_goals: number of goals to consider when estimating probabilities
        """
        super().__init__(ordered_teams, home_advantage, max_goals)

        self.db_table_name = "time_decay_parameters"

        self.xi_rate = xi_rate

        if current_year is None:
            self.current_year =  date.today().year
        else:
            self.current_year = current_year

    @measure_time
    def fit(self, df, save_db=True, x0=None):
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

        # No bounds for normal parameters
        bounds = [(None, None)] * num_params

        # Last parameter is rho
        bounds[-1] = (-1, 1)

        result = optimize.minimize(
            self._negative_log_likelihood,
            x0=x0,
            args=(df,),
            bounds=bounds,
            method="L-BFGS-B"
        )

        self.result = result
        (self.attack, self.defence,
         self.home_adv_param, self.rho_param) = self._unpack_params(result.x)

        if save_db:
            self._save_parameters()

        self.is_fitted = True

    def load_parameters(self, db_path):
        with sqlite3.connect(db_path) as connect:
            cursor = connect.cursor()

            # load all the attack and defence parameters
            cursor.execute(
                f"""
                SELECT country, attack, defence
                FROM {self.db_table_name}
                """
            )

            rows = cursor.fetchall()

            params = {}

            for country, attack_value, defence_value in rows:
                params[country] = (attack_value, defence_value)

            attack, defence = [], []

            for country in self.ordered_teams:
                if country not in params:
                    raise ValueError(f"No saved parameters found for {country}.")

                attack_value, defence_value = params[country]

                attack.append(attack_value)
                defence.append(defence_value)

            # load rho_param from global database
            cursor.execute(
                """
                SELECT param_value
                FROM global_variables
                WHERE param_name = ?
                """,
                ("time_decay_rho_param",)
            )

            result = cursor.fetchone()

            if result is None:
                raise ValueError("No saved time_decay_rho_param found in the database.")

            rho_param = result[0]

            # load home_adv_param if it exists
            home_adv_param = 0
            if self.home_advantage:
                cursor.execute(
                    """
                    SELECT param_value
                    FROM global_variables
                    WHERE param_name = ?
                    """,
                    ("time_decay_home_advantage",)
                )

                result = cursor.fetchone()

                if result is None:
                    raise ValueError("No saved time_decay_home_advantage found in the database.")

                home_adv_param = result[0]

        self.attack = np.array(attack)
        self.defence = np.array(defence)
        self.home_adv_param = home_adv_param
        self.rho_param = rho_param
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

        attack, defence, home_adv, rho_param = self._unpack_params(params)

        for _, row in df.iterrows():
            home_index = self.team_to_index[row["home_team_name"]]
            away_index = self.team_to_index[row["away_team_name"]]

            eta_home = attack[home_index] + defence[away_index] + home_adv
            eta_away = attack[away_index] + defence[home_index]

            lam = np.exp(eta_home)
            mu = np.exp(eta_away)

            # alter lower score probability
            x, y = row["home_team_score"], row["away_team_score"]

            tau_value = self._tau(x, y, lam, mu, rho_param)

            if tau_value <= 0:
                return 1e10

            # add time decay
            year_diff = self.current_year - row["year"]
            w = np.exp(- self.xi_rate * year_diff)

            add_value = w * (
                    x * eta_home - lam +
                    y * eta_away - mu +
                    np.log(tau_value)
            )

            total += add_value

        return -total

    def _get_num_params(self):
        num_params = 2 * self.number_of_teams

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
            idx += 1
        else:
            gamma = 0

        rho = params[idx]

        return alpha, beta, gamma, rho

    def _save_parameters(self):
        """
        Save parameters to database
        """
        save_global_parameter("time_decay_home_advantage", self.home_adv_param)
        save_global_parameter("time_decay_rho_param", self.rho_param)
        for i in range(self.number_of_teams):
            save_team_parameter(
                self.ordered_teams[i],
                self.attack[i],
                self.defence[i],
                self.db_table_name
            )