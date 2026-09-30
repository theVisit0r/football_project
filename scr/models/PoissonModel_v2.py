import sqlite3
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson
import scipy.optimize as optimize

from models.BaseFootballModel import BaseFootballModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilityDB import save_team_parameter, save_global_parameter
from utilityFunctions import measure_time


class PoissonModel_v2(BaseFootballModel):
    def __init__(self, ordered_teams, team_stats, col_to_use, max_goals=7):
        """`

        Parameters
        ----------
        ordered_teams: must input a list of teams in alphabetical order
        team_stats: dataframe of all team stats. Format: Each row is one team for one season,
                    with attack and defence stats labled with atk_ or def_ respectively
        home_advantage: boolean variable, considering extra home advantage parameter
        max_goals: number of goals to consider when estimating probabilities
        """
        home_advantage = 0 # Dont want to use this as doesnt make sense #todo (review maths)
        super().__init__(ordered_teams, home_advantage, max_goals)

        self.db_table_name = "poisson_weights_v1"

        self.team_stats = team_stats

        self._check_columns_input(col_to_use)
        self.col_to_use = col_to_use


    @measure_time
    def fit(self, df, save_db=False, x0=None): #todo: update save_db by making a new schema
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
        # num_params = self._get_num_params() #todo: CHANGE ME
        num_params = 14

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
        self.attack, self.defence = self._unpack_params(result.x)

        if save_db:
            self._save_parameters()

        self.is_fitted = True

    # todo: I will need to add a new schema and append paramenters to a new database
    def load_parameters(self, db_path):
        with sqlite3.connect(db_path) as connect:
            cursor = connect.cursor()

            cursor.execute(
                f"""
                SELECT stat_name, attack_weight, defence_weight
                FROM {self.db_table_name}
                """
            )

            rows = cursor.fetchall()

            params = {}

            for stat_name, attack_weight, defence_weight in rows:
                params[stat_name] = (attack_weight, defence_weight)

            attack, defence = [], []

            for stat_name in self.ordered_teams:
                if stat_name not in params:
                    raise ValueError(f"No saved parameters found for {stat_name}.")

                attack_weight, defence_weight = params[stat_name]

                attack.append(attack_weight)
                defence.append(defence_weight)


        self.attack = np.array(attack)
        self.defence = np.array(defence)
        self.is_fitted = True

        print("Loaded Parameters Successfully.")

    def predict(self, home_team, away_team):
        """

        Parameters
        ----------
        home_team: home team name (long)
        away_team: away team name (long)

        Returns
        -------
        Most likely match results using model parameters
        """
        # Sanity checks - make sure data exists
        self._param_checker()
        self._team_checker(home_team, away_team)

        # for now this will be hard coded, need to be dynamic
        use_cols = ["Gls", "Sh", "SoT", "SoT%", "G/Sh", "G/SoT", "PK%"]
        atk_cols = [f"atk_{col}" for col in use_cols]
        def_cols = [f"def_{col}" for col in use_cols]

        # reterives team stats for home and away team
        home_stats = self.team_stats.loc[
            self.team_stats["team"] == home_team
        ]
        away_stats = self.team_stats.loc[
            self.team_stats["team"] == away_team
        ]

        home_lambda = np.exp(
            np.dot(self.attack, home_stats[atk_cols].to_numpy(dtype=float).ravel()) +
            np.dot(self.defence, away_stats[def_cols].to_numpy(dtype=float).ravel())
        )

        away_lambda = np.exp(
            np.dot(self.attack, away_stats[atk_cols].to_numpy(dtype=float).ravel()) +
            np.dot(self.defence, home_stats[def_cols].to_numpy(dtype=float).ravel())
        )

        ft_goal_probs = self._full_match_goal_probabilities(home_lambda, away_lambda)

        match_results = self._match_outcome_summary(ft_goal_probs, home_team, away_team)

        match_results["home_expected_goals"] = home_lambda
        match_results["away_expected_goals"] = away_lambda

        return match_results

    def _negative_log_likelihood(self, params, df):
        """

        Parameters
        ----------
        params: input weight parameters
        df: match outcome data

        Returns
        -------
        Negative Log Likelihood value, used for optimizing weights
        """
        total = 0

        attack_weights, defence_weights = self._unpack_params(params)

        # for now this will be hard coded, need to be dynamic
        use_cols = ["Gls", "Sh", "SoT", "SoT%", "G/Sh", "G/SoT", "PK%"]
        atk_cols = [f"atk_{col}" for col in use_cols]
        def_cols = [f"def_{col}" for col in use_cols]

        for _, row in df.iterrows():

            # need a dot product
            home_stats = self.team_stats.loc[
                self.team_stats["team"] == row["home_team_name"]
            ]
            away_stats = self.team_stats.loc[
                self.team_stats["team"] == row["away_team_name"]
            ]

            eta_home = (np.dot(attack_weights,home_stats[atk_cols].to_numpy(dtype=float).ravel())
                        + np.dot(defence_weights,away_stats[def_cols].to_numpy(dtype=float).ravel())
                        )

            eta_away = (np.dot(attack_weights,away_stats[atk_cols].to_numpy(dtype=float).ravel())
                        + np.dot(defence_weights,home_stats[def_cols].to_numpy(dtype=float).ravel())
                        )

            lam = np.exp(eta_home)
            mu = np.exp(eta_away)

            x, y = row["home_team_score"], row["away_team_score"]

            add_value = (
                x * eta_home - lam +
                y * eta_away - mu
            )

            total += add_value

        return -total

    # todo: not being used right now
    def _get_num_params(self):
        """
        Calculates the number of parameters being modelled

        Returns
        -------
        Integer value, representing the number of parameters which will be estimated
        """
        num_params = 2 * self.number_of_teams - 1

        if self.home_advantage:
            num_params += 1

        return num_params

    # todo: This function will need to be adjusted for the new model
    def _unpack_params(self, params,num=7): # todo: update 7 so it is dynamic!
        # These are just weights
        attack = params[0:num]

        defence = params[num:2*num]

        return attack, defence

    def _save_parameters(self):
        """
        Save parameters to a databse
        """
        for i in range(self.number_of_teams):
            save_team_parameter(
                self.ordered_teams[i],
                self.attack[i], # attack weight
                self.defence[i], # defence weight
                self.db_table_name
            )

    def _check_columns_exist(self, col_to_use):
        """
        Checks if the given columns to be used are in the stats database
        """
        for col in col_to_use:
            if col not in self.team_stats.columns():
                raise ValueError(f"Column {col} is not in the stats database")

    def _check_columns_input(self, col_to_use):
        # Makes sure columns to use are in the stats database
        self._check_columns_exist(col_to_use)






# todo: 1) Rewrite the _unpack_params to include 15 parameters: 1 home adv, 7 atk/7 def
# todo: 2) Update _get_num_params
# todo: 3) Update the input params into the optimiser in fit