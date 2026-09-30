import pandas as pd
import numpy as np

import sqlite3
import json
import time

import save_to_db
from paths import DB_PATH

class ModelEvaluator:
    def __init__(self, models, model_inputs, df, group_size=3,
                 _a=True, _l_l=True, _b_s=True, _e_g_e=True,
                 save_params=False):
        """

        :param models: dictionary of all the models
        :param model_inputs: globals parameters to be used for each model
        :param df: data to use for evaluation
        :param data_split_col: name of the column used to identify groupings  when training/testing
        :param group_size: grouping size for train-test sets, last value is test set, rest is training set per group
        :param _a: run accuracy metric
        :param _l_l: run log loss metric
        :param _b_s: run brier score metric
        :param _e_g_e: run expected goal error metric
        """
        self.models = models
        self.model_inputs = model_inputs

        self.df = df

        self.group_size = group_size
        self.train_test_groupings = None

        # should metrics be used?
        self._a = _a
        self._l_l = _l_l
        self._b_s = _b_s
        self._e_g_e = _e_g_e

        self.model_results = None

        self.db_name = "model_evaluation"
        self.save_params = save_params

    def evaluate_models(self, del_old_tbl=True):
        """
        Function used to evaluate multiple models with one dataset
        :param del_old_tbl: save ontop of old save or delete all previous
        :return: database with all the model metrics
        """
        self._train_test_split()

        self.model_results = []

        for split_id, groups in enumerate(self.train_test_groupings):

            train_set, test_set = self._train_test_groupings(groups)

            valid_teams = self._valid_teams(train_set)

            for model_name, model_obj in self.models.items():

                result_output = self._eval_one_model_split(model_name, model_obj, split_id,
                                                           groups, train_set, test_set, valid_teams,
                                                           self.save_params)

                self.model_results.append(result_output)

        if self.model_results is None:
            raise ValueError("There are no results to save!")
        else:
            if del_old_tbl:
                self._delete_old_save()
            self._save_model_results()

    def view_results(self):
        """

        :return: returns the last saved database
        """
        with sqlite3.connect(DB_PATH) as connect:
            query = f"""
                SELECT *
                FROM {self.db_name}
                ORDER BY accuracy DESC, log_loss ASC, brier_score ASC, model_name, split_id
                    """

            results = pd.read_sql_query(query, connect)

        return results

    # Metric Functions
    def _accuracy(self,test_set):
        """
        Calculates the percentage of correct match outcomes
        :param test_set: data used for calculating metric
        :return: accuracy metric
        """
        total = 0
        accuracy = 0

        for _, row in test_set.iterrows():
            prediction = row["predicted_result"]
            if prediction is None:
                continue

            total += 1

            if prediction["most_likely_outcome"] == row["result"]:
                accuracy += 1

        if total == 0:
            return None

        return accuracy/total

    def _log_loss(self, test_set):
        """
        Calculates the log loss metric
        :param test_set: data used for calculating metric
        :return: returns the log loss matrix
        """
        total = 0
        log_loss = 0
        skipped = 0

        for _, row in test_set.iterrows():
            prediction = row["predicted_result"]

            if prediction is None:
                continue

            prob = prediction[f"{row['result']}_probability"]

            if prob <= 1e-15:
                prob = 1e-15

            if np.isnan(prob) or np.isinf(prob):
                skipped += 1
                continue

            total += 1
            log_loss += np.log(prob)

        if total == 0:
            return None, 0

        return -log_loss / total, skipped

    def _brier_score(self, test_set):
        """
        Calculates the brier score metric
        :param test_set: data used for calculating metric
        :return: brier score
        """
        total = 0
        brier_score = 0

        possible_outcomes =["home", "draw", "away"]
        for _, row in test_set.iterrows():
            prediction = row["predicted_result"]

            if prediction is None:
                continue

            total += 1

            real_result = np.isin(possible_outcomes, row["result"]).astype(int)

            model_result_prob = [
                prediction["home_probability"],
                prediction["draw_probability"],
                prediction["away_probability"]
            ]

            brier_score += np.sum(np.square(np.subtract(real_result, model_result_prob)))

        if total == 0:
            return None

        return brier_score / total

    def _exp_goal_error(self, test_set):
        """
        Calculates the expected goal error
        :param test_set: data used for calculating metric
        :return: expected goal error
        """
        total = 0
        exp_goal_error = 0

        for _, row in test_set.iterrows():
            prediction = row["predicted_result"]

            if prediction is None:
                continue

            total += 1

            goals = [
                row["home_team_score"],
                row["away_team_score"]
                ]

            exp_goals = [
                prediction["home_expected_goals"],
                prediction["away_expected_goals"]
                ]

            exp_goal_error += np.sum(np.abs(np.subtract(goals, exp_goals)))

        if total == 0:
            return None

        return exp_goal_error / total

    # Utility Functions
    def _train_test_split(self):
        """
        Splits the years into groupings
        :return: List of grouped years, using the last year as the test set and the rest as training set
        """
        data_split = np.sort(self.df["group_id"].unique())
        n = len(data_split)
        if n < self.group_size:
            groupings =  [data_split]
        else:
            groupings = [data_split[i:i + self.group_size]
                         for i in range(n - (self.group_size-1))]

        self.train_test_groupings = groupings

    def _train_test_groupings(self, data_split):
        """

        Parameters
        ----------
        data_split: array of unique data 'splitters' (how to group data)

        Returns
        -------
        Train set and test set
        """
        train_set = self.df[self.df["group_id"].isin(data_split[:-1])]
        test_set = self.df[self.df["group_id"] == data_split[-1]].copy()

        return train_set, test_set

    def _eval_one_model_split(self, model_name, model_obj, split_id, data_split,
                              train_set, test_set, valid_teams, save_params):
        test_set["predicted_result"] = None

        model = model_obj(**self.model_inputs) # Initialise model

        fit_start = time.perf_counter() # Time the fit
        model.fit(train_set, save_db=save_params) # Fit model
        fitting_time = time.perf_counter() - fit_start

        predict_start = time.perf_counter() # Time the prediction
        test_set, skipped_matches, failed_predictions \
            = self._make_predictions(model, test_set, valid_teams) # Make predictions
        predict_time = time.perf_counter() - predict_start

        result_output = {
            "model_name": model_name,
            "split_id": split_id,
            "train_group": [int(split) for split in data_split[:-1]],
            "test_group": int(data_split[-1]),
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "expected_goal_error": None,
            "skipped_matches": skipped_matches,
            "failed_predictions": failed_predictions,
            "fitting_time": fitting_time,
            "prediction_time": predict_time,
            "evaluation_time": None
        }

        eval_start = time.perf_counter() # Time the evaluation metrics
        if self._a:
            result_output["accuracy"] = self._accuracy(test_set)

        if self._l_l:
            result_output["log_loss"], skipped_l_l = self._log_loss(test_set)
            result_output["skipped_matches"] += skipped_l_l

        if self._b_s:
            result_output["brier_score"] = self._brier_score(test_set)

        if self._e_g_e:
            result_output["expected_goal_error"] = self._exp_goal_error(test_set)

        eval_time = time.perf_counter() - eval_start
        result_output["evaluation_time"] = eval_time

        return result_output

    def _make_predictions(self, model, test_set, valid_teams):
        """

        Parameters
        ----------
        model: current fitted model
        test_set: data to be used for predictions
        valid_teams: teams which have fitted parameters

        Returns
        -------
        updated df (test_set), and skipped/failed counts
        """
        skipped_matches = 0
        failed_predictions = 0

        for idx, row in test_set.iterrows():
            if row["home_team_name"] not in valid_teams:
                skipped_matches += 1
                continue

            if row["away_team_name"] not in valid_teams:
                skipped_matches += 1
                continue

            try:
                prediction = model.predict(
                    row["home_team_name"],
                    row["away_team_name"]
                )

                test_set.at[idx, "predicted_result"] = prediction

            except Exception:
                failed_predictions += 1
                continue

        return test_set, skipped_matches, failed_predictions

    def _valid_teams(self, train_set):
        """

        :param train_set: data used to find all unique team names
        :return: unique team names in training set
        """
        valid_teams = pd.concat([
                train_set["home_team_name"],
                train_set["away_team_name"]
            ]).unique()
        return valid_teams

    # SQL Functions
    def _delete_old_save(self):
        """

        :return: Deletes the last saved database
        """
        with sqlite3.connect(DB_PATH) as connect:
            cursor = connect.cursor()

            cursor.execute(
                f"DELETE FROM {self.db_name}"
            )

    def _save_model_results(self):
        """

        :return: Saves data from evaluation into a database
        """
        with sqlite3.connect(DB_PATH) as connect:
            cursor = connect.cursor()

            query = f"""
            INSERT OR REPLACE INTO {self.db_name} (
                model_name,
                split_id,
                train_group,
                test_group,
                accuracy,
                log_loss,
                brier_score,
                expected_goal_error,
                skipped_matches,
                failed_predictions,
                fitting_time,
                prediction_time,
                evaluation_time
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            for result in self.model_results:
                cursor.execute(
                    query,
                    (
                        result["model_name"],
                        result["split_id"],
                        json.dumps(result["train_group"]),
                        result["test_group"],
                        result["accuracy"],
                        result["log_loss"],
                        result["brier_score"],
                        result["expected_goal_error"],
                        result["skipped_matches"],
                        result["failed_predictions"],
                        result["fitting_time"],
                        result["prediction_time"],
                        result["evaluation_time"]
                    )
                )
