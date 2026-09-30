import numpy as np
from scipy.stats import poisson

class BaseFootballModel:
    def __init__(self, ordered_teams, home_advantage=True, max_goals=7):
        """
        All the data which each model will store as standard
        Parameters
        ----------
        ordered_teams: array of all the teams modelled, in alphabetical order
        home_advantage: boolean variable, should a home advantage term be included
        max_goals: maximum number of goals to consider in the prediction model
        """
        self.home_advantage = home_advantage

        self.ordered_teams = ordered_teams
        self.team_to_index = self._ordered_team_index()
        self.number_of_teams = len(self.ordered_teams)

        self.max_goals = max_goals

        # atk and def statistics storage
        self.attack = None
        self.defence = None

        # extra parameter storage
        self.home_adv_param = None
        self.rho_param = None

        # model result indicators
        self.result = None
        self.is_fitted = False

    def _ordered_team_index(self):
        """
        Gives all the ordered team names an index in a dictionary
        Returns
        -------
        Dictionary of teams with an index
        """
        return {team: i for i, team in enumerate(self.ordered_teams)}

    def _get_most_likely_outcome(self, home_prob, draw_prob, away_prob):
        """
        Provides the result of a game given probabilities
        Parameters
        ----------
        home_prob: home win probability
        draw_prob: draw probability
        away_prob: away win probability

        Returns
        -------
        A string value, of home, draw or away
        """
        probs = [home_prob, draw_prob, away_prob]
        outcomes = ["home", "draw", "away"]

        winner_index = np.argmax(probs)
        winner = outcomes[winner_index]

        return winner

    def _most_likely_scoreline(self, ft_goal_prob):
        """
        Find the highest probability in the matrix and its resp. score
        Parameters
        ----------
        ft_goal_prob: matrix of scoreline probabilities

        Returns
        -------
        most likely scoreline
        """

        max_index = np.argmax(ft_goal_prob)

        away_goals, home_goals = np.unravel_index(
            max_index,
            ft_goal_prob.shape
        )

        highest_prob = ft_goal_prob[away_goals, home_goals]

        return home_goals, away_goals, highest_prob

    def _match_outcome_summary(self, ft_goal_prob, home_team, away_team, normalise=True):
        """
        Calculates predicted match statistics
        Parameters
        ----------
        ft_goal_prob: probability matrix
        home_team: home team name
        away_team: away team name
        normalise: normalise probabilities to total probability

        Returns
        -------
        dictionary of match predictions
        """

        total_probability = ft_goal_prob.sum()

        if normalise:
            ft_goal_prob = ft_goal_prob / total_probability

        # rows = away goals, columns = home goals
        home_prob = np.triu(ft_goal_prob, k=1).sum()
        draw_prob = np.trace(ft_goal_prob)
        away_prob = np.tril(ft_goal_prob, k=-1).sum()

        winner = self._get_most_likely_outcome(
            home_prob,
            draw_prob,
            away_prob
        )

        most_likely_home_goals, most_likely_away_goals, most_likely_score_prob = (
            self._most_likely_scoreline(ft_goal_prob)
        )

        results = {
            "home_team": home_team,
            "away_team": away_team,

            "most_likely_home_goals": most_likely_home_goals,
            "most_likely_away_goals": most_likely_away_goals,
            "most_likely_score_prob": most_likely_score_prob,

            "home_probability": home_prob,
            "draw_probability": draw_prob,
            "away_probability": away_prob,

            "most_likely_outcome": winner,

            "total_probability_modelled": total_probability,

            "home_expected_goals": None,
            "away_expected_goals": None,

            "home_odds": self._prob_to_odds(home_prob),
            "draw_odds": self._prob_to_odds(draw_prob),
            "away_odds": self._prob_to_odds(away_prob)
        }

        return results

    def _full_match_goal_probabilities(self, home_lambda, away_lambda):
        """
        Calculates a matrix of probabilities for different game outcomes
        Parameters
        ----------
        home_lambda: home lambda value for the poisson distribution
        away_lambda: away lambda value for the poisson distribution

        Returns
        -------
        Matrix of probabilities
        """
        goals = np.arange(self.max_goals + 1)

        home_goal_probs = poisson.pmf(goals, mu=home_lambda)
        away_goal_probs = poisson.pmf(goals, mu=away_lambda)

        ft_goal_probs = np.outer(away_goal_probs, home_goal_probs)

        if self.rho_param is not None:
            ft_goal_probs = self._tau_adjustment(ft_goal_probs, goals, home_lambda, away_lambda)

        return ft_goal_probs

    def _tau(self, x, y, lam, mu, rho):
        """
        Extra parameters that adjusts the lower score probabilities
        Parameters
        ----------
        x: home team real goals
        y: away team real goals
        lam: home team expected goals
        mu: away team expected goals
        rho: lower score adjustment

        Returns
        -------
        Adjustment multiplier
        """
        if x == 0 and y == 0:
            return 1 - (lam * mu * rho)
        elif x == 0 and y == 1:
            return 1 + (lam * rho)
        elif x == 1 and y == 0:
            return 1 + (mu * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1

    def _tau_adjustment(self, ft_goal_probs, goals, home_lambda, away_lambda): # todo: this is inefficient
        """
        Iterates through the scores and adjusts if meets dixon_coles condition
        Parameters
        ----------
        ft_goal_probs: probability matrix of all goal probabilities
        goals: maximum number of goals to consider
        home_lambda: expected home goals scored
        away_lambda: expected away goals scored

        Returns
        -------
        probability matrix adjusted
        """
        for away_goals in goals:
            for home_goals in goals:
                tau_value = self._tau(
                    home_goals,
                    away_goals,
                    home_lambda,
                    away_lambda,
                    self.rho_param
                )

                ft_goal_probs[away_goals, home_goals] *= tau_value

        return ft_goal_probs

    def _team_checker(self, home_team, away_team):
        """
        Checks if the given teams are in the trained data.
        Used to check if given teams have atk and def stats.
        Parameters
        ----------
        home_team: home team name
        away_team: away team name

        Returns
        -------
        Raises an error if team has no atk or def stats
        """
        if home_team not in self.team_to_index:
            raise ValueError(f"Unknown home team: {home_team}")

        if away_team not in self.team_to_index:
            raise ValueError(f"Unknown away team: {away_team}")

    def _param_checker(self):
        """
        Checks if the method has attack or defence parameters loaded
        Returns
        -------
        Raises an error if no team stats are present
        """
        if self.attack is None or self.defence is None:
            raise ValueError("Model must be fitted before calling predict().")

    def _prob_to_odds(self, probability):
        """
        Given a probability, it converts it to odds that are used by bookies
        Parameters
        ----------
        probability: just a probability

        Returns
        -------
        Odds representation of the probability
        """
        odds = (1 - probability) / probability
        return odds

    def _odds_to_prob(self, odds):
        """
        Given an odds, it converts it to a probability
        Parameters
        ----------
        odds: just the odds

        Returns
        -------
        Probabilistic representation of the odds
        """
        probability = 1 / (1 + odds)
        return probability
