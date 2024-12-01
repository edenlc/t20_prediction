"""
This file contains the code to simulate a cricket game ball-by-ball.

The flow of the program is as follows:
1. Initialize CricketSimulator object - this will load all of the models and distributions needed
2. Initialize teams
3. Simulate the game

Within the game simulation:
1. Choose initial batters and bowlers based on the number of runs they score/concede
2. Simulate each over:
    For each ball:
        a) Get stats for current batter and bowler
        b) Simulate ball
        c) Update runs, wickets, and striker
        d) If last wicket falls, end innings
        e) If wicket, replace batter
        f) If an odd number of runs scored, swap striker and non-striker
    Within each ball:
        a) Determine if any extras are scored
        b) Determine if a wicket fallss
        c) If no wicket falls, determine the number of runs scored
3. Return the runs, wickets, and key events

Due to time constrains due to other work, the code is somewhat messy and repetitive.
Namely, the code for powerplay and non-powerplay are almost identical, except for the column names.
With more time, I would refactor the code to place the repeated parts into a single function, using a parameter to alter the column names as needed.
Due to the slightly messy nature of it, I have included line-by-line comments in some places explaining what is happening.
"""

import pandas as pd
import numpy as np
from joblib import load
from enum import Enum
from dataclasses import dataclass

@dataclass
class GameState:
    """Current state of the game"""
    over: int
    ball: int
    runs: int
    wickets: int
    is_powerplay: bool
    is_free_hit: bool
    current_bowler: str
    current_batter: str
    non_striker: str


class CricketSimulator:
    def __init__(self):
        self.batting_team_df = None
        self.bowling_team_df = None

        self.key_events = []

        self.state = GameState(
            over=0, ball=0, runs=0, wickets=0,
            is_powerplay=True, is_free_hit=False,
            current_bowler=None, current_batter=None, non_striker=None
        )

        # Models:
        self.wicket_models = {
            'powerplay_is_wicket': load('../models/simulation_models/powerplay_wicket_model.joblib'),
            'non_powerplay_is_wicket': load('../models/simulation_models/non_powerplay_wicket_model.joblib'),
        }
        self.binary_extras_models = {
            'powerplay_is_extra': load('../models/simulation_models/powerplay_extras_binary_model.joblib'),
            'non_powerplay_is_extra': load('../models/simulation_models/non_powerplay_extras_binary_model.joblib'),
        }
        self.what_extras_models = {
            'powerplay_byes': load('../models/simulation_models/powerplay_byes_model.joblib'),
            'powerplay_legbyes': load('../models/simulation_models/powerplay_legbyes_model.joblib'),
            'powerplay_wides': load('../models/simulation_models/powerplay_wides_model.joblib'),
            'powerplay_no_balls': load('../models/simulation_models/powerplay_noballs_model.joblib'),
            'non_powerplay_byes': load('../models/simulation_models/non_powerplay_byes_model.joblib'),
            'non_powerplay_legbyes': load('../models/simulation_models/non_powerplay_legbyes_model.joblib'),
            'non_powerplay_wides': load('../models/simulation_models/non_powerplay_wides_model.joblib'),
            'non_powerplay_no_balls': load('../models/simulation_models/non_powerplay_noballs_model.joblib'),
        }

        # Distributions:
        self.extras_distributions = {
            'penalty_chance': pd.read_csv('../models/simulation_dists/penalty_distribution.csv', header=None, index_col=0).squeeze(),
            'powerplay_wides': pd.read_csv('../models/simulation_dists/powerplay_wides_probs.csv', header=None, index_col=0).squeeze(),
            'powerplay_byes': pd.read_csv('../models/simulation_dists/powerplay_byes_probs.csv', header=None, index_col=0).squeeze(),
            'powerplay_legbyes': pd.read_csv('../models/simulation_dists/powerplay_legbyes_probs.csv', header=None, index_col=0).squeeze(),
            'non_powerplay_wides': pd.read_csv('../models/simulation_dists/non_powerplay_wides_probs.csv', header=None, index_col=0).squeeze(),
            'non_powerplay_byes': pd.read_csv('../models/simulation_dists/non_powerplay_byes_probs.csv', header=None, index_col=0).squeeze(),
            'non_powerplay_legbyes': pd.read_csv('../models/simulation_dists/non_powerplay_legbyes_probs.csv', header=None, index_col=0).squeeze(),
        }

        self.runs_distribution = {
            'freehit_powerplay_better_batter': pd.read_csv('../models/simulation_dists/freehit_powerplay_better_batter_runs_dist.csv', header=None, index_col=0).squeeze(),
            'freehit_powerplay_better_bowler': pd.read_csv('../models/simulation_dists/freehit_powerplay_better_bowler_runs_dist.csv', header=None, index_col=0).squeeze(),
            'freehit_powerplay_same': pd.read_csv('../models/simulation_dists/freehit_powerplay_same_runs_dist.csv', header=None, index_col=0).squeeze(),
            'freehit_non_powerplay_better_batter': pd.read_csv('../models/simulation_dists/freehit_non_powerplay_better_batter_runs_dist.csv', header=None, index_col=0).squeeze(),
            'freehit_non_powerplay_better_bowler': pd.read_csv('../models/simulation_dists/freehit_non_powerplay_better_bowler_runs_dist.csv', header=None, index_col=0).squeeze(),
            'freehit_non_powerplay_same': pd.read_csv('../models/simulation_dists/freehit_non_powerplay_same_runs_dist.csv', header=None, index_col=0).squeeze(),

            'non_freehit_powerplay_better_batter': pd.read_csv('../models/simulation_dists/non_freehit_powerplay_better_batter_runs_dist.csv', header=None, index_col=0).squeeze(),
            'non_freehit_powerplay_better_bowler': pd.read_csv('../models/simulation_dists/non_freehit_powerplay_better_bowler_runs_dist.csv', header=None, index_col=0).squeeze(),
            'non_freehit_powerplay_same': pd.read_csv('../models/simulation_dists/non_freehit_powerplay_same_runs_dist.csv', header=None, index_col=0).squeeze(),
            'non_freehit_non_powerplay_better_batter': pd.read_csv('../models/simulation_dists/non_freehit_non_powerplay_better_batter_runs_dist.csv', header=None, index_col=0).squeeze(),
            'non_freehit_non_powerplay_better_bowler': pd.read_csv('../models/simulation_dists/non_freehit_non_powerplay_better_bowler_runs_dist.csv', header=None, index_col=0).squeeze(),
            'non_freehit_non_powerplay_same': pd.read_csv('../models/simulation_dists/non_freehit_non_powerplay_same_runs_dist.csv', header=None, index_col=0).squeeze(),
        }
    
    def init_teams(self, batting_team_df, bowling_team_df):
        self.batting_team_df = batting_team_df.copy()
        self.bowling_team_df = bowling_team_df.copy()
        self.bowling_team_df['overs_bowled'] = 0

    def simulate_ball(self, bowler_batter_stats):
        """
        Simulates a single ball in a cricket game.
        1. Determines if any extras are scored.
        2. Determines if a wicket falls
        3. If no wicket falls, determines the number of runs scored.
        """
        self.state.is_free_hit = False
        player_out = False

        if self.state.is_powerplay:
            if bowler_batter_stats['powerplay_runs_batter_mean'].values[0] >= bowler_batter_stats['powerplay_batter_runs_conceded_mean'].values[0] * 1.2:
                better = 'batter'
            elif bowler_batter_stats['powerplay_batter_runs_conceded_mean'].values[0] >= bowler_batter_stats['powerplay_runs_batter_mean'].values[0] * 1.2:
                better = 'bowler'
            else:
                better = 'same'
            # Determine extras
            is_extra_proba = self.predict_event_probability(self.binary_extras_models['powerplay_is_extra'], bowler_batter_stats, fudge_factor=0.01)
            if is_extra_proba > np.random.random():
                extras = self.determine_extras(bowler_batter_stats)
            else:
                extras = 0

            # Determine wicket
            if self.state.is_free_hit:
                bowler_batter_stats.insert(0, 'freehit', 1)
            else:
                bowler_batter_stats.insert(0, 'freehit', 0)
            
            wicket_probability = self.predict_event_probability(self.wicket_models['powerplay_is_wicket'], bowler_batter_stats, fudge_factor=0.01)
            if wicket_probability > np.random.random():
                player_out = True
                runs = 0
            else:
                # Determine runs
                if self.state.is_free_hit:
                    if better == 'batter':
                        runs = self.sample_from_distribution(self.runs_distribution['freehit_powerplay_better_batter'])
                    elif better == 'bowler':
                        runs = self.sample_from_distribution(self.runs_distribution['freehit_powerplay_better_bowler'])
                    else:
                        runs = self.sample_from_distribution(self.runs_distribution['freehit_powerplay_same'])
                else:
                    if better == 'batter':
                        runs = self.sample_from_distribution(self.runs_distribution['non_freehit_powerplay_better_batter'])
                    elif better == 'bowler':
                        runs = self.sample_from_distribution(self.runs_distribution['non_freehit_powerplay_better_bowler'])
                    else:
                        runs = self.sample_from_distribution(self.runs_distribution['non_freehit_powerplay_same'])

        else:
            if bowler_batter_stats['non_powerplay_runs_batter_mean'].values[0] >= bowler_batter_stats['non_powerplay_batter_runs_conceded_mean'].values[0] * 1.2:
                better = 'batter'
            elif bowler_batter_stats['non_powerplay_batter_runs_conceded_mean'].values[0] >= bowler_batter_stats['non_powerplay_runs_batter_mean'].values[0] * 1.2:
                better = 'bowler'
            else:
                better = 'same'
            # Determine extras
            is_extra_proba = self.predict_event_probability(self.binary_extras_models['non_powerplay_is_extra'], bowler_batter_stats, fudge_factor=0.01)
            if is_extra_proba > np.random.random():
                extras = self.determine_extras(bowler_batter_stats)
            else:
                extras = 0

            # Determine wicket
            if self.state.is_free_hit:
                bowler_batter_stats.insert(0, 'freehit', 1)
            else:
                bowler_batter_stats.insert(0, 'freehit', 0)
            # Determine wicket
            wicket_probability = self.predict_event_probability(self.wicket_models['non_powerplay_is_wicket'], bowler_batter_stats, fudge_factor=0.01)
            if wicket_probability > np.random.random():
                player_out = True
                runs = 0
            else:
                # Determine runs
                if self.state.is_free_hit:
                    if better == 'batter':
                        runs = self.sample_from_distribution(self.runs_distribution['freehit_non_powerplay_better_batter'])
                    elif better == 'bowler':
                        runs = self.sample_from_distribution(self.runs_distribution['freehit_non_powerplay_better_bowler'])
                    else:
                        runs = self.sample_from_distribution(self.runs_distribution['freehit_non_powerplay_same'])
                else:
                    if better == 'batter':
                        runs = self.sample_from_distribution(self.runs_distribution['non_freehit_non_powerplay_better_batter'])
                    elif better == 'bowler':
                        runs = self.sample_from_distribution(self.runs_distribution['non_freehit_non_powerplay_better_bowler'])
                    else:
                        runs = self.sample_from_distribution(self.runs_distribution['non_freehit_non_powerplay_same'])
        return {
            'extras': extras,
            'wicket': player_out,
            'runs': runs,
        }

    
    def determine_extras(self, bowler_batter_stats):
        extras_runs = 0

        if self.state.is_powerplay:
            is_wide_proba = self.predict_event_probability(self.what_extras_models['powerplay_wides'], bowler_batter_stats, fudge_factor=0)
            if is_wide_proba > np.random.random():
                extras_runs += self.sample_from_distribution(self.extras_distributions['powerplay_wides'])
            is_byes_proba = self.predict_event_probability(self.what_extras_models['powerplay_byes'], bowler_batter_stats, fudge_factor=0)
            if is_byes_proba > np.random.random():
                extras_runs += self.sample_from_distribution(self.extras_distributions['powerplay_byes'])
            is_legbyes_proba = self.predict_event_probability(self.what_extras_models['powerplay_legbyes'], bowler_batter_stats, fudge_factor=0)
            if is_legbyes_proba > np.random.random():
                extras_runs += self.sample_from_distribution(self.extras_distributions['powerplay_legbyes'])
            is_no_balls_proba = self.predict_event_probability(self.what_extras_models['powerplay_no_balls'], bowler_batter_stats, fudge_factor=0)
            if is_no_balls_proba > np.random.random():
                extras_runs += 1
                self.state.is_free_hit = True
            penalty_chance = self.sample_from_distribution(self.extras_distributions['penalty_chance'])
            if penalty_chance == 1:
                extras_runs += 5


        else:
            is_wide_proba = self.extras_distributions['non_powerplay_wides'].iloc[bowler_batter_stats]
            if is_wide_proba > np.random.random():
                extras_runs += self.sample_from_distribution(self.extras_distributions['non_powerplay_wides'])
            is_byes_proba = self.extras_distributions['non_powerplay_byes'].iloc[bowler_batter_stats]
            if is_byes_proba > np.random.random():
                extras_runs += self.sample_from_distribution(self.extras_distributions['non_powerplay_byes'])
            is_legbyes_proba = self.extras_distributions['non_powerplay_legbyes'].iloc[bowler_batter_stats]
            if is_legbyes_proba > np.random.random():
                extras_runs += self.sample_from_distribution(self.extras_distributions['non_powerplay_legbyes'])
            is_no_balls_proba = self.extras_distributions['non_powerplay_no_balls'].iloc[bowler_batter_stats]
            if is_no_balls_proba > np.random.random():
                extras_runs += 1
                self.state.is_free_hit = True
            penalty_chance = self.sample_from_distribution(self.extras_distributions['penalty_chance'])
            if penalty_chance == 1:
                extras_runs += 5
        return extras_runs

    def simulate_over(self):
        while self.state.ball < 6:
            self.state.ball += 1
            # Get stats for current batter and bowler
            try:
                current_bowler_stats = self.bowling_team_df.loc[self.state.current_bowler]
                current_batter_stats = self.batting_team_df.loc[self.state.current_batter]

                batter_bowler_stats = pd.concat([current_batter_stats, current_bowler_stats], axis=0).to_frame().T

                if self.state.is_powerplay:
                    batter_bowler_stats = batter_bowler_stats[[col for col in batter_bowler_stats.columns if 'powerplay' in col[:9]]]
                else:
                    batter_bowler_stats = batter_bowler_stats[[col for col in batter_bowler_stats.columns if 'non_powerplay' in col[:13]]]
            except KeyError as e:
                print(f"KeyError: {e}")
                print(f"Over: {self.state.over}, Ball: {self.state.ball}")
                raise e

            # Simulate ball
            try:
                ball_result = self.simulate_ball(batter_bowler_stats)
            except Exception as e:
                print(f"Error simulating ball: {e}")
                print(f"Over: {self.state.over}, Ball: {self.state.ball}")
                raise e
            self.key_events.append(f"Over {self.state.over} ball {self.state.ball}: {ball_result['runs']} runs, {ball_result['extras']} extras")

            # Update Runs made
            self.state.runs += ball_result['runs'] + ball_result['extras']
            # If wicket we want to replace the batter with the next-best batter
            if ball_result['wicket']:
                self.state.wickets += 1
                self.key_events.append(f"{self.state.current_batter} is out in over {self.state.over} ball {self.state.ball}")
                if self.state.wickets == 10:
                    return
                # Remove current batter from batting lineup
                self.batting_team_df = self.batting_team_df.drop(self.state.current_batter)
                # Replace with next-best batter depending on whether powerplay or not
                if self.state.is_powerplay:
                    next_batter = self.batting_team_df.sort_values(by='powerplay_runs_batter_mean', ascending=False).index[0]
                else:
                    next_batter = self.batting_team_df.sort_values(by='non_powerplay_runs_batter_mean', ascending=False).index[0]
                self.state.current_batter = next_batter
            # Swap batter and non-striker if odd number of runs
            else:
                if ball_result['runs'] % 2 == 1:
                    self.state.current_batter, self.state.non_striker = self.state.non_striker, self.state.current_batter
        return


    def simulate_innings(self):
        while self.state.over < 20:
            # Init over
            self.state.over += 1
            self.state.ball = 0
            if self.state.over <= 6:
                self.state.is_powerplay = True
            else:
                self.state.is_powerplay = False

            # Simulate over
            self.simulate_over()

            # Process end of over events
            if self.state.wickets == 10:
                return
            
            # Update bowling team's overs bowled
            self.bowling_team_df.loc[self.state.current_bowler, 'overs_bowled'] += 1
            available_bowlers = self.bowling_team_df[(self.bowling_team_df['overs_bowled'] < 4) &
                                                     (self.bowling_team_df.index != self.state.current_bowler)]
            if self.state.is_powerplay:
                next_bowler = available_bowlers.sort_values(by='powerplay_batter_runs_conceded_mean', ascending=True).index[0]
            else:
                next_bowler = available_bowlers.sort_values(by='non_powerplay_batter_runs_conceded_mean', ascending=True).index[0]
            self.state.current_bowler = next_bowler

            # Swap batter and non-striker
            self.state.current_batter, self.state.non_striker = self.state.non_striker, self.state.current_batter
            
    def run_simulation(self):
        self.state.current_batter = self.batting_team_df.sort_values(by='powerplay_runs_batter_mean', ascending=False).index[0]
        self.state.non_striker = self.batting_team_df.sort_values(by='powerplay_runs_batter_mean', ascending=False).index[0]
        self.state.current_bowler = self.bowling_team_df.sort_values(by='powerplay_batter_runs_conceded_mean', ascending=True).index[0]
        
        self.simulate_innings()

        return {
            'runs': self.state.runs,
            'wickets': self.state.wickets,
            'key_events': self.key_events,
        }



    ######### Helper functions ##########################

    def sample_from_distribution(self, distribution):
        return np.random.choice(
            a=distribution.index,      # possible values
            p=distribution.values,     # probabilities
            size=1                  # number of samples
        )
    
    def predict_event_probability(self, model, X, fudge_factor=0.01):
        event_prob = model.predict_proba(X)[:, 1]
        event_prop = event_prob * (1 - fudge_factor) + fudge_factor
        return event_prop
