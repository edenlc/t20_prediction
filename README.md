# Hawkeye Coding Challenge: Predicting the Outcome of a Cricket Game

This repository contains my solution to the Hawkeye coding challenge. Specifically, it contains my code for predicting the outcome of T20 cricket matches, and my chosen extension of the task: similating a cricket match with the aim of predicting the result and order of events within it.

I have used Jupyter notebooks to document my thinking and results for this task alongside my code. They can be found in the Notebooks folder.
If you would like to run the code yourself, you can do so by running the notebooks in this order:
1. data_investigation.ipynb
2. feature_extraction.ipynb
3. models.ipynb
4. ball_modeling.ipynb
(notebooks 3. and 4. require notebooks 1. and 2. to be run first, but not eachother)

All required packages can be installed using poetry, poetry.lock and pyproject.toml for more details.
The all_matches.csv file is too large to be uploaded to github, but can be saved locally by running notebook 1.
Models/ was also excluded from the repository due to its size, but can be saved locally by running notebooks 3. and 4.

## Modeling of Cricket Match Results

I was able to achieve a RMSE of 32.5 for runs and 1.94 for wickets when using random forest regression models to predict the final innings score of a cricket team. 

For features, I calculated total, powerplay and non powerplay statistics for the batters and bowlers on a team (with some fielder-related wickets and extras being incorporated into these values). I then combined these features together to create a single set of statistics for each game, in which I used a weighted average of the statistics for each game's batters and bowlers.

For a baseline model, I trained a linear regression model on the data, achieving a baseline RMSE of 32.9 and 2.13 for runs and wickets respectively. Following this, I trained and tuned a series of random forest regression models. These were able to marginally outperform the baseline, achieving a RMSE of 32.5 and 1.94 for runs and wickets respectively.

If I were able to improve upon this further, I would use my improved knowledge of cricket gained from fully completing the challenge to further expand upon and refine my feature set, as well as considering more complex models such as other regression models or even neural networks. Based upon the results of the simpler feature set with the random forest model, I think that it is reasonable to expect that such improvements and changes could lead to better results in the future.

## Cricket Match Simulation

For my extension to the project, I decided to try and simulate a cricket match ball by ball using a series of models and distributions. When I started the coding challenge, my knowledge of cricket was very limited, and I hoped that this would be a unique way to better learn and understand the game. 

My intial set targets were approximately half that of my basline linear regression model from the original task. I was able to achieve marginally better RMSE and MAPE values, however worse R^2 values for both runs and wickets. With more time, I would further refine the simulation and improve the individual models within it, my hope being that due to the increased parameters available to the simulation, the results could be improved to where they acutally outperform the linear regression model.

# File Structure

Data : Contains all the data used in this project
|-- t20s_json : Contains all of the raw t20 json files downloaded from cricksheet
|-- saved_data : Contains all the processed data used in this project

models : Contains all the trained models and saved distributions used in this project
|-- match_models : Contains all the trained models for predicting the outcome of a cricket match
|-- simulation_dists : Contains all the saved distributions used in the cricket simulation
|-- simulation_models : Contains all the trained models used in the cricket simulation

Notebooks : Contains all the notebooks used in this project
|-- data_investigation.ipynb : Initial inspection of raw data and conversion to csv
|-- feature_extraction.ipynb : Extraction of features from the data
|-- models.ipynb : Training and tuning of models
|-- ball_modeling.ipynb : Simulation of cricket matches

simulation : Contains the code for the cricket simulation
|-- cricket_simulation.py : The cricket simulator class and methods