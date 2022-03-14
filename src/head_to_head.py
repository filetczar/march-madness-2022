import pandas as pd 
from math import log, exp
import numpy as np 

# data sets needed
# training: 1. stats 2. rankings 
# prediction: 1. stats 2. rankings 
# Note add in feature: expected score here 
def build_head_to_head(df: pd.DataFrame, 
                        features_df: pd.DataFrame = None):
    
    feature_names = [c for c in features_df.columns if c not in ['Season', 'TeamID', 'year_trend']]
    rename_dict = {}
    for f in feature_names:
        rename_dict[f] = f"h{f}"
    games_df = pd.merge(df, features_df, left_on = ['Season', 'hTeamID'], right_on = ['Season', 'TeamID'], how = 'left')
    games_df.rename(columns = rename_dict, inplace=True)
    rename_dict = {}
    for f in feature_names:
        rename_dict[f] = f"a{f}" 
    games_df = pd.merge(df, features_df, left_on = ['Season', 'aTeamID'], right_on = ['Season', 'TeamID'], how = 'left')
    games_df.rename(columns = rename_dict, inplace=True)
    
    # difference features 
    for feat in feature_names:
        games_df[f"{feat}_diff"] = games_df[f"h{feat}"] - games_df[f"a{feat}"] 
        try:
            games_df[f"{feat}_ratio"] = games_df[f"h{feat}"]/games_df[f"a{feat}"] 
        except ZeroDivisionError:
            games_df[f"{feat}_ratio"] = games_df[f"h{feat}"]

    return games_df 
 
