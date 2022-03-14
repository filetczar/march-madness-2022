import pandas as pd
from math import exp, log
import numpy as np 
from itertools import product

# need ID (Season_hTeamID_lTeamID), hTeamID, lTeamID, Season, y 
def build_labelled_data(raw_df_path = '/Users/philazar/Desktop/march-madness/data/data-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv'):
    df = pd.read_csv(raw_df_path)
    df['WTeamID'] = df['WTeamID'].astype(int)
    df['LTeamID'] = df['LTeamID'].astype(int)
    df_pos_class = df.loc[df['WTeamID'] < df['LTeamID'], :].copy()
    df_pos_class['hTeamID'] = df_pos_class['WTeamID']
    df_pos_class['aTeamID'] = df_pos_class['LTeamID'] 
    df_pos_class['Y'] = 1.0

    df_min_class = df.loc[df['WTeamID'] > df['LTeamID'], :].copy()
    df_min_class['hTeamID'] = df_min_class['LTeamID']
    df_min_class['aTeamID'] = df_min_class['WTeamID'] 
    df_min_class.loc[:, 'Y'] = 0.0
    df = pd.concat([df_min_class, df_pos_class])
    df['aTeamID'] = df['aTeamID'].astype(int)
    df['hTeamID'] =df['hTeamID'].astype(int)  
    df['ID'] = df.apply(lambda x: '%s_%s_%s' % (str(x['Season']),str(x['hTeamID']),str(x['aTeamID'])),axis=1)
    return df

def matchup_data(raw_df_path = '/Users/philazar/Desktop/march-madness/data/data-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv',
                season= 2021):
    df = pd.read_csv(raw_df_path)
    df = df.loc[df['Season'] == season][['Season', 'TeamID']]
    teams = df.TeamID 
    new_df = pd.DataFrame(list(product(teams, teams)), columns = ['TeamID1', 'TeamID2'])
    new_df = new_df.loc[new_df['TeamID1'] != new_df['TeamID2'],:].sort_values(['TeamID1'], ascending = True)
    new_df = new_df.loc[new_df['TeamID1'] < new_df['TeamID2'], :]
    new_df.rename(columns = {'TeamID1': 'hTeamID', 'TeamID2': 'aTeamID'}, inplace=True)
    new_df['Season'] = season
    new_df['ID'] = new_df.apply(lambda x: '%s_%s_%s' % (str(x['Season']),str(x['hTeamID']),str(x['aTeamID'])),axis=1)
    return new_df 


