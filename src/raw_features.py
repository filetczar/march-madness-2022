from typing_extensions import Self
import pandas as pd
import os

class RawFeatures(object):
    def __init__(self, min_year: int, max_year: int, data_dir: str):
        if min_year < 2003:
            min_year = 2003
        if max_year > 2022:
            max_year = 2022
        self.min_year = min_year
        self.max_year = max_year
        self.data_dir = data_dir 
    
    def team_df_build(raw_df_name = 'MNCAATourneyCompactResults' ):
        df = pd.read_csv(f"{Self.data_dir}/{raw_df_name}.csv")
        winning_teams = df[['Season', 'WTeamID']].loc[(df['Season'] >= Self.min_year) & (df['Season'] <= Self.max_year)]\
                            .drop_duplicates()\
                            .rename_columns({'WTeamID': 'TeamID'})
        losing_teams = df[['Season', 'LTeamID']].loc[(df['Season'] >= Self.min_year) & (df['Season'] <= Self.max_year)]\
                        .drop_duplicates()\
                        .rename_columns({'LTeamID': 'TeamID'}) 
        
        all_teams = pd.concat([winning_teams, losing_teams]).drop_duplicates()

        Self.feature_set = all_teams 
    
    def tourn_seed(raw_df_name = 'MNCAATourneySeeds'):
        df =  pd.read_csv(f"{Self.data_dir}/{raw_df_name}.csv").loc[(df['Season'] >= Self.min_year) & (df['Season'] <= Self.max_year)]
        df['tourn_seed'] = df["Seed"].apply(lambda x: int(x[1:3]))
        Self.feature_set = pd.merge(Self.feature_set, df[['TeamID','Season','tourn_seed']], on = ['Season', 'TeamID'], how='left').fillna(16)

    
    def rankings(raw_df_name = 'MMasseyOrdinals', good_rankings = ['POM', 'RPI', 'AP', 'NET', 'KPK','MAS', 'SAG','LRMC', 'USA','MOR']):
        last_day = 133
        df = pd.read_csv(f"{Self.data_dir}/{raw_df_name}.csv")\
                .loc[(df['Season'] >= Self.min_year) & (df['Season'] <= Self.max_year)]\
                .loc[df['SystemName'].isin(good_rankings)]
        current_rankings = df.loc[df['RankingDayNum'] == last_day]
        current_rank_piv = pd.pivot_table(current_rankings, index = ['Season', 'TeamID'], columns = ['SystemName'], values = ['OrdinalRank'])\
                            .reset_index(col_level=1)
        current_rank_piv.columns = current_rank_piv.columns.droplevel(0)
        current_rank_piv['avg_rank_c'] = current_rank_piv[[c for c in current_rank_piv.columns if c not in ['Season', 'TeamID']]]\
            .mean(axis=1, skipna=True)
        
        for c in good_rankings:
            current_rank_piv.loc[current_rank_piv[c].isnull(), c]= current_rank_piv['avg_rank_c']

        # Rank trends: Last 31 days, last 61 days, last 91 days, last 3 years 
        weekly_cuts = [4,8,12]
        df['rownum'] = df.groupby(['SystemName', 'Season'])['RankingDayNum'].rank(method='dense', ascending=False)
        hist_long = df.loc[df['rownum'].isin(weekly_cuts)].groupby(['Season', "TeamID", "rownum"])['OrdinalRank'].mean().reset_index()
        hist_piv = hist_long.pivot_table( index=['Season', 'TeamID'], columns = ['rownum'], values = ['OrdinalRank']).reset_index()
        hist_piv.columns = ['Season', 'TeamID', 'avg_rank_l31', 'avg_rank_l61', 'avg_rank_l91']
        
        # fill with the max (unranked)
        max_fill = current_rank_piv.avg_rank_c.max()
        all_ranks = pd.merge(current_rank_piv, hist_piv, on = ['Season', 'TeamID'], how = 'left').fillna(max_fill)
        all_ranks['rank_l31_delta'] = all_ranks['avg_rank_l31']/all_ranks['avg_rank_c']
        all_ranks['rank_l61_delta'] = all_ranks['avg_rank_l61']/all_ranks['avg_rank_c']
        all_ranks['rank_l91_delta'] = all_ranks['avg_rank_l61']/all_ranks['avg_rank_c']
        
        # pre season
        df['rownum'] = df.groupby(['SystemName', 'Season'])['RankingDayNum'].rank(method='dense', ascending=True)
        pre_long = df.loc[df['rownum'] == 1].groupby(['Season', "TeamID"])['OrdinalRank'].mean().reset_index()
        pre_long.columns = ['Season', 'TeamID', 'avg_rank_pre']
        all_ranks = pd.merge(all_ranks, pre_long, on=['Season', 'TeamID'], how='left').fillna(max_fill)
        all_ranks['avg_rank_exp'] = all_ranks['avg_rank_pre']/all_ranks['avg_rank_c']

        # yearly trend metrics? 



        

            







    


