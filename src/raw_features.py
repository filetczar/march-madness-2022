import pandas as pd
import os
from math import log 
import numpy as np

class RawFeatures(object):
    def __init__(self, min_year: int, max_year: int, data_dir: str):
        if min_year < 2003:
            min_year = 2003
        if max_year > 2022:
            max_year = 2022
        self.min_year = min_year
        self.max_year = max_year
        self.data_dir = data_dir

    def build_feature_set(self) -> pd.DataFrame: 
        self.team_df_build()
        self.tourn_seed()
        self.rankings()
        self.conference_champ()
        if 'year_trend' not in self.feature_set.columns:
            self.feature_set['year_trend'] = self.feature_set['Season'].apply(lambda x: log((x+.01) - self.min_year))
        self.reg_season_stats()
        self.opponent_stats()
        return self.feature_set
    
    def team_df_build(self, raw_df_name = 'MNCAATourneyCompactResults' ):
        df = pd.read_csv(f"{self.data_dir}/{raw_df_name}.csv")
        winning_teams = df[['Season', 'WTeamID']].loc[(df['Season'] >= self.min_year) & (df['Season'] <= self.max_year)]\
                            .drop_duplicates()\
                            .rename(columns={'WTeamID': 'TeamID'})
        losing_teams = df[['Season', 'LTeamID']].loc[(df['Season'] >= self.min_year) & (df['Season'] <= self.max_year)]\
                        .drop_duplicates()\
                        .rename(columns={'LTeamID': 'TeamID'}) 
        
        all_teams = pd.concat([winning_teams, losing_teams]).drop_duplicates()

        self.feature_set = all_teams 
    
    def tourn_seed(self, raw_df_name = 'MNCAATourneySeeds'):
        df =  pd.read_csv(f"{self.data_dir}/{raw_df_name}.csv")
        df = df.loc[(df['Season'] >= self.min_year) & (df['Season'] <= self.max_year)]
        df['tourn_seed'] = df["Seed"].apply(lambda x: int(x[1:3]))
        self.feature_set = pd.merge(self.feature_set, df[['TeamID','Season','tourn_seed']], on = ['Season', 'TeamID'], how='left').fillna(16)

    
    def rankings(self, 
                raw_df_name = 'MMasseyOrdinals', 
                good_rankings = ['POM', 'RPI', 'AP', 'NET', 'KPK','MAS', 'SAG', 'USA','MOR']):
        last_day = 133
        df = pd.read_csv(f"{self.data_dir}/{raw_df_name}.csv")
        df = df.loc[(df['Season'] >= self.min_year) & (df['Season'] <= self.max_year)]\
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

        self.feature_set = pd.merge(self.feature_set, all_ranks, on = ['Season', 'TeamID'], how='left').fillna(max_fill)
    
    def conference_champ(self, 
                        raw_df_name = 'MConferenceTourneyGames',
                        major_conf = ['big_twelve', 'pac_ten','big_ten', 'pac_twelve', 'big_east', 'acc', 'sec']):
        df = pd.read_csv(f"{self.data_dir}/{raw_df_name}.csv")
        df['champ_game'] = df.groupby(['Season', 'ConfAbbrev'])['DayNum'].rank(method='first', ascending = False)
        conference_champs = df.loc[df['champ_game'] == 1][['Season', 'WTeamID']]
        conference_champs.rename(columns = {'WTeamID': 'TeamID'}, inplace=True)
        conference_champs['conf_champ'] = 1 
        self.feature_set = pd.merge(self.feature_set, conference_champs, on = ['Season', 'TeamID'], how='left').fillna(0)

        winning_teams = df[['Season', 'ConfAbbrev', 'WTeamID']].rename(columns = {'WTeamID': 'TeamID'})
        losing_teams = df[['Season', 'ConfAbbrev', 'LTeamID']].rename(columns = {'LTeamID': 'TeamID'})  
        all_teams = pd.concat([winning_teams, losing_teams]).drop_duplicates()
        all_teams.loc[all_teams['ConfAbbrev'].isin(major_conf), 'major_conf'] = 1 
        all_teams.fillna(0, inplace=True)
        all_teams = all_teams[['Season', 'TeamID', 'major_conf']]
        self.feature_set = pd.merge(self.feature_set, all_teams, on = ['Season', 'TeamID'], how='left').fillna(0)

    def reg_season_stats(self, raw_df_name = 'MRegularSeasonDetailedResults'):
        df = pd.read_csv(f"{self.data_dir}/{raw_df_name}.csv")
        # team, season, total games 
        # total stats 
        # WFGM3,WFGA3,WFTM,WFTA,WOR,WDR,WAst,WTO,WStl,WBlk
        winning_teams_all = df.groupby(['Season', 'WTeamID'])\
                            .agg({'DayNum': ['nunique'], 
                                'WScore': ['sum'], 
                                'WFGM': ['sum'], 
                                'WFGA': ['sum'], 
                                'WFGM3': ['sum'], 
                                'WFGA3': ['sum'], 
                                'WFTM': ['sum'], 
                                'WFTA': ['sum'], 
                                'WOR': ['sum'], 
                                'WDR': ['sum'], 
                                'WAst': ['sum'], 
                                'WTO': ['sum'], 
                                'WStl': ['sum'], 
                                'WBlk': ['sum'], 
                                'WPF': ['sum']}).reset_index()
        losing_teams_all = df.groupby(['Season', 'LTeamID'])\
                            .agg({'DayNum': ['nunique'], 
                                'LScore': ['sum'], 
                                'LFGM': ['sum'], 
                                'LFGA': ['sum'], 
                                'LFGM3': ['sum'], 
                                'LFGA3': ['sum'], 
                                'LFTM': ['sum'], 
                                'LFTA': ['sum'], 
                                'LOR': ['sum'], 
                                'LDR': ['sum'], 
                                'LAst': ['sum'], 
                                'LTO': ['sum'], 
                                'LStl': ['sum'], 
                                'LBlk': ['sum'], 
                                'LPF': ['sum']}).reset_index()
        losing_teams_all.columns = ['Season', 'TeamID', 'games', 'total_score', 'fgm', 'fga', 
                                    'fgm3', 'fga3', 'ftm', 'fta', 'orb', 'drb', 'ast', 'to', 'stl', 'blk', 'fouls']
        winning_teams_all.columns = ['Season', 'TeamID', 'games', 'total_score', 'fgm', 'fga', 
                                    'fgm3', 'fga3', 'ftm', 'fta', 'orb', 'drb', 'ast', 'to', 'stl', 'blk', 'fouls']
        losing_teams_all['wins'] = 0
        winning_teams_all['wins'] = winning_teams_all['games']

        reg_stats = pd.concat([losing_teams_all, winning_teams_all])
        agg_dict = {}
        for c in reg_stats.columns:
            if c in ['TeamID', 'Season']:
                continue
            else:
                agg_dict[c] = ['sum']
        total_reg_stats = reg_stats.groupby(['Season', 'TeamID'])\
                        .agg(agg_dict).reset_index()
        total_reg_stats.columns = ['Season', 'TeamID', 'games', 'total_score', 'fgm', 'fga', 
                                    'fgm3', 'fga3', 'ftm', 'fta', 'orb', 'drb', 'ast', 'to', 'stl', 'blk','fouls', 'wins'] 

        # per game and season stats 
        total_reg_stats['win_perc'] = total_reg_stats['wins']/total_reg_stats['games']
        total_reg_stats['ppg'] = total_reg_stats['total_score']/total_reg_stats['games']
        total_reg_stats['fg_perc'] = total_reg_stats['fgm']/total_reg_stats['fga']
        total_reg_stats['fg3_perc'] = total_reg_stats['fgm3']/total_reg_stats['fga3']
        total_reg_stats['ft_perc'] = total_reg_stats['ftm']/total_reg_stats['fta']
        total_reg_stats['rb_pg'] = (total_reg_stats['orb'] + total_reg_stats['drb'])/total_reg_stats['games']
        total_reg_stats['orb_pg'] = total_reg_stats['orb']/total_reg_stats['games']
        total_reg_stats['drb_pg'] = total_reg_stats['drb']/total_reg_stats['games']
        total_reg_stats['apg'] = total_reg_stats['ast']/total_reg_stats['games']
        total_reg_stats['ast_tov'] = total_reg_stats['ast']/total_reg_stats['to']
        total_reg_stats['tov_pg'] =total_reg_stats['to']/total_reg_stats['games']
        total_reg_stats['stl_pg'] = total_reg_stats['stl']/total_reg_stats['games']
        total_reg_stats['second_chance'] = (total_reg_stats['stl'] + total_reg_stats['orb'])/total_reg_stats['to']
        total_reg_stats['p_fgm_ast'] = total_reg_stats['ast']/total_reg_stats['fgm']
        total_reg_stats['blk_pg'] = total_reg_stats['blk']/total_reg_stats['games']
        total_reg_stats['blk_to_fouls'] = total_reg_stats['blk']/total_reg_stats['fouls']
        total_reg_stats['fouls_pg'] = total_reg_stats['fouls']/total_reg_stats['games']
        total_reg_stats['tsp'] = total_reg_stats['total_score']/(2*(total_reg_stats['fga'] + .44*total_reg_stats['fta']))
        

        features_to_keep = ['Season','TeamID', 'win_perc', 'ppg', 'fg_perc', 'fg3_perc', 
                            'ft_perc', 'rb_pg', 'orb_pg', 'drb_pg', 'apg', 'ast_tov', 'tov_pg', 'stl_pg', 'second_chance', 
                            'p_fgm_ast', 'blk_pg', 'blk_to_fouls', 'fouls_pg', 'tsp']   
        reg_stats_feature_set = total_reg_stats[features_to_keep]  
        self.feature_set = pd.merge(self.feature_set, reg_stats_feature_set, on = ['Season', 'TeamID'], how='left').fillna(0)

    
    def opponent_stats(self, raw_df_name = 'MRegularSeasonCompactResults'):
        """_summary_
        oppg: opponents points per game 
        pyth_wins: Pythagoreon Wins
        p_win_l10: percent wins by a score of 10 or less 
        win_perc_l10: Last 10 games winning percentage (including conf tour games)
        log_ppg_oppg: log of points per game divided by oppg 
        Args:
            raw_df_name (str, optional): _description_. Defaults to 'MRegularSeasonDetailedResults'.
        """
        df = pd.read_csv(f"{self.data_dir}/{raw_df_name}.csv")
        conf_df = pd.read_csv(f"{self.data_dir}/MConferenceTourneyGames.csv")
        df['score_diff'] = df['WScore'] - df['LScore']
        close_games = df.loc[(df['score_diff'] <= 10) | (df['NumOT'] > 0)]
        close_wins = close_games.groupby(['Season', 'WTeamID'])['DayNum'].count().reset_index()
        close_wins.columns = ['Season', 'TeamID', 'close_wins']
        close_losses = close_games.groupby(['Season', 'LTeamID'])['DayNum'].count().reset_index() 
        close_losses.columns = ['Season', 'TeamID', 'close_losses']
        all = pd.concat([close_losses[['Season', 'TeamID']], close_wins[['Season', 'TeamID']]]).drop_duplicates()
        all = pd.merge(all, close_wins, on = ['Season', 'TeamID'], how='left').fillna(0)
        all = pd.merge(all, close_losses, on= ['Season', 'TeamID'], how='left').fillna(0)
        all['close_win_perc'] = all['close_wins']/(all['close_wins'] + all['close_losses'])
        self.feature_set = pd.merge(self.feature_set, all[['Season', 'TeamID', 'close_win_perc']], 
                                        on = ['Season', 'TeamID'], how='left').fillna(0)
        # last ten win percentage 
        cols = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
        all_games = df.sort_values(['Season', 'DayNum'], ascending = True)
        all_games['game_id'] = np.arange(len(all_games))
        # Season, TeamID, game_id, win (1/0)
        last_10_w = all_games[['Season', 'DayNum', 'game_id', 'WTeamID', 'WScore']]
        last_10_w['win'] = 1 
        last_10_w.columns = ['Season', 'DayNum', 'game_id', 'TeamID', 'points','win']
        last_10_l = all_games[['Season', 'DayNum', 'game_id', 'LTeamID', 'LScore']]
        last_10_l['win'] = 0
        last_10_l.columns = ['Season', 'DayNum', 'game_id', 'TeamID', 'points','win']
        last_10 = pd.concat([last_10_l, last_10_w])
        last_10['cut'] = last_10.groupby(['Season', 'TeamID'])['game_id'].rank(method='dense', ascending = False)
        last_10_wins = last_10.loc[last_10['cut'] <= 10].groupby(['Season', 'TeamID'])['win'].sum().reset_index()
        last_10_wins.columns = ['Season', 'TeamID', 'wins']
        last_10_wins['win_perc_l10'] = last_10_wins['wins']/10
        # merge
        self.feature_set = pd.merge(self.feature_set, last_10_wins[['Season', 'TeamID', 'win_perc_l10']], 
                                    on = ['Season', 'TeamID'], how = 'left')
        # oppg 
        total_points = last_10.groupby(['game_id'])['points'].sum().reset_index()
        total_points.columns = ['game_id', 'total_points']
        all_games_points = last_10.merge(total_points, on ='game_id', how='left').fillna(0)
        all_games_points['opp_score'] = all_games_points['total_points'] - all_games_points['points']
        op_pg = all_games_points.groupby(['Season', 'TeamID']).agg({'opp_score': ['sum'], 
                                                                    'game_id': ['nunique'], 
                                                                    'points': ['sum']}).reset_index()
        # NEED TO FIX THE COLUMN INDEXES 
        op_pg.columns = ['Season', 'TeamID', 'opp_score', 'games', 'total_points']
        op_pg['oppg'] = op_pg['opp_score']/op_pg['games']
        op_pg['pythag_wins'] = op_pg['total_points']**(11.5)/(op_pg['total_points']**(11.5) + op_pg['opp_score']**(11.5))
        fillavg = op_pg['oppg'].mean()
        fillavg_p = op_pg['pythag_wins'].mean()
        self.feature_set = pd.merge(self.feature_set, 
                                    op_pg[['Season', 'TeamID', 'oppg']], on = ['Season', 'TeamID'], how ='left').fillna(fillavg)
        self.feature_set = pd.merge(self.feature_set, 
                                    op_pg[['Season', 'TeamID', 'pythag_wins']], on = ['Season', 'TeamID'], how ='left').fillna(fillavg_p)
        self.feature_set['log_ppg_oppg'] = self.feature_set['ppg']/self.feature_set['oppg']

        




        


       

        





# testing
all_years = RawFeatures(min_year = 2003, 
                        max_year = 2021, 
                        data_dir = '/Users/philazar/Desktop/march-madness/data/data-2022/MDataFiles_Stage1')

all_data = all_years.build_feature_set()
for team_id in [1228, 1260]:
    print(all_data.loc[all_data['TeamID'] == team_id].sort_values(['Season']).head(50))
    print(all_data.loc[all_data['TeamID'] == team_id].sort_values(['Season'])[['Season', 'ppg', 'oppg', 'win_perc', 'pythag_wins']]) 




        

            







    


