from head_to_head import build_head_to_head
from label_data import build_labelled_data, matchup_data
from raw_features import RawFeatures
import pandas as pd 
# build raw feature sets (stats + rankings) : (2 csvs) DONE
# label data for both training (same): df DONE
# prediction data for both training (same): df DONE
# head to head for training + prediction (stats + rankings) (2 csvs) WORKING ON 
# put into one main? 
# pull into notebook for
#  1. feature selection 2. XGboost for both 3. Xgboost for combining 4. posterior (injuries + seniors)

raw_features = RawFeatures(min_year = 1985, 
                            max_year = 2022, 
                            stage=2,
                            data_dir = "/Users/philazar/Desktop/march-madness/data/data-2022/")
stats_df = raw_features.build_feature_set(type='stats', save=True)
rank_df = raw_features.build_feature_set(type='rank', save=True) 

label_df = build_labelled_data('/Users/philazar/Desktop/march-madness/data/data-2022/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
prediction_data = matchup_data(season = 2022, 
                                raw_df_path = '/Users/philazar/Desktop/march-madness/data/data-2022/MDataFiles_Stage2/MNCAATourneySeeds.csv')

columns = ['Season', 'ID', 'hTeamID', 'aTeamID', 'Y']
prediction_data['Y'] = -1 
all_games = pd.concat([label_df[columns], prediction_data[columns]])

stats_data = build_head_to_head(df = all_games, 
                                features_df = stats_df)
# if time, build a linear model prediction log points? 
stats_data['exp_score'] = (.35*stats_data['h_ppg'] + .65*stats_data['a_oppg']) - (.35*stats_data['a_ppg'] + .65*stats_data['h_oppg'])
rank_data = build_head_to_head(df = all_games, 
                                features_df = rank_df)
stats_data.to_csv("/Users/philazar/Desktop/march-madness/data/data-2022/model-dev/training/stats_data.csv", index=False)
rank_data.to_csv("/Users/philazar/Desktop/march-madness/data/data-2022/model-dev/training/rank_data.csv", index=False)

