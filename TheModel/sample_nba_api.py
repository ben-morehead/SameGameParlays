import time
from nba_api.stats.static import players
from nba_api.stats.static import teams 
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playerprofilev2
from nba_api.stats.library.parameters import SeasonAll
import numpy as np
import pandas as pd

#DATA TO MOVE FROM TIME_INFO TO PLAYER INFO
#AllStar_Year (True/False)
#HEIGHT, WEIGHT, DISPLAY_FIRST_LAST, COUNTRY, FROM_YEAR, TO_YEAR
if __name__ == "_main_":
    player_amount = 500
    min_points = 0
    min_years = 0
    skip_point = 1651
    output_file_name = "player_data_minpoints{}_minyears{}.csv".format(min_points, min_years)

    if min_points == 0 and min_years == 0:
        output_file_name = "player_data_all.csv"

    player_dict = players.get_players()
    team_dict = teams.get_teams()
    additional_data = ['HEIGHT', 
                       'WEIGHT',
                       'DISPLAY_FIRST_LAST',
                       'COUNTRY',
                       'FROM_YEAR',
                       'TO_YEAR',
                       'PTS_AVG']

    full_set_time_df = []
    full_game_df_list = []
    full_game_indiv = {}
    full_set_initialized = False

    for i, player in enumerate(player_dict):
        if i < skip_point:
            continue
        points_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).player_headline_stats.get_data_frame()
        time_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id']).common_player_info.get_data_frame()
        if len(points_info["PTS"]) > 0:
            if points_info["PTS"][0] >= min_points: 
                if len(time_info["TO_YEAR"]) > 0:
                    if (int(time_info["TO_YEAR"][0]) - int(time_info["FROM_YEAR"][0])) >= min_years:

                        #PREPARING INPUT DATA
                        full_set_time_df.append(time_info)
                        game_log = playergamelog.PlayerGameLog(player_id=player['id'], season=SeasonAll.all).get_data_frames()[0]
                        try:
                            all_star_years = playerprofilev2.PlayerProfileV2(player_id=player['id']).season_totals_all_star_season.get_data_frame()['SEASON_ID'].tolist()
                        except:
                            print("{}. Skipping Player | DEBUG CODE 99 | All Star Years did not Load".format(i))
                            continue
                        finally:
                            for label in additional_data:

                                if label == 'PTS_AVG': label_data = points_info['PTS'][0]
                                else: label_data = time_info[label][0]

                                game_log[label] = label_data

                            #PREPARING LABEL DATA
                            updated_season_list = []
                            for val in all_star_years:
                                new_val = int(val.split('-')[0])
                                new_val = 20000 + new_val
                                updated_season_list.append(str(new_val))
                            game_log["ALLSTAR_YEAR"] = game_log["SEASON_ID"].isin(updated_season_list)

                            full_game_indiv[player['id']] = game_log
                            full_game_df_list.append(game_log)

                            print("{}. Adding {} to Data Set | DEBUG CODE 1".format(i, player['full_name']))
                    else:
                        print("{}. Skipping Player | DEBUG CODE 1".format(i))
                else:
                    if (int(time_info["TO_YEAR"]) - int(time_info["FROM_YEAR"])) >= min_years:
                        #PREPARING INPUT DATA
                        full_set_time_df.append(time_info)
                        game_log = playergamelog.PlayerGameLog(player_id=player['id'], season=SeasonAll.all).get_data_frames()[0]
                        try:
                            all_star_years = playerprofilev2.PlayerProfileV2(player_id=player['id']).season_totals_all_star_season.get_data_frame()['SEASON_ID'].tolist()
                        except: 
                            print("{}. Skipping Player | DEBUG CODE 99 | All Star Years did not Load".format(i))
                        finally:
                            for label in additional_data:
                                label_data = time_info[label]
                                game_log[label] = label_data

                            #PREPARING LABEL DATA
                            updated_season_list = []
                            for val in all_star_years:
                                new_val = int(val.split('-')[0])
                                new_val = 20000 + new_val
                                updated_season_list.append(str(new_val))
                            game_log["ALLSTAR_YEAR"] = game_log["SEASON_ID"].isin(updated_season_list)

                            full_game_indiv[player['id']] = game_log
                            full_game_df_list.append(game_log)

                            print("{}. Adding {} to Data Set | DEBUG CODE 2".format(i, player['full_name']))
                    else:
                        print("{}. Skipping Player | DEBUG CODE = 2".format(i))
            else:
                print("{}. Skipping Player | DEBUG CODE = 3".format(i))
        else: print("{}. Skipping Player | DEBUG CODE = 4 - NO AVG DATA".format(i))
        if i%50 == 0:
            check_point_df = full_game_df_list
            check_point_df = pd.concat(check_point_df)
            check_point_df.replace(r'^\s*$', 0, regex=True)
            check_point_df.to_csv("checkpoint_{}_dataset.csv".format(i))
            print("{}. Saved Checkpointed DataSet".format(i))
        time.sleep(2)
    full_game_df = pd.concat(full_game_df_list)
    full_game_df.replace(r'^\s*$', 0, regex=True)
    full_game_df.to_csv(output_file_name)

    print("\nScript Settings\n-------------------")
    print("Output File Name: {}".format(output_file_name))
    print("Min Points: {}".format(min_points))
    print("Min Years: {}".format(min_years))
    print("Skip Point: {}".format(skip_point))
    print("\nInternal States\n---------------------")
    print("Number of Players: {}".format(len(full_game_indiv.keys())))
    print("Length of Full Dataset: {}".format(len(full_game_df)))