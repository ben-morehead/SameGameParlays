import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random
import time
from datetime import date
import sqlite3
import logging

from nba_api.stats.library.parameters import Season

from nba_api.stats.static import players
from nba_api.stats.static import teams 
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.live.nba.endpoints import boxscore



import sys
import os
from PIL import Image

class Helper():
    def __init__(self):
        pass


class DataFormatter():
    def __init__(self):
        #Note: For RNN If the data formatting doesn't work out, we can make a self.max_size = 30 as a safety.

        self.today = date.today().strftime("%d/%m/%y")
        self.game_path = "player_games.json"
        self.db_table_name = "gamedetails"
        self.ommitted_columns = ["index", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_NAME", "NICKNAME", "START_POSITION", "COMMENT", "MIN"]
        self.setup_logger()
        self.helper = Helper()
        self.player_games = self.load_game_json()
        #DELETE PRINT

    """EXETERNAL FILE INTERACTION : ADMIN STUFF MOSTLY"""
    
    def setup_logger(self):
        logging.basicConfig(filename='dataformatter.log', format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S', filemode='w', level=logging.INFO)

    def load_game_json(self):
        with open(self.game_path) as fp:
            ret = json.load(fp)
        fp.close()
        return ret

    def save_game_json(self):
        self.player_games["last_updated"] = self.today
        with open(self.game_path, "w+") as fp:
            json.dump(self.player_games,fp)
        fp.close()

    """THE FUNCTIONAL CODE"""

    def check_player_in_cache(self, player_id):
        if player_id not in self.player_games.keys():
            self.player_games[str(player_id)] = self.get_player_history(player_id)

    def get_box_score(self, game_id):
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).player_stats.get_data_frame()
        updated_box_score = box_score.dropna(subset=["MIN"]).reset_index().drop(self.ommitted_columns, axis=1).rename(columns={'TO': 'TOV'})
        return updated_box_score

    def get_game_from_db(self, cursor, game_id):
        get_cmd = f"SELECT * FROM {self.db_table_name} WHERE GAME_ID={game_id};"
        cursor.execute(get_cmd)
        select_res = cursor.fetchall()
        full_array = np.delete(np.array(select_res), obj=0, axis=1)
        return full_array

    def get_player_datapoint(self, player_id, num_games, most_recent=False):
        #Getting player list of all the games
        self.check_player_in_cache(player_id)

        game_list = self.player_games[str(player_id)]
        game_list = list(map(int, self.player_games[str(player_id)]))
        label = None

        connection = sqlite3.connect('gamedata.db')
        cursor = connection.cursor()

        #Checking to ensure there are enough games to make a datapoint
        if len(game_list) < (num_games):
            print("Sequence Length too Large")
            logging.info("- get_player_datapoint - Sequence Length too Large")

        #Determine whether grabbing random set of games or just the most recent(depending on if we want to sample currently or use dataformatting)
        if most_recent is True:
            print("Most Recent {}".format(num_games))
            #index = 1 #IF GAME OF DAY IS GIVING ISSUE
            index = 0 #OTHERWISE
        else:
            print("Random {}".format(num_games))
            index=random.randint(0, len(game_list) - num_games + 1)
            label_game = game_list[index-1]
            cursor.execute(f"SELECT PTS FROM {self.db_table_name} WHERE GAME_ID={label_game} AND PLAYER_ID={player_id}")
            points = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM (SELECT DISTINCT TEAM_ID FROM {self.db_table_name} WHERE GAME_ID={label_game}) WHERE NOT TEAM_ID=(SELECT TEAM_ID FROM {self.db_table_name} WHERE GAME_ID={label_game} AND PLAYER_ID={player_id})")
            opposing_team = cursor.fetchone()[0]
            label_index, _ = self.points_to_label(points)
        
        #Prepare the game list for getting db data
        game_sublist = game_list[index:(index + num_games)]
        game_sublist.reverse()


        #Get database data and 0-pad to ensure stackability
        full_seq = []
        max_size = 0
        for game_id in game_sublist:
            game_arr = self.get_game_from_db(cursor, game_id)
            full_seq.append(game_arr)
            if game_arr.shape[0] > max_size: max_size = game_arr.shape[0]

        for i,score in enumerate(full_seq):
            rows_to_add = max_size - score.shape[0]
            if rows_to_add > 0:
                zero_arr = np.zeros([rows_to_add, score.shape[1]])
                full_seq[i] = np.append(score, zero_arr, axis=0)

        #Reshaping for datapoint purposes
        datapoint = np.stack(full_seq)
        datapoint = np.reshape(datapoint, [datapoint.shape[0], datapoint.shape[1] * datapoint.shape[2]])

        cursor.close()
        connection.close()

        #Return Statement
        if most_recent: return datapoint
        else: return datapoint, opposing_team, label_index
    
    def get_player_history(self, player_id):
        available_seasons = list(commonplayerinfo.CommonPlayerInfo(player_id=player_id).available_seasons.get_data_frame()['SEASON_ID'])
        first_year = int(available_seasons[0][1:])
        last_year = int(available_seasons[-1][1:])
        yearz = range(last_year+1, first_year, -1)
        
        game_id_list = []

        for year in yearz:
            next_one = str((year + 1) % 100).zfill(2)
            season_id = "{}-{}".format(year,next_one)
            player_game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season_id).get_data_frames()[0]
            game_id_list.extend(list(player_game_log["Game_ID"]))
            time.sleep(0.5)

        ret = game_id_list
        return ret

    def get_player_id(self, player_name):
        player_id = players.find_players_by_full_name(player_name)[0]["id"]
        return player_id
    
    def get_player_name(self, player_id):
        player_name = players.find_player_by_id(player_id)["full_name"]
        return player_name
    
    #**************************
    
    def get_team_datapoint(self, team_id, num_games, most_recent=False):
        logging.info(f"Getting datapoint for last {num_games} games of: {self.get_team_name(team_id)} ({team_id})")
        print(f"Getting datapoint for last {num_games} games of: {self.get_team_name(team_id)} ({team_id})")
        return team_id

    #*****************************

    def get_team_id(self, team_name):
        team_id = teams.find_teams_by_full_name(team_name)[0]["id"]#players.find_players_by_full_name(player_name)[0]["id"]
        teams.find_team
        return team_id
    
    def get_team_name(self, team_id):
        team_name = teams.find_team_name_by_id(team_id)["full_name"]#players.find_player_by_id(player_id)["full_name"]
        return team_name

    def get_todays_games(self, game_id):
        #Will involve grabbing from the odds import
        pass

    def inititalize_database(self, player_id):
        #THIS WILL BE DEPRECATED AFTER I GET THIS WHOLE SQL THING PROPER
        logging.info("Running DataFormatter.initialize_database()")
        sample_gameid = self.get_player_history(player_id)[0]
        sample_dataframe = self.get_box_score(sample_gameid)
        columns = list(sample_dataframe.columns)
        print(columns)

        col_datatypes = []
        for i, col in enumerate(columns):
            if i < 3:
                col_datatypes.append('int')
            else:
                col_datatypes.append('float')

        #Initialize Database
        con = sqlite3.connect('gamedata.db')
        
        #Create the table
        createStatement = f'CREATE TABLE IF NOT EXISTS {self.db_table_name} ('
        for i in range(len(columns)):
            createStatement = f'{createStatement}\n{columns[i]} {col_datatypes[i]},'
        createStatement = createStatement[:-1] + ' );'
        

        #Do initial insertion (Devin Booker)
        cur = con.cursor()
        cur.execute(createStatement)
        con.commit()

        cur.close()
        con.close()
        return createStatement

    def insert_into_db(self, cursor, box_score):
        ''' 
        Inserts provided box_score into db indicated by cursor. If it already is in there function returns 0, otherwise if it is added succesfull function returns 1
        '''
        # Getting string values
        field_string = ""
        for val in box_score.columns:
            field_string=f"{field_string}'{val}', "
        field_string = field_string[:-2]

        # Getting values and executing
        for i,row in box_score.iterrows():
            row_list = list(row)
            row_list[0] = int(row_list[0])
            row_list = list(map(str, row_list))
            row_str = ", ".join(row_list)
            sql_cmd = f"INSERT INTO {self.db_table_name} ({field_string}) VALUES ({row_str});"
            cursor.execute(sql_cmd)
        return 1

    def points_to_label(self, points):
        label = np.zeros([11])        
        if points >= 50:
            index = -1
        else:
            index = int(np.floor(points / 5))
        label[index] = 1
        label.astype(float)
        return index, label

    def update_player_db(self, player_id):
        connection = sqlite3.connect('gamedata.db')
        cursor = connection.cursor()

        #Updating self.player_games if need be
        if self.player_games["last_updated"] is not self.today:
            all_games = self.get_player_history(player_id)
            self.player_games[str(player_id)] = all_games
        else:
            all_games = self.player_games[str(player_id)]
        
        #insert player_id into the database
        for game_id in all_games:
            check_cmd = f"SELECT EXISTS(SELECT * FROM {self.db_table_name} WHERE GAME_ID={game_id});"
            cursor.execute(check_cmd)
            game_exists = cursor.fetchall()[0][0]
            
            if not game_exists:
                box_score = self.get_box_score(game_id)
                insert_ret = self.insert_into_db(cursor, box_score=box_score)
                if insert_ret: connection.commit()
                logging.info("Game Inserted into Database: {}".format(game_id))

        logging.info("Player Data Added to Database: {}({})".format(self.get_player_name(player_id), player_id))
        cursor.close()
        connection.close()
        return -1

    def __del__(self):
        self.save_game_json()

class DataProcessor():
    def __init__(self):
        pass

class DeepModel(nn.Module):
    def __init__(self):
        pass

class PointPredictor():
    def __init__(self):
        pass

class Suggestor():
    def __init__(self):
        pass

class ParlYay():
    def __init__(self):
        pass


def test():
    dataf = DataFormatter()
    #Getting NBA Data:
    player_name = "Devin Booker"
    team_name = "Los Angeles Lakers"
    seq_len = 5
    player_season = "2021-22"
    team_id = teams.find_teams_by_full_name(team_name)[0]["id"]
    player_id = players.find_players_by_full_name(player_name)[0]["id"]

    print("Player Name: {} | Player ID: {}".format(player_name, player_id))
    print("-----------------")
    #dataf.inititalize_database(player_id)
    #dataf.update_player_db(player_id)
    #ret = dataf.get_player_datapoint(player_id, seq_len, most_recent=False)
    ret = dataf.get_team_datapoint(team_id, seq_len, most_recent=False)
    #ret = dataf.get_player_tensor(player_id)
    #ret = dataf.get_player_last_opponent(player_id)
    print(ret)

   


if __name__ == "__main__":
    test()
