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
        self.today = date.today().strftime("%d/%m/%y")
        self.game_path = "player_games.json"
        self.db_table_name = "gamedetails"
        self.ommitted_columns = ["index", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_NAME", "NICKNAME", "START_POSITION", "COMMENT", "MIN"]
        self.setup_logger()
        self.helper = Helper()
        self.player_games = self.load_game_json()
        #DELETE PRINT

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

    def get_player_datapoint(self, player_id, num_games, most_recent=False):
        #Getting player list of all the games
        if player_id not in self.player_games.keys():
            self.player_games[str(player_id)] = self.get_player_history(player_id)
        game_list = self.player_games[str(player_id)]
        game_list = list(map(int, self.player_games[str(player_id)]))

        connection = sqlite3.connect('gamedata.db')
        cursor = connection.cursor()

        #Checking to ensure there are enough games to make a datapoint
        if len(game_list) < (num_games):
            print("Sequence Length too Large")
            logging.info("- get_player_datapoint - Sequence Length too Large")

        #Determine whether grabbing random set of games or just the most recent(depending on if we want to sample currently or use dataformatting)
        if most_recent is True:
            print("Most Recent {}".format(num_games))
            index = 1 #IF GAME OF DAY IS GIVING ISSUE
            #index = 0 #OTHERWISE

        else:
            print("Random {}".format(num_games))
            index=random.randint(0, len(game_list) - num_games + 1)
            label_game = game_list[index-1]
        
        #Prepare the game list for getting db data
        game_sublist = game_list[index:(index + num_games)]
        game_sublist.reverse()

        #Get database data
        full_seq = []
        max_size = 0
        for game_id in game_sublist:
            game_arr = self.get_game_from_db(cursor, game_id)
            full_seq.append(game_arr)
            if game_arr.shape[0] > max_size: max_size = game_arr.shape[0]
        for i,score in full_seq:
            rows_to_add = max_size - score.shape[0]
            print(full_seq)
        datapoint = np.stack(full_seq)
        print(datapoint.shape)

        
        cursor.close()
        connection.close()

    def get_player_id(self, player_name):
        player_id = players.find_players_by_full_name(player_name)[0]["id"]
        return player_id
    
    def get_player_name(self, player_id):
        player_name = players.find_player_by_id(player_id)["full_name"]
        return player_name

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

    def get_box_score(self, game_id):
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).player_stats.get_data_frame()
        updated_box_score = box_score.dropna(subset=["MIN"]).reset_index().drop(self.ommitted_columns, axis=1).rename(columns={'TO': 'TOV'})
        return updated_box_score
    
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

    def get_game_from_db(self, cursor, game_id):
        get_cmd = f"SELECT * FROM {self.db_table_name} WHERE GAME_ID={game_id};"
        print(get_cmd)
        cursor.execute(get_cmd)
        select_res = cursor.fetchall()
        print(select_res)
        full_array = np.delete(np.array(select_res), obj=0, axis=1)
        return full_array
    
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

    def get_todays_games(self, game_id):
        pass

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
    seq_len = 5
    player_season = "2021-22"
    player_id = players.find_players_by_full_name(player_name)[0]["id"]
    #available_seasons = commonplayerinfo.CommonPlayerInfo(player_id=player_id).available_seasons.get_data_frame()
    #player_game_log = playergamelog.PlayerGameLog(player_id=player_id, season=player_season).get_data_frames()[0]
    #sample_game_id = player_game_log.loc[0,"Game_ID"]
    
    print("Player Name: {} | Player ID: {}".format(player_name, player_id))
    print("BOX SCORE\n-----------------")
    #dataf.inititalize_database(player_id)
    #dataf.update_player_db(player_id)
    ret = dataf.get_player_datapoint(player_id, seq_len, most_recent=True)
    #os.remove("gamedata.db")
    #ret = dataf.inititalize_database(player_id=player_id)
    #ret = dataf.update_player_db(player_id = player_id)

    #ret = dataf.get_player_tensor(player_id)
    print(ret)




if __name__ == "__main__":
    test()
