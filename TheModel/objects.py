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
import requests
import sqlite3

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



class DataFormatter():
    def __init__(self):
        self.ommitted_columns = ["index", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_NAME", "NICKNAME", "START_POSITION", "COMMENT", "MIN"]
    
    def get_box_score(self, game_id):
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).player_stats.get_data_frame()
        updated_box_score = box_score.dropna(subset=["MIN"]).reset_index().drop(self.ommitted_columns, axis=1)
        return updated_box_score
    
    def get_todays_games(self, game_id):
        pass

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

#DEPRECATED: For now at least 1/22/2022
class KeyTracker():
    def __init__(self):
        # Internal elements
        # self.key_path will vary machine to machine as its private information
        self.key_path = "api-keys.json"
        self.key_dict = self.load_json()
        self.max_uses = 500
        self.current_key = self.get_usable_key()

    def load_json(self):
        with open(self.key_path) as fp:
            ret_dict = json.load(fp)
        fp.close()
        return ret_dict
    
    def get_usable_key(self):
        for key in self.key_dict.keys():
            if self.key_dict[key]["available"] == True:
                return key
        print("ERROR: No usable api keys for The-Odds-Api")
        return("BADKEY")

    def update_tracker(self, uses=1):
        if self.current_key != "BADKEY":
            self.key_dict[self.current_key]["uses_left"] = self.key_dict[self.current_key]["uses_left"] - uses
            self.key_dict[self.current_key]["uses_total"] = self.key_dict[self.current_key]["uses_total"] + uses
            if self.key_dict[self.current_key]["uses_left"] <= 0:
                self.key_dict[self.current_key]["available"] = False
            self.current_key = self.get_usable_key()

    def get_api_key(self):
        if self.current_key != "BADKEY":
            return self.key_dict[self.current_key]["key"]
        else:
            print("ERROR: No usable api keys for The-Odds-Api")
            
    
    def backup_json(self):
        with open(self.key_path, "w+") as fp:
            json.dump(self.key_dict, fp)
        fp.close()

    def reset_keys(self):
        for key in self.key_dict.keys():
            self.current_key = key
            self.key_dict[self.current_key]["uses_left"] = self.max_uses
            self.key_dict[self.current_key]["uses_total"] = 0
            self.key_dict[self.current_key]["available"] = True
        self.current_key = self.get_usable_key()
        print("SUCCESS: Reset the JSON keys")
    
    def get_request(self, path, params, odds_api=False):
        if odds_api:
            api_key = self.get_api_key()
            params['api_key'] = api_key

        response = requests.get(path, params)

        if response.status_code != 200:
            print(f'Failed to get odds: status_code {response.status_code}, response body {response.text}')
        else:
            resp_json = response.json()
            if odds_api:
                self.update_tracker()

        return resp_json

    def __del__(self):
        self.backup_json()


def test():
    dataf = DataFormatter()
    #Getting NBA Data:
    player_name = "Devin Booker"
    player_season = "2021-22"
    player_id = players.find_players_by_full_name(player_name)[0]["id"]
    available_seasons = commonplayerinfo.CommonPlayerInfo(player_id=player_id).available_seasons.get_data_frame()
    player_game_log = playergamelog.PlayerGameLog(player_id=player_id, season=player_season).get_data_frames()[0]
    sample_game_id = player_game_log.loc[0,"Game_ID"]
    
    print("Player Name: {} | Player ID: {}".format(player_name, player_id))
    print("BOX SCORE\n-----------------")
    print(dataf.get_box_score(sample_game_id))


if __name__ == "__main__":
    test()
