import utils
import pandas as pd
import const
import numpy as np
from datetime import date, datetime, timedelta
import math
from time import time
import pickle
import cleaning as cl
from tensorflow import keras
import re


class Features:
  def __init__(self, FS, W1SP, W2SP, WSP, WRP, TPW, ACES, DF, UE, WIS, BP, NA, A1S, A2S, COMPLETE, SERVEADV, DIRECT, RETIRED, FATIGUE, UNC):
    self.FS = FS
    self.W1SP = W1SP
    self.W2SP = W2SP
    self.WSP = WSP
    self.WRP = WRP
    self.TPW = TPW
    self.ACES = ACES
    self.DF = DF
    self.UE = UE
    self.WIS = WIS
    self.BP = BP
    self.NA = NA
    self.A1S = A1S
    self.A2S = A2S
    self.COMPLETE = COMPLETE
    self.SERVEADV = SERVEADV
    self.DIRECT = DIRECT
    self.RETIRED = RETIRED
    self.FATIGUE = FATIGUE
    self.UNC = UNC


def update():

    print("Importing players...")
    # Importing Players Table
    query = "SELECT * FROM players_atp"
    df = utils.access_read(query)
    df = df[['ID_P', 'NAME_P', 'DATE_P', 'COUNTRY_P', 'PRIZE_P', 'POINT_P', 'RANK_P']]
    df.columns = ['PlayerID', 'Name', 'Birthdate', 'Country', 'Prize', 'ATPPoints', 'ATPRank']
    utils.sql_write(df, 'Players')
    df['Birthdate'] = pd.to_datetime(df['Birthdate']).apply(lambda x: x.date())
    df.to_pickle("players.pkl")

    print("Importing surfaces...")
    # Importing Surface Table
    query = "SELECT * FROM courts"
    df = utils.access_read(query)
    df.columns = ['SurfaceID', 'Name']
    utils.sql_write(df, 'Surface')
    df.to_pickle("surface.pkl")

    print("Importing tournaments...")
    # Importing Tournaments
    query = "SELECT * FROM tours_atp"
    df = utils.access_read(query)
    df = df[['ID_T', 'NAME_T', 'ID_C_T', 'DATE_T', 'RANK_T', 'COUNTRY_T']]
    df.columns = ['TournamentID', 'Name', 'SurfaceID', 'Date', 'RankID', 'Country']
    utils.sql_write(df, 'Tournaments')
    df['Date'] = pd.to_datetime(df['Date']).apply(lambda x: x.date())
    df.to_pickle("tournaments.pkl")

    print("Importing today's games...")
    # Importing Today's games
    query = 'SELECT * FROM today_atp'
    df = utils.access_read(query)
    # Unknown players
    df = df[(df['ID1'] != 3699) & (df['ID1'] != 3700)]
    df = df[(df['ID2'] != 3699) & (df['ID2'] != 3700)]
    df = df[df['RESULT'] == '']
    df = df[['TOUR', 'ID1', 'ID2']]
    df.columns = ['TournamentID', 'PlayerID_1', 'PlayerID_2']
    pl = pd.read_pickle("players.pkl")
    df = pd.merge(df, pl, 'inner', left_on=['PlayerID_1'], right_on=['PlayerID'])
    df = pd.merge(df, pl, 'inner', left_on=['PlayerID_2'], right_on=['PlayerID'])
    # Double Players
    df = df[~df['Name_x'].str.contains('/', regex=False)]
    df = df[~df['Name_y'].str.contains('/', regex=False)]
    df = df[['PlayerID_1', 'PlayerID_2', 'TournamentID']]
    # Surface
    tour = pd.read_pickle("tournaments.pkl")
    df = pd.merge(df, tour, 'left', on=['TournamentID'])
    df = df[['PlayerID_1', 'PlayerID_2', 'TournamentID', 'SurfaceID']]
    # Date
    now = datetime.now()
    df['Date'] = datetime(year=now.year, month=now.month, day=now.day).date()
    # Odds
    query = "SELECT * FROM odds_atp"
    odds = utils.access_read(query)
    odds = odds[odds['ID_B_O'] == const.BETTING_SITE] # only 1 betting site for now
    odds = odds[['ID1_O', 'ID2_O', 'ID_T_O', 'K1', 'K2']]
    odds.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Odds_1', 'Odds_2']
    df = pd.merge(df, odds, 'left', on=['PlayerID_1', 'PlayerID_2', 'TournamentID'])
    df = df[~np.isnan(df['Odds_1'])]
    df = df.reset_index(drop=True)
    df.to_pickle("Matches/matches_" + str(now.year) + "_" + str(now.month) + '_' + str(now.day) + ".pkl")

    print("Importing matches...")
    # Importing Games
    query = "SELECT * FROM games_atp"
    games = utils.access_read(query)
    games = games[['ID1_G', 'ID2_G', 'ID_T_G', 'DATE_G', 'RESULT_G']]
    games.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Date', 'Result']
    query = "SELECT * FROM odds_atp"
    odds = utils.access_read(query)
    odds = odds[odds['ID_B_O'] == const.BETTING_SITE] # only 1 betting site for now
    odds = odds[['ID1_O', 'ID2_O', 'ID_T_O', 'K1', 'K2']]
    odds.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Odds_1', 'Odds_2']
    query = "SELECT * FROM stat_atp"
    stat = utils.access_read(query)
    # First Serve Percentage
    stat['FSP_1'] = stat['FS_1'] / stat['FSOF_1']
    stat['FSP_2'] = stat['FS_2'] / stat['FSOF_2']
    # Win on 1st serve
    stat['W1SP_1'] = stat['W1S_1'] / stat['W1SOF_1']
    stat['W1SP_2'] = stat['W1S_2'] / stat['W1SOF_2']
    # Win on 2nd serve
    stat['W2SP_1'] = stat['W2S_1'] / stat['W2SOF_1']
    stat['W2SP_2'] = stat['W2S_2'] / stat['W2SOF_2']
    # Win on receiving
    stat['RPWP_1'] = stat['RPW_1'] / stat['RPWOF_1']
    stat['RPWP_2'] = stat['RPW_2'] / stat['RPWOF_2']
    # Break points percentage
    stat['BP_1'] = stat['BP_1'] / stat['BPOF_1']
    stat['BP_2'] = stat['BP_2'] / stat['BPOF_2']
    # Net approaches percentage
    stat['NA_1'] = stat['NA_1'] / stat['NAOF_1']
    stat['NA_2'] = stat['NA_2'] / stat['NAOF_2']
    # Total points won percentage
    stat['TPWP_1'] = stat['TPW_1'] / (stat['TPW_1'] + stat['TPW_2'])
    stat['TPWP_2'] = stat['TPW_2'] / (stat['TPW_1'] + stat['TPW_2'])

    stat = stat[['ID1', 'ID2', 'ID_T', 'FSP_1', 'FSP_2', 'ACES_1', 'ACES_2', 'DF_1', 'DF_2', 'UE_1', 'UE_2',
                 'W1SP_1', 'W1SP_2', 'W2SP_1', 'W2SP_2', 'RPWP_1', 'RPWP_2', 'WIS_1', 'WIS_2', 'BP_1', 'BP_2',
                 'NA_1', 'NA_2', 'TPWP_1', 'TPWP_2', 'FAST_1', 'FAST_2', 'A1S_1', 'A1S_2', 'A2S_1', 'A2S_2']]
    stat.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'FirstServePercentage_1', 'FirstServePercentage_2',
                    'Aces_1', 'Aces_2', 'DoubleFaults_1', 'DoubleFaults_2', 'UnforcedErrors_1', 'UnforcedErrors_2',
                    'WonFirstServePercentage_1', 'WonFirstServePercentage_2', 'WonSecondServePercentage_1',
                    'WonSecondServePercentage_2', 'WonReturnPercentage_1', 'WonReturnPercentage_2', 'Winners_1',
                    'Winners_2', 'BreakPointsWonPercentage_1', 'BreakPointsWonPercentage_2',
                    'NetApproachesWonPercentage_1', 'NetApproachesWonPercentage_2',
                    'TotalPointsWonPercentage_1', 'TotalPointsWonPercentage_2', 'FastestServe_1', 'FastestServe_2',
                    'AverageFirstServeSpeed_1', 'AverageFirstServeSpeed_2', 'AverageSecondServeSpeed_1',
                    'AverageSecondServeSpeed_2']
    # Merging games and odds
    df = pd.merge(games, odds, 'inner', on=['PlayerID_1', 'PlayerID_2', 'TournamentID'])
    odds1 = odds.rename(columns={'PlayerID_1': 'PlayerID_2',
                                 'PlayerID_2': 'PlayerID_1',
                                 'Odds_1': 'Odds_2',
                                 'Odds_2': 'Odds_1'})
    df1 = pd.merge(games, odds1, 'inner', on=['PlayerID_1', 'PlayerID_2', 'TournamentID'])
    df.equals(df1)
    df_odds = df.append(df1, sort=False)
    df_odds = df_odds.sort_values(by=['Date', 'PlayerID_1'])
    # Merging games and stats
    df_stat = pd.merge(games, stat, 'inner', on=['PlayerID_1', 'PlayerID_2', 'TournamentID'])
    df_stat = df_stat.sort_values(by=['Date', 'PlayerID_1'])
    # Merging both dataframes
    df_matches = pd.merge(df_odds, df_stat, 'inner', on=['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Date', 'Result'])
    df_matches = df_matches.sort_values(by=['Date', 'PlayerID_1'])
    utils.sql_write(df_matches, 'Matches')
    df_matches['Date'] = pd.to_datetime(df_matches['Date']).apply(lambda x: x.date())
    df_matches.to_pickle("matches.pkl")

    print("Updating features...")
    # Updating features
    updateFeatures()


# Finding comparable IDs
def FindComps(matches, ID1, ID2, GameDate):
    # All playerIDs played by player 1 and player2
    df_1 = matches[['PlayerID_1', 'PlayerID_2']].loc[((matches['PlayerID_1'] == ID1) |
                                                      (matches['PlayerID_2'] == ID1)) &
                                                     (matches['Date'] < GameDate)]
    # All distinct IDs
    df_1 = pd.DataFrame(pd.unique(df_1[['PlayerID_1', 'PlayerID_2']].values.ravel('K'))).sort_values(by=0)
    df_1.columns = ['ID']
    df_1 = df_1.loc[df_1['ID'] != ID1].reset_index(drop=True)

    # All playerIDs played by player 2
    df_2 = matches[['PlayerID_1', 'PlayerID_2']].loc[((matches['PlayerID_1'] == ID2) |
                                                           (matches['PlayerID_2'] == ID2)) &
                                                          (matches['Date'] < GameDate)]
    # All distinct IDs
    df_2 = pd.DataFrame(pd.unique(df_2[['PlayerID_1', 'PlayerID_2']].values.ravel('K'))).sort_values(by=0)
    df_2.columns = ['ID']
    df_2 = df_2.loc[df_2['ID'] != ID2].reset_index(drop=True)

    # Merging to find all comps IDs
    df = pd.merge(df_1, df_2, 'inner', on=['ID'])
    return df


def FindCompsGames(matches, ID1, ID2, GameDate):
    comps = FindComps(matches, ID1, ID2, GameDate)
    df = matches.loc[(((matches['PlayerID_1'] == ID1) & (matches['PlayerID_2'].isin(comps['ID']))) |
                      ((matches['PlayerID_2'] == ID1) & (matches['PlayerID_1'].isin(comps['ID'])))) &
                     (matches['Date'] < GameDate)].reset_index(drop=True)
    return df

# Discounting Stats
def TimeDiscounting(timeDF):
    timeDF = np.minimum(pow(const.DF, timeDF), const.DF)
    return timeDF


def AddWeightings(matches, ID1, ID2, GameDate, Surface):
    df = FindCompsGames(matches, ID1, ID2, GameDate)
    T = (GameDate - df['Date']).dt.days / 365
    df['Time Weighting'] = TimeDiscounting(T)
    df['Surface Weighting'] = 0.0
    for i, s in enumerate(df['SurfaceID']):
        df['Surface Weighting'].iat[i] = const.SURFACE_WEIGHTING[s][Surface]
    df['Weight'] = df['Time Weighting'] * df['Surface Weighting']
    return df


def Direct(matches, ID1, ID2, GameDate):
    ID1_win = matches['PlayerID_1'].loc[(matches['PlayerID_1'] == ID1) & (matches['PlayerID_2'] == ID2) & (matches['Date'] < GameDate)].size
    ID2_win = matches['PlayerID_1'].loc[(matches['PlayerID_1'] == ID2) & (matches['PlayerID_2'] == ID1) & (matches['Date'] < GameDate)].size
    return ID1_win / (ID1_win + ID2_win) if (ID1_win + ID2_win) else 0.5


def Fatigue(matches, ID1, ID2, GameDate):
    fatigue1 = 0.0
    fatigue2 = 0.0
    df1 = matches[(matches['PlayerID_1'] == ID1) | (matches['PlayerID_2'] == ID1)]
    df1 = df1[(df1['Date'] < GameDate) & (df1['Date'] >= GameDate - timedelta(days=3))]
    if not df1.empty:
        for i, r in df1.iterrows():
            myStr = r['Result']
            nums = re.findall('\d+', myStr)
            breakpoints = re.findall("\((\d+)\)", myStr)
            nums = [int(x) for x in nums]
            breakpoints = [int(x) for x in breakpoints]
            fatigue1 += (sum(nums) - sum(breakpoints)) * const.FDF ** (GameDate - r['Date']).days
    df1 = matches[(matches['PlayerID_1'] == ID2) | (matches['PlayerID_2'] == ID2)]
    df1 = df1[(df1['Date'] < GameDate) & (df1['Date'] >= GameDate - timedelta(days=3))]
    if not df1.empty:
        for i, r in df1.iterrows():
            myStr = r['Result']
            nums = re.findall('\d+', myStr)
            breakpoints = re.findall("\((\d+)\)", myStr)
            nums = [int(x) for x in nums]
            breakpoints = [int(x) for x in breakpoints]
            nums = [int(x) for x in nums]
            fatigue2 += (sum(nums) - sum(breakpoints)) * const.FDF ** (GameDate - r['Date']).days
    return fatigue1 - fatigue2


def Retired(matches, ID1, ID2, GameDate):
    # Did he lose last game by retirement
    df = matches[matches['PlayerID_2'] == ID1]
    df = df[df['Date'] < GameDate]
    retired = 0
    if not df.empty:
        df = df.tail(1)
        if "ret" in df['Result'].iat[0]:
            # check if he won a game after coming back
            df1 = matches[matches['PlayerID_1'] == ID1]
            df1 = df1[df1['Date'] < GameDate]
            df1 = df1.tail(1)
            if not df1.empty:
                if df['Date'].iat[0] > df1['Date'].iat[0]:
                    retired += 1

        # Did he lose last game by retirement
    df = matches[matches['PlayerID_2'] == ID2]
    df = df[df['Date'] < GameDate]
    if not df.empty:
        df = df.tail(1)
        if "ret" in df['Result'].iat[0]:
            # check if he won a game after coming back
            df1 = matches[matches['PlayerID_1'] == ID2]
            df1 = df1[df1['Date'] < GameDate]
            df1 = df1.tail(1)
            if not df1.empty:
                if df['Date'].iat[0] > df1['Date'].iat[0]:
                    retired -= 1

    return retired


# f = pd.read_pickle("features_v2.pkl").reset_index(drop=True)
# m = pd.read_pickle("matches.pkl")
# f['FATIGUE'] = 0.0
# for i, r in f.iterrows():
#     f['FATIGUE'].iat[i] = Fatigue(m, r['PlayerID_1'], r['PlayerID_2'], r['Date'])
#     if i % 1000 == 0:
#         print(i)


def replace_nan(x):
    return 0 if math.isnan(x) else x


def createFeatures(matches, ID1, ID2, GameDate, Surface):
    df = FindComps(matches, ID1, ID2, GameDate)
    if df.empty:
        return None
    else:
        df1 = AddWeightings(matches, ID1, ID2, GameDate, Surface)
        df2 = AddWeightings(matches, ID2, ID1, GameDate, Surface)
        # if (df1 is None) | (df2 is None):
        #     return 0.0
        totalFS1 = 0.0
        totalFS2 = 0.0
        totalW1SP1 = 0.0
        totalW1SP2 = 0.0
        totalW2SP1 = 0.0
        totalW2SP2 = 0.0
        totalWSP1 = 0.0
        totalWSP2 = 0.0
        totalWRP1 = 0.0
        totalWRP2 = 0.0
        totalTPW1 = 0.0
        totalTPW2 = 0.0
        totalACES1 = 0.0
        totalACES2 = 0.0
        totalDF1 = 0.0
        totalDF2 = 0.0
        totalUE1 = 0.0
        totalUE2 = 0.0
        totalWIS1 = 0.0
        totalWIS2 = 0.0
        totalBP1 = 0.0
        totalBP2 = 0.0
        totalNA1 = 0.0
        totalNA2 = 0.0
        totalA1S1 = 0.0
        totalA1S2 = 0.0
        totalA2S1 = 0.0
        totalA2S2 = 0.0
        totalUNC = 0.0
        for i, c in enumerate(df['ID']):
            # If Comp is Player 1
            df3 = df1.loc[(df1['PlayerID_1'] == df['ID'].iat[i])]
            # If Comp is Player 2
            df4 = df1.loc[(df1['PlayerID_2'] == df['ID'].iat[i])]
            df4 = df4.rename(columns={'PlayerID_1': 'PlayerID_2',
                                     'PlayerID_2': 'PlayerID_1',
                                     'Odds_1': 'Odds_2',
                                     'Odds_2': 'Odds_1',
                                     'FirstServePercentage_1': 'FirstServePercentage_2',
                                     'FirstServePercentage_2': 'FirstServePercentage_1',
                                     'Aces_1': 'Aces_2',
                                     'Aces_2': 'Aces_1',
                                     'DoubleFaults_1': 'DoubleFaults_2',
                                     'DoubleFaults_2': 'DoubleFaults_1',
                                     'UnforcedErrors_1': 'UnforcedErrors_2',
                                     'UnforcedErrors_2': 'UnforcedErrors_1',
                                     'WonFirstServePercentage_1': 'WonFirstServePercentage_2',
                                     'WonFirstServePercentage_2': 'WonFirstServePercentage_1',
                                     'WonSecondServePercentage_1': 'WonSecondServePercentage_2',
                                     'WonSecondServePercentage_2': 'WonSecondServePercentage_1',
                                     'WonReturnPercentage_1': 'WonReturnPercentage_2',
                                     'WonReturnPercentage_2': 'WonReturnPercentage_1',
                                     'Winners_1': 'Winners_2',
                                     'Winners_2': 'Winners_1',
                                     'BreakPointsWonPercentage_1': 'BreakPointsWonPercentage_2',
                                     'BreakPointsWonPercentage_2': 'BreakPointsWonPercentage_1',
                                     'NetApproachesWonPercentage_1': 'NetApproachesWonPercentage_2',
                                     'NetApproachesWonPercentage_2': 'NetApproachesWonPercentage_1',
                                     'TotalPointsWon_1': 'TotalPointsWon_2',
                                     'TotalPointsWon_2': 'TotalPointsWon_1',
                                     'FastestServe_1': 'FastestServe_2',
                                     'FastestServe_2': 'FastestServe_1',
                                     'AverageFirstServeSpeed_1': 'AverageFirstServeSpeed_2',
                                     'AverageFirstServeSpeed_2': 'AverageFirstServeSpeed_1',
                                     'AverageSecondServeSpeed_1': 'AverageSecondServeSpeed_2',
                                     'AverageSecondServeSpeed_2': 'AverageSecondServeSpeed_1'})
            df_player1 = df3.append(df4, sort=False)
            # If Comp is Player 1
            df5 = df2.loc[(df2['PlayerID_1'] == df['ID'].iat[i])]
            # If Comp is Player 2
            df6 = df2.loc[(df2['PlayerID_2'] == df['ID'].iat[i])]
            df6 = df6.rename(columns={'PlayerID_1': 'PlayerID_2',
                                      'PlayerID_2': 'PlayerID_1',
                                      'Odds_1': 'Odds_2',
                                      'Odds_2': 'Odds_1',
                                      'FirstServePercentage_1': 'FirstServePercentage_2',
                                      'FirstServePercentage_2': 'FirstServePercentage_1',
                                      'Aces_1': 'Aces_2',
                                      'Aces_2': 'Aces_1',
                                      'DoubleFaults_1': 'DoubleFaults_2',
                                      'DoubleFaults_2': 'DoubleFaults_1',
                                      'UnforcedErrors_1': 'UnforcedErrors_2',
                                      'UnforcedErrors_2': 'UnforcedErrors_1',
                                      'WonFirstServePercentage_1': 'WonFirstServePercentage_2',
                                      'WonFirstServePercentage_2': 'WonFirstServePercentage_1',
                                      'WonSecondServePercentage_1': 'WonSecondServePercentage_2',
                                      'WonSecondServePercentage_2': 'WonSecondServePercentage_1',
                                      'WonReturnPercentage_1': 'WonReturnPercentage_2',
                                      'WonReturnPercentage_2': 'WonReturnPercentage_1',
                                      'Winners_1': 'Winners_2',
                                      'Winners_2': 'Winners_1',
                                      'BreakPointsWonPercentage_1': 'BreakPointsWonPercentage_2',
                                      'BreakPointsWonPercentage_2': 'BreakPointsWonPercentage_1',
                                      'NetApproachesWonPercentage_1': 'NetApproachesWonPercentage_2',
                                      'NetApproachesWonPercentage_2': 'NetApproachesWonPercentage_1',
                                      'TotalPointsWon_1': 'TotalPointsWon_2',
                                      'TotalPointsWon_2': 'TotalPointsWon_1',
                                      'FastestServe_1': 'FastestServe_2',
                                      'FastestServe_2': 'FastestServe_1',
                                      'AverageFirstServeSpeed_1': 'AverageFirstServeSpeed_2',
                                      'AverageFirstServeSpeed_2': 'AverageFirstServeSpeed_1',
                                      'AverageSecondServeSpeed_1': 'AverageSecondServeSpeed_2',
                                      'AverageSecondServeSpeed_2': 'AverageSecondServeSpeed_1'})
            df_player2 = df5.append(df6, sort=False)

            FS1 = df_player1['FirstServePercentage_2'].mean()
            totalFS1 += FS1
            FS2 = df_player2['FirstServePercentage_2'].mean()
            totalFS2 += FS2
            W1SP1 = df_player1['WonFirstServePercentage_2'].mean()
            totalW1SP1 += W1SP1
            W1SP2 = df_player2['WonFirstServePercentage_2'].mean()
            totalW1SP2 += W1SP2
            W2SP1 = df_player1['WonSecondServePercentage_2'].mean()
            totalW2SP1 += W2SP1
            W2SP2 = df_player2['WonSecondServePercentage_2'].mean()
            totalW2SP2 += W2SP2
            WSP1 = W1SP1 * FS1 + W2SP1 * (1 - FS1)
            totalWSP1 += WSP1
            WSP2 = W1SP2 * FS2 + W2SP2 * (1 - FS2)
            totalWSP2 += WSP2
            WRP1 = df_player1['WonReturnPercentage_2'].mean()
            totalWRP1 += WRP1
            WRP2 = df_player2['WonReturnPercentage_2'].mean()
            totalWRP2 += WRP2
            TPW1 = df_player1['TotalPointsWonPercentage_2'].mean()
            totalTPW1 += TPW1
            TPW2 = df_player2['TotalPointsWonPercentage_2'].mean()
            totalTPW2 += TPW2
            ACES1 = df_player1['Aces_2'].mean()
            totalACES1 += ACES1
            ACES2 = df_player2['Aces_2'].mean()
            totalACES2 += ACES2
            DF1 = df_player1['DoubleFaults_2'].mean()
            totalDF1 += DF1
            DF2 = df_player2['DoubleFaults_2'].mean()
            totalDF2 += DF2
            UE1 = df_player1['UnforcedErrors_2'].mean()
            totalUE1 += UE1
            UE2 = df_player2['UnforcedErrors_2'].mean()
            totalUE2 += UE2
            WIS1 = df_player1['Winners_2'].mean()
            totalWIS1 += WIS1
            WIS2 = df_player2['Winners_2'].mean()
            totalWIS2 += WIS2
            BP1 = df_player1['BreakPointsWonPercentage_2'].mean()
            totalBP1 += BP1
            BP2 = df_player2['BreakPointsWonPercentage_2'].mean()
            totalBP2 += BP2
            NA1 = df_player1['NetApproachesWonPercentage_2'].mean()
            totalNA1 += NA1
            NA2 = df_player2['NetApproachesWonPercentage_2'].mean()
            totalNA2 += NA2
            A1S1 = df_player1['AverageFirstServeSpeed_2'].mean()
            totalA1S1 += A1S1
            A1S2 = df_player2['AverageFirstServeSpeed_2'].mean()
            totalA1S2 += A1S2
            A2S1 = df_player1['AverageSecondServeSpeed_2'].mean()
            totalA2S1 += A2S1
            A2S2 = df_player2['AverageSecondServeSpeed_2'].mean()
            totalA2S2 += A2S2
            UNC = sum(df_player1['Weight']) * sum(df_player2['Weight'])
            totalUNC += UNC

        n = len(df['ID'])
        if n != 0:
            totalFS1 /= n
            totalFS2 /= n
            totalW1SP1 /= n
            totalW1SP2 /= n
            totalW2SP1 /= n
            totalW2SP2 /= n
            totalWSP1 /= n
            totalWSP2 /= n
            totalWRP1 /= n
            totalWRP2 /= n
            totalTPW1 /= n
            totalTPW2 /= n
            totalACES1 /= n
            totalACES2 /= n
            totalDF1 /= n
            totalDF2 /= n
            totalUE1 /= n
            totalUE2 /= n
            totalWIS1 /= n
            totalWIS2 /= n
            totalBP1 /= n
            totalBP2 /= n
            totalNA1 /= n
            totalNA2 /= n
            totalA1S1 /= n
            totalA1S2 /= n
            totalA2S1 /= n
            totalA2S2 /= n
        else:
            return None

        COMPLETE1 = totalWSP1 * totalWRP1
        COMPLETE2 = totalWSP2 * totalWRP2
        SERVEADV1 = totalWSP1 - totalWRP2
        SERVEADV2 = totalWSP2 - totalWRP1

        f = Features(replace_nan(totalFS1 - totalFS2),
                     replace_nan(totalW1SP1 - totalW1SP2),
                     replace_nan(totalW2SP1 - totalW2SP2),
                     replace_nan(totalWSP1 - totalWSP2),
                     replace_nan(totalWRP1 - totalWRP2),
                     replace_nan(totalTPW1 - totalTPW2),
                     replace_nan(totalACES1 - totalACES2),
                     replace_nan(totalDF1 - totalDF2),
                     replace_nan(totalUE1 - totalUE2),
                     replace_nan(totalWIS1 - totalWIS2),
                     replace_nan(totalBP1 - totalBP2),
                     replace_nan(totalNA1 - totalNA2),
                     replace_nan(totalA1S1 - totalA1S2),
                     replace_nan(totalA2S1 - totalA2S2),
                     replace_nan(COMPLETE1 - COMPLETE2),
                     replace_nan(SERVEADV1 - SERVEADV2),
                     replace_nan(Direct(matches, ID1, ID2, GameDate)),
                     replace_nan(Retired(matches, ID1, ID2, GameDate)),
                     replace_nan(Fatigue(matches, ID1, ID2, GameDate)),
                     replace_nan(totalUNC))
        return f

#
# def updateFeatures():
#     start = time()
#     matches = pd.read_pickle("matches.pkl")
#     matches = matches.loc[matches['Date'].notnull()]
#     # matches = cl.pre_clean(matches)
#     tournaments = pd.read_pickle("tournaments.pkl")
#     df = pd.merge(matches, tournaments[['TournamentID','SurfaceID']], 'inner', on=['TournamentID'])
#     # Compare matches with features
#     features = df[['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Date', 'Odds_1', 'Odds_2']]
#     features['FS'] = 0.0
#     features['W1SP'] = 0.0
#     features['W2SP'] = 0.0
#     features['WSP'] = 0.0
#     features['WRP'] = 0.0
#     features['TPW'] = 0.0
#     features['ACES'] = 0.0
#     features['DF'] = 0.0
#     features['UE'] = 0.0
#     features['WIS'] = 0.0
#     features['BP'] = 0.0
#     features['NA'] = 0.0
#     features['A1S'] = 0.0
#     features['A2S'] = 0.0
#     features['COMPLETE'] = 0.0
#     features['SERVEADV'] = 0.0
#     features['DIRECT'] = 0.5
#     features['UNC'] = 0.0
#     for i, r in df.iterrows():
#         f = createFeatures(df, r['PlayerID_1'], r['PlayerID_2'], r['Date'], r['SurfaceID'])
#         if f is not None:
#             features['FS'].iat[i] = f.FS
#             features['W1SP'].iat[i] = f.W1SP
#             features['W2SP'].iat[i] = f.W2SP
#             features['WSP'].iat[i] = f.WSP
#             features['WRP'].iat[i] = f.WRP
#             features['TPW'].iat[i] = f.TPW
#             features['ACES'].iat[i] = f.ACES
#             features['DF'].iat[i] = f.DF
#             features['UE'].iat[i] = f.UE
#             features['WIS'].iat[i] = f.WIS
#             features['BP'].iat[i] = f.BP
#             features['NA'].iat[i] = f.NA
#             features['A1S'].iat[i] = f.A1S
#             features['A2S'].iat[i] = f.A2S
#             features['COMPLETE'].iat[i] = f.COMPLETE
#             features['SERVEADV'].iat[i] = f.SERVEADV
#             features['DIRECT'].iat[i] = f.DIRECT
#             features['UNC'].iat[i] = f.UNC
#         if i % 100 == 0:
#             print(i)
#
#     end = time()
#     print('Time: ' + str(end - start) + 's')
#     features.to_pickle("features.pkl")
#     return

def updateFeatures():

    start = time()
    matches = pd.read_pickle("matches.pkl")
    matches = matches.loc[matches['Date'].notnull()]
    # matches = cl.pre_clean(matches)
    tournaments = pd.read_pickle("tournaments.pkl")
    df = pd.merge(matches, tournaments[['TournamentID', 'SurfaceID']], 'inner', on=['TournamentID'])
    # f = pd.read_pickle("features_v2.pkl")
    # # f = pd.read_pickle("features.pkl")
    # maxDate = max(f['Date'] - timedelta(days=3))
    # print(maxDate)
    # Compare matches with features
    features = df[['PlayerID_1', 'PlayerID_2', 'TournamentID', 'SurfaceID', 'Date', 'Odds_1', 'Odds_2']]
    # features = features[features['Date'] >= maxDate].reset_index(drop=True)
    print(features['PlayerID_1'].count())
    features['FS'] = 0.0
    features['W1SP'] = 0.0
    features['W2SP'] = 0.0
    features['WSP'] = 0.0
    features['WRP'] = 0.0
    features['TPW'] = 0.0
    features['ACES'] = 0.0
    features['DF'] = 0.
    features['UE'] = 0.0
    features['WIS'] = 0.0
    features['BP'] = 0.0
    features['NA'] = 0.0
    features['A1S'] = 0.0
    features['A2S'] = 0.0
    features['COMPLETE'] = 0.0
    features['SERVEADV'] = 0.0
    features['DIRECT'] = 0.5
    features['RETIRED'] = 0
    features['FATIGUE'] = 0.0
    features['UNC'] = 0.0
    for i, r in features.iterrows():
        f = createFeatures(df, r['PlayerID_1'], r['PlayerID_2'], r['Date'], r['SurfaceID'])
        if f is not None:
            features['FS'].iat[i] = f.FS
            features['W1SP'].iat[i] = f.W1SP
            features['W2SP'].iat[i] = f.W2SP
            features['WSP'].iat[i] = f.WSP
            features['WRP'].iat[i] = f.WRP
            features['TPW'].iat[i] = f.TPW
            features['ACES'].iat[i] = f.ACES
            features['DF'].iat[i] = f.DF
            features['UE'].iat[i] = f.UE
            features['WIS'].iat[i] = f.WIS
            features['BP'].iat[i] = f.BP
            features['NA'].iat[i] = f.NA
            features['A1S'].iat[i] = f.A1S
            features['A2S'].iat[i] = f.A2S
            features['COMPLETE'].iat[i] = f.COMPLETE
            features['SERVEADV'].iat[i] = f.SERVEADV
            features['DIRECT'].iat[i] = f.DIRECT
            features['RETIRED'].iat[i] = f.RETIRED
            features['FATIGUE'].iat[i] = f.FATIGUE
            features['UNC'].iat[i] = f.UNC
        print(i)
        if (i % 2000 == 0) & (i != 0):
            print(i)
            break

    end = time()
    print('Time: ' + str(end - start) + 's')
    # f = pd.read_pickle("features_v2.pkl")
    # f = f.append(features, sort=False)
    # f = f.sort_values(by=['Date']).reset_index(drop=True)
    pl = pd.read_pickle("players.pkl")
    # f = pd.merge(f, pl, 'inner', left_on=['PlayerID_1'], right_on=['PlayerID'])
    f = pd.merge(features, pl, 'inner', left_on=['PlayerID_1'], right_on=['PlayerID'])
    f = pd.merge(f, pl, 'inner', left_on=['PlayerID_2'], right_on=['PlayerID'])
    f = f[~f['Name_x'].str.contains('/', regex=False)]
    f = f[~f['Name_y'].str.contains('/', regex=False)]
    f = f.sort_values(by=['Date'])
    f = f.reset_index(drop=True)
    f = f[['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Date', 'Odds_1', 'Odds_2',
       'FS', 'W1SP', 'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'UE', 'WIS',
       'BP', 'NA', 'A1S', 'A2S', 'COMPLETE', 'SERVEADV', 'DIRECT', 'RETIRED', 'FATIGUE', 'UNC']]
    f = f.drop_duplicates()
    f.to_pickle("features_v3.pkl")
    print("pickled")
    return



def new_games(data):
    matches = pd.read_pickle("matches.pkl")
    matches = matches.loc[matches['Date'].notnull()]
    # matches = cl.pre_clean(matches)
    tournaments = pd.read_pickle("tournaments.pkl")
    df = pd.merge(matches, tournaments[['TournamentID', 'SurfaceID']], 'inner', on=['TournamentID'])
    features = data
    features['FS'] = 0.0
    features['W1SP'] = 0.0
    features['W2SP'] = 0.0
    features['WSP'] = 0.0
    features['WRP'] = 0.0
    features['TPW'] = 0.0
    features['ACES'] = 0.0
    features['DF'] = 0.0
    features['UE'] = 0.0
    features['WIS'] = 0.0
    features['BP'] = 0.0
    features['NA'] = 0.0
    features['A1S'] = 0.0
    features['A2S'] = 0.0
    features['COMPLETE'] = 0.0
    features['SERVEADV'] = 0.0
    features['DIRECT'] = 0.5
    features['RETIRED'] = 0
    features['FATIGUE'] = 0.0
    features['UNC'] = 0.0
    for i, r in data.iterrows():
        f = createFeatures(df, r['PlayerID_1'], r['PlayerID_2'], r['Date'], r['SurfaceID'])
        if f is not None:
            features['FS'].iat[i] = f.FS
            features['W1SP'].iat[i] = f.W1SP
            features['W2SP'].iat[i] = f.W2SP
            features['WSP'].iat[i] = f.WSP
            features['WRP'].iat[i] = f.WRP
            features['TPW'].iat[i] = f.TPW
            features['ACES'].iat[i] = f.ACES
            features['DF'].iat[i] = f.DF
            features['UE'].iat[i] = f.UE
            features['WIS'].iat[i] = f.WIS
            features['BP'].iat[i] = f.BP
            features['NA'].iat[i] = f.NA
            features['A1S'].iat[i] = f.A1S
            features['A2S'].iat[i] = f.A2S
            features['COMPLETE'].iat[i] = f.COMPLETE
            features['SERVEADV'].iat[i] = f.SERVEADV
            features['DIRECT'].iat[i] = f.DIRECT
            features['RETIRED'].iat[i] = f.RETIRED
            features['FATIGUE'].iat[i] = f.FATIGUE
            features['UNC'].iat[i] = f.UNC
    return features


