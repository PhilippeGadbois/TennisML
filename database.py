import utils
import pandas as pd
import const

def update():
    # Importing Players Table
    query = "SELECT * FROM players_atp"
    df = utils.access_read(query)
    df = df[['ID_P', 'NAME_P', 'DATE_P', 'COUNTRY_P', 'PRIZE_P', 'POINT_P', 'RANK_P']]
    df.columns = ['PlayerID', 'Name', 'Birthdate', 'Country', 'Prize', 'ATPPoints', 'ATPRank']
    utils.sql_write(df, 'Players')

    # Importing Surface Table
    query = "SELECT * FROM courts"
    df = utils.access_read(query)
    df.columns = ['SurfaceID', 'Name']
    utils.sql_write(df, 'Surface')

    # Importing Tournaments
    query = "SELECT * FROM tours_atp"
    df = utils.access_read(query)
    df = df[['ID_T', 'NAME_T', 'ID_C_T', 'DATE_T', 'RANK_T', 'COUNTRY_T']]
    df.columns = ['TournamentID', 'Name', 'SurfaceID', 'Date', 'RankID', 'Country']
    utils.sql_write(df, 'Tournaments')

    # Importing Games
    query = "SELECT * FROM games_atp"
    games = utils.access_read(query)
    games = games[['ID1_G', 'ID2_G', 'ID_T_G', 'DATE_G', 'RESULT_G']]
    games.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'Date', 'Result']
    query = "SELECT * FROM odds_atp"
    odds = utils.access_read(query)
    odds = odds[odds['ID_B_O'] == 1] # only 1 betting site for now
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
    stat = stat[['ID1', 'ID2', 'ID_T', 'FSP_1', 'FSP_2', 'ACES_1', 'ACES_2', 'DF_1', 'DF_2', 'UE_1', 'UE_2',
                 'W1SP_1', 'W1SP_2', 'W2SP_1', 'W2SP_2', 'RPWP_1', 'RPWP_2', 'WIS_1', 'WIS_2', 'BP_1', 'BP_2',
                 'BPOF_1', 'BPOF_2', 'NA_1', 'NA_2', 'NAOF_1', 'NAOF_2', 'TPW_1', 'TPW_2', 'FAST_1', 'FAST_2',
                 'A1S_1', 'A1S_2', 'A2S_1', 'A2S_2']]
    stat.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'FirstServePercentage_1', 'FirstServePercentage_2',
                    'Aces_1', 'Aces_2', 'DoubleFaults_1', 'DoubleFaults_2', 'UnforcedErrors_1', 'UnforcedErrors_2',
                    'WonFirstServePercentage_1', 'WonFirstServePercentage_2', 'WonSecondServePercentage_1',
                    'WonSecondServePercentage_2', 'WonReturnPercentage_1', 'WonReturnPercentage_2', 'Winners_1',
                    'Winners_2', 'BreakPointsWon_1', 'BreakPointsWon_2', 'BreakPointsTotal_1', 'BreakPointsTotal_2',
                    'NetApproachesWon_1', 'NetApproachesWon_2', 'NetApproachesTotal_1', 'NetApproachesTotal_2',
                    'TotalPointsWon_1', 'TotalPointsWon_2', 'FastestServe_1', 'FastestServe_2',
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

# Finding comparable IDs
def FindComps(ID1, ID2, GameDate):
    query = "DECLARE @Player_ID INTEGER; "\
            "DECLARE @GameDate VARCHAR(255); "\
            "SET @Player_ID = " + str(ID1) +\
            "SET @GameDate = " + GameDate +\
            "SELECT DISTINCT PlayerID_1 AS ID "\
            "  FROM [TennisML].[Stat].[Matches] "\
            "  WHERE PlayerID_2 = @Player_ID AND Date < @GameDate "\
            "UNION "\
            "SELECT DISTINCT PlayerID_2 AS ID "\
            "  FROM [TennisML].[Stat].[Matches] "\
            "  WHERE PlayerID_1 = @Player_ID AND Date < @GameDate "
    df_1 = utils.sql_read(query)
    query = "DECLARE @Player_ID INTEGER; "\
            "DECLARE @GameDate VARCHAR(255); "\
            "SET @Player_ID = " + str(ID2) +\
            "SET @GameDate = " + GameDate +\
            "SELECT DISTINCT PlayerID_1 AS ID "\
            "  FROM [TennisML].[Stat].[Matches] "\
            "  WHERE PlayerID_2 = @Player_ID AND Date < @GameDate "\
            "UNION "\
            "SELECT DISTINCT PlayerID_2 AS ID "\
            "  FROM [TennisML].[Stat].[Matches] "\
            "  WHERE PlayerID_1 = @Player_ID AND Date < @GameDate "
    df_2 = utils.sql_read(query)
    df = df_1[df_1.isin(df_2)]
    return df.dropna().astype(int)

def FindCompsGames(ID1, ID2, GameDate):
    ID1 = 19
    ID2 = 5992
    GameDate = '20140101'
    df = FindComps(ID1, ID2, GameDate)
    # TODO: Complete function. Adding ids to sql query
    comps = df.to_string(index=False)
    query = "SELECT * "\
            "FROM TennisML.Stat.Matches "\
            "WHERE (PlayerID_1 IN (" + ID1 + ") AND PlayerID_2 IN (5992, 2)) "\
            "OR (PlayerID_1 IN (5992, 2) AND PlayerID_2 IN (19)) "\
            "AND Date < '20140101'"

# Discounting Stats
def TimeDiscounting(time):
    return min(const.DF ** time, const.DF)



df = FindCompsGames(19, 5992, '20150101')



    # Making Features
    query = "SELECT * FROM Stat.Features"
    features = utils.sql_read(query)
    query = "SELECT * FROM Stat.Matches"
    matches = utils.sql_read(query)
    # Compare matches with features


    # For each missing rows, build features