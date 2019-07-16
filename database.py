import utils

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
