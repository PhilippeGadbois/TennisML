import utils

query = "SELECT * FROM Stat.Players"
df = utils.sql_read(query)

query = "SELECT * FROM players_atp"
df = utils.access_read(query)
df = df[['ID_P', 'NAME_P', 'DATE_P', 'COUNTRY_P', 'PRIZE_P', ]]

utils.sql_write(df, 'Stat.Players')