import cleaning as cl
from collections import OrderedDict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import const
from datetime import date, datetime
import seaborn as sns
sns.set(color_codes=True)
from matplotlib import pyplot
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import database as db
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D

def build_variables(x):

    rows_list = []
    for i, r in x.iterrows():
        dict = OrderedDict()
        dict.update(r)
        rows_list.append(dict)

    rows_list = pd.DataFrame(rows_list)
    rows_list['FS'] = -rows_list['FS']
    rows_list['W1SP'] = -rows_list['W1SP']
    rows_list['W2SP'] = -rows_list['W2SP']
    rows_list['WSP'] = -rows_list['WSP']
    rows_list['WRP'] = -rows_list['WRP']
    rows_list['TPW'] = -rows_list['TPW']
    rows_list['ACES'] = -rows_list['ACES']
    rows_list['DF'] = -rows_list['DF']
    rows_list['BP'] = -rows_list['BP']
    rows_list['COMPLETE'] = -rows_list['COMPLETE']
    rows_list['SERVEADV'] = -rows_list['SERVEADV']
    rows_list['DIRECT'] = (1 - rows_list['DIRECT'])
    rows_list['Odds'] = - rows_list['Odds']
    rows_list['Y'] = 0
    x = x.append(rows_list, sort=False)
    x = cl.post_clean(x)
    return x

def learn():

    f = pd.read_pickle("features_v2.pkl")
    f['Odds_1'] = 1 / f['Odds_1']
    f['Odds_2'] = 1 / f['Odds_2']
    f['Odds'] = f['Odds_1'] - f['Odds_2']
    # Remove backtest
    f = f[(f['Date'] < datetime(year=2019, month=1, day=1).date())]
    f = f.reset_index(drop=False)
    f['Y'] = 1
    f = build_variables(f)
    f = f[f['Date'] < datetime(year=2019, month=1, day=1).date()]
    x = f[['FS','W1SP','W2SP','WSP','WRP','TPW','ACES','DF','BP','COMPLETE','SERVEADV','DIRECT', 'RETIRED', 'FATIGUE', 'Odds']]
    y = f[['Y']]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100000)
    now = datetime.now()
    checkpoint_path = "Models/model_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + ".h5"

    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=300, mode='min', min_delta=0.0001),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    ]

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train.shape[1],)),
        keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(const.L_SIZE, activation=tf.nn.sigmoid),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC'])

    # history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10000, batch_size=8192, callbacks=[es])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100000, batch_size=8192, callbacks=keras_callbacks)

    # plot training history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    save(model)

    return model


def save(model):
    # serialize model to JSON
    now = datetime.now()
    model_json = model.to_json()
    with open("Models/model_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("Models/model_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "_" + str(now.minute) + ".h5")
    print("Saved model to disk")


def load(model):
    # load json and create model
    json_file = open("Models/" + str(model) + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Models/" + str(model) + ".h5")
    print("Loaded model from disk")
    return loaded_model


def profitability(model):
    model = load("model_2019_10_20_14_50")
    f = pd.read_pickle("features_v2.pkl")
    f['Odds_1_prob'] = 1 / f['Odds_1']
    f['Odds_2_prob'] = 1 / f['Odds_2']
    f['Odds'] = f['Odds_1_prob'] - f['Odds_2_prob']
    # Remove backtest
    f = f[(f['Date'] >= datetime(year=2019, month=1, day=1).date())]
    # f = f[(f['Date'] < datetime(year=2019, month=1, day=2).date())]
    f = f.reset_index(drop=True)
    f['Y'] = 1
    f = build_variables(f).reset_index(drop=True)
    x = f[['FS','W1SP','W2SP','WSP','WRP','TPW','ACES','DF','BP','COMPLETE','SERVEADV','DIRECT', 'RETIRED', 'FATIGUE', 'Odds']].reset_index(drop=True)

    f['Y_hat'] = pd.DataFrame(model.predict(x))
    f['diff'] = np.where(f['Y']==1, f['Y_hat'] - f['Odds_1_prob'], f['Y_hat'] - f['Odds_2_prob'])
    f['odds@bet'] = np.where(f['Y']==1, f['Odds_1'], f['Odds_2'])

    pl = pd.read_pickle("players.pkl")
    f = pd.merge(f, pl, 'inner', left_on=['PlayerID_1'], right_on=['PlayerID'])
    f = pd.merge(f, pl, 'inner', left_on=['PlayerID_2'], right_on=['PlayerID'])

    threshold_diff = 0.05
    threshold_yhat = 0.4
    bets = f[(f['diff'] > threshold_diff) & (f['Y_hat'] > threshold_yhat)].reset_index(drop=True)
    # bets = f[(f['diff'] > threshold_diff)].reset_index(drop=True)

    bets = bets.sort_values(by=['Date']).reset_index(drop=True)

    # Calculate balance
    bets['balance_Fixed'] = 0.0
    bets['balance_Prop'] = 0.0
    bets = bets.sort_values(by=['Date']).reset_index(drop=True)
    for i, r in bets.iterrows():
        if i == 0:
            prevBalanceFixed = 1000.0
            prevBalanceProp = 1000.0
        else:
            prevBalanceFixed = bets['balance_Fixed'].iat[i - 1]
            prevBalanceProp = bets['balance_Prop'].iat[i - 1]
        if bets['Y'].iat[i] == 1:
            bets['balance_Fixed'].iat[i] = prevBalanceFixed + 50 * ((bets['odds@bet'].iat[i]) - 1.0)
            bets['balance_Prop'].iat[i] = prevBalanceProp + (prevBalanceProp * 0.05) * ((bets['odds@bet'].iat[i]) - 1.0)
        else:
            bets['balance_Fixed'].iat[i] = prevBalanceFixed - 50
            bets['balance_Prop'].iat[i] = prevBalanceProp - (prevBalanceProp * 0.05)

    count = 0
    maxCount = 0
    # Longest lose streak
    for i, r in bets.iterrows():
        x = r['Y']
        if x == 0:
            count += 1
            if count > maxCount:
                maxCount = count
                print("maxCount:" + str(maxCount) + "Date:" + str(r['Date']))
        else:
            count = 0




    # months = bets['Date'].groupby([pd.to_datetime(bets['Date']).dt.months]).agg('count')
    # months.mean()

    # ROI graph
    graph = pd.DataFrame(columns=['Diff', 'Yhat', 'ROI'])
    k = 0
    for i in range(0, 12):
        threshold_diff = 0.0 + i * 0.01
        for j in range(0, 5):
            df = f
            threshold_yhat = 0.5 + j * 0.1
            df = df[(df['diff'] > threshold_diff) & (df['Y_hat'] > threshold_yhat)].reset_index(drop=True)
            df['Profit'] = df['Y'] * df['odds@bet'] - 1
            roi = df['Profit'].sum() / df['Profit'].count()
            temp = [threshold_diff, threshold_yhat, roi]
            graph.loc[k] = temp
            k += 1

    X = graph['Diff']
    Y = graph['Yhat']
    # X, Y = np.meshgrid(X, Y)
    Z = graph['ROI']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel('diff')
    ax.set_ylabel('yhat')
    ax.set_zlabel('ROI')
    plt.show()




def bets():
    print('Updating...')
    db.update()
    now = datetime.now()
    date = datetime(year=now.year, month=now.month, day=now.day).date()
    features = pd.read_pickle("Matches/matches_" + str(now.year) + "_" + str(now.month) + '_' + str(now.day) + ".pkl")
    features1 = features[['PlayerID_2', 'PlayerID_1', 'TournamentID', 'SurfaceID', 'Date', 'Odds_2', 'Odds_1']]
    features1.columns = ['PlayerID_1', 'PlayerID_2', 'TournamentID', 'SurfaceID', 'Date', 'Odds_1', 'Odds_2']
    features = features.append(features1, sort=False).reset_index(drop=True)
    print('Creating features...')
    f = db.new_games(features)
    print('Cleaning...')
    f2 = pd.read_pickle("features_v2.pkl")
    f = f.append(f2, sort=False)
    f['Odds_1_prob'] = 1 / f['Odds_1']
    f['Odds_2_prob'] = 1 / f['Odds_2']
    f['Odds'] = f['Odds_1_prob'] - f['Odds_2_prob']
    f = f.drop(['TournamentID'], axis=1)
    # f = f.drop(['SurfaceID'], axis=1)
    f = f.reset_index(drop=True)
    f = cl.post_clean(f)
    f = f[f['Date'] >= date]
    f = f[np.isfinite(f['SurfaceID'])]
    print('Evaluating...')
    if f.empty:
        print('No games above uncertainty threshold')
        return f
    else:
        x = f[['FS', 'W1SP', 'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'BP', 'COMPLETE', 'SERVEADV', 'DIRECT', 'RETIRED', 'FATIGUE', 'Odds']].reset_index(drop=True)
        f = f.reset_index(drop=True)
        model = load("model_2019_10_20_14_50")
        f['Y_hat'] = pd.DataFrame(model.predict(x))
        f['diff'] = f['Y_hat'] - f['Odds_1_prob']
        return f


def save_to_csv():
    start = time.time()
    print("__________________________________")
    print("Starting job")
    f = bets()
    threshold_diff = 0.05
    threshold_yhat = 0.4
    f = f[(f['diff'] > threshold_diff) & (f['Y_hat'] > threshold_yhat)].reset_index(drop=True)
    if f.empty:
        # TODO add check in csv
        print("No bets")
    else:
        prevBets = pd.read_csv('bets.csv', header=0, chunksize=100000)
        chunks = []
        for chunk in prevBets:
            chunks.append(chunk)
        f = f.append(chunks, sort=False)
        f.to_csv('bets.csv', encoding='utf-8')
        print("New bets!")


    end = time.time()
    print('Time: ' + str(end - start) + 's')
    print("__________________________________")
    return f


def manual_calc(f):
    f = bets()
    pl = pd.read_pickle("players.pkl")
    df = pd.merge(f, pl, 'inner', left_on=['PlayerID_1'], right_on=['PlayerID'])
    df = pd.merge(df, pl, 'inner', left_on=['PlayerID_2'], right_on=['PlayerID'])
    df = df[['PlayerID_1', 'PlayerID_2', 'SurfaceID', 'Date', 'Name_x','Name_y', 'Odds_1', 'Odds_2', 'FS', 'W1SP',
       'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'BP', 'COMPLETE', 'SERVEADV',
       'DIRECT', 'RETIRED', 'FATIGUE', 'UNC', 'Odds_1_prob', 'Odds_2_prob', 'Odds', 'Y_hat', 'diff']]
    df.columns = ['PlayerID_1', 'PlayerID_2', 'SurfaceID', 'Date','Name_1','Name_2', 'Odds_1', 'Odds_2', 'FS', 'W1SP',
       'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'BP', 'COMPLETE', 'SERVEADV',
       'DIRECT', 'RETIRED', 'FATIGUE', 'UNC', 'Odds_1_prob', 'Odds_2_prob', 'Odds', 'Y_hat', 'diff']
    df.to_csv("manualCalc.csv")
    f = pd.read_csv("manualCalc.csv")
    f['Odds_1_prob'] = 1 / f['Odds_1']
    f['Odds_2_prob'] = 1 / f['Odds_2']
    f['Odds'] = f['Odds_1_prob'] - f['Odds_2_prob']
    x = f[['FS', 'W1SP', 'W2SP', 'WSP', 'WRP', 'TPW', 'ACES', 'DF', 'BP', 'COMPLETE', 'SERVEADV', 'DIRECT', 'RETIRED', 'FATIGUE',
           'Odds']].reset_index(drop=True)
    f = f.reset_index(drop=True)
    model = load("model_2019_10_20_14_50")
    f['Y_hat'] = pd.DataFrame(model.predict(x))
    f['diff'] = f['Y_hat'] - f['Odds_1_prob']
    threshold_diff = 0.05
    threshold_yhat = 0.4
    df = f[(f['diff'] > threshold_diff) & (f['Y_hat'] > threshold_yhat)].reset_index(drop=True)



def main():
    f = save_to_csv()



# TODO: Check that reversal in bets() recalibrates the features
# TODO: max date in updateFeatures()


