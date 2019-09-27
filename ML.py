import cleaning as cl
from collections import OrderedDict
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import const
import numpy as np
import datetime

def build_variables():
    f = pd.read_pickle("features.pkl")
    f = cl.post_clean(f)
    f = f.reset_index(drop=True)
    f['Y'] = 1
    x = f

    rows_list = []
    for i, r in x.iterrows():
        dict = OrderedDict()
        dict.update(r)
        rows_list.append(dict)
        if i % 100 == 0:
            print(i)

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
    rows_list['Y'] = 0
    f = f.append(rows_list, sort=False)
    return f

f = build_variables()

x = f[['FS','W1SP','W2SP','WSP','WRP','TPW','ACES','DF','BP','COMPLETE','SERVEADV','DIRECT']]
# x = f[['FS','W1SP','W2SP','WSP','WRP','TPW','ACES','DF','BP','COMPLETE','SERVEADV']]
y = f[['Y']]
X_train, X_1, Y_train, Y_1 = train_test_split(x, y, test_size=0.4, random_state=0, shuffle=True)
X_val, X_test, Y_val, Y_test = train_test_split(X_1, Y_1, test_size=0.5, random_state=0, shuffle=True)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(x.shape[1],)),
    keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
	keras.layers.Dense(const.L_SIZE, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=1)
val_loss, val_acc = model.evaluate(X_val, Y_val)
test_loss, test_acc = model.evaluate(X_test, Y_test)
now = datetime.datetime.now()

# serialize model to JSON
model_json = model.to_json()
with open("model_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + ":" + str(now.minute) + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_" + str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + ":" + str(now.minute) + ".h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

Y_hat = pd.DataFrame(loaded_model.predict(X_test)).reset_index(drop=True)
Y_real = Y_test.reset_index(drop=True)
comparison = Y_real.join(Y_hat)
comparison.columns = ['Y_real', 'Y_hat']
comparison['Y_bet'] = np.where(comparison['Y_hat'] < 0.5, 0, 1)
comparison['Win'] = np.where((comparison['Y_real'] - comparison['Y_bet']) == 0, 1, 0)
winrate = comparison['Win'].sum() / comparison['Win'].count()


