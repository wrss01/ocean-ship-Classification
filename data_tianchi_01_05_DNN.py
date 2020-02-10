
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add
from keras.callbacks import Callback, EarlyStopping
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from collections import defaultdict
from sklearn import metrics

all_df=pd.read_csv('data_train_o_02.csv',index_col='tt')

all_df = all_df.pivot_table(values=['x', 'y','v','d'], index=['id', 'ty'],columns=['t'],fill_value=0.0)

all_df=all_df.reset_index()
d = defaultdict(LabelEncoder)
df = all_df[['ty']].apply(lambda x:d[x.name].fit_transform(x) if type(x[0]) is str or math.isnan(x[0])  else x)
scaler = StandardScaler()
x_train= scaler.fit_transform(all_df[['x','y','v']]) #,'d']])
#x_train = df.loc[:,names_x].values
y_train = df[['ty']].values

from keras.utils import to_categorical
y_train_c = to_categorical(y_train)

def NN_model():
    init = keras.initializers.glorot_uniform(seed=2)
    model = keras.models.Sequential()
    model.add(Dense(units=4096, input_dim=x_train.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(units=2048, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=2048, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=1024, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=1024, kernel_initializer=init, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=512, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=512, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=256, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=256, kernel_initializer=init, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=128, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=128, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=64, kernel_initializer=init, activation='relu'))
    model.add(Dense(units=64, kernel_initializer=init, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=32, kernel_initializer=init, activation='relu'))
    #model.add(Dense(units=32, kernel_initializer=init, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=32, kernel_initializer=init, activation='softplus')) #softplus
    model.add(Dense(units=3, kernel_initializer=init, activation='softmax'))
    
    return model

class Metric(Callback):
    def __init__(self, model, callbacks, data):
        super().__init__()
        self.model = model
        self.callbacks = callbacks
        self.data = data

    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_end(self, batch, logs=None):
        X_train, y_train = self.data[0][0], self.data[0][1]
        y_pred3 = self.model.predict(X_train)
        y_pred = np.zeros((len(y_pred3), ))
        y_true = np.zeros((len(y_pred3), ))
        for i in range(len(y_pred3)):
            y_pred[i] = list(y_pred3[i]).index(max(y_pred3[i]))
        for i in range(len(y_pred3)):
            y_true[i] = list(y_train[i]).index(max(y_train[i]))
        trn_s = f1_score(y_true, y_pred, average='macro')
        logs['trn_score'] = trn_s
        
        X_val, y_val = self.data[1][0], self.data[1][1]
        y_pred3 = self.model.predict(X_val)
        y_pred = np.zeros((len(y_pred3), ))
        y_true = np.zeros((len(y_pred3), ))
        for i in range(len(y_pred3)):
            y_pred[i] = list(y_pred3[i]).index(max(y_pred3[i]))
        for i in range(len(y_pred3)):
            y_true[i] = list(y_val[i]).index(max(y_val[i]))
        val_s = f1_score(y_true, y_pred, average='macro')
        logs['val_score'] = val_s
        print('trn_score', trn_s, 'val_score', val_s)

        for callback in self.callbacks:
            callback.on_epoch_end(batch, logs)


n_splits = 7
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20200203)

b_size = 150
max_epochs = 1000
models = []
for fold, (trn_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
    print('fold:', fold)
    X_train, Y_train = x_train[trn_idx], y_train_c[trn_idx] #df[names_x].loc[trn_idx], df[names_y].loc[trn_idx] # x_train[trn_idx], y_train[trn_idx]
    X_val, Y_val = x_train[val_idx], y_train_c[val_idx]
    Y_val_o=y_train[val_idx]
    
    model = NN_model()
    simple_adam = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
    es = EarlyStopping(monitor='val_score', patience=1000, verbose=1, mode='max')#, restore_best_weights=True,)
    es.set_model(model)
    metric = Metric(model, [es], [(X_train, Y_train), (X_val, Y_val)])
    model.fit(X_train, Y_train, batch_size=b_size, epochs=max_epochs, 
              validation_data = [X_val, Y_val],
              callbacks=[metric], shuffle=True, verbose=1)
    
    models.append(model)
    test_pred = model.predict(X_val)
    test_pred_v = np.argmax(test_pred, axis=1)
    score_ = metrics.f1_score(Y_val_o, test_pred_v, average='macro')
    print('*'*20)
    print('total score is : {}'.format(score_))
    print('*'*20)
    break
