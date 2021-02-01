import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.model_selection import KFold

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def load_dataset():
    dataset = pd.read_excel('./SSNHL.xlsx', sheet_name='SSNHL_REFINE')
    label = dataset['(Label)']
    del dataset['(in1)']
    del dataset['(in15)']
    del dataset['(in16)']
    del dataset['(Label)']

    dataset = dataset.values
    label = label.values

    train_set = dataset[0:760]
    train_label = label[0:760]
    test_set = dataset[760:800]

    return train_set, train_label, test_set

def stacking_ensemble(model, train_set, train_label, test_set, n_folds=5):
    kfold = KFold(n_splits=n_folds, random_state=0)

    train_fold_predict = np.zeros((train_set.shape[0], 1))
    test_predict = np.zeros((test_set.shape[0], n_folds))

    for count, (train_index, val_index) in enumerate(kfold.split(train_set)):
        x_train = train_set[train_index]
        y_train = train_label[train_index]
        x_val = train_set[val_index]

        opt = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(x_train, y_train, epochs=100000, batch_size=256, verbose=2)
        train_fold_predict[val_index, :] = model.predict(x_val).reshape(-1, 1)
        test_predict[:, count] = model.predict(test_set)[:, 0]

    test_predict_mean = np.mean(test_predict, axis=1).reshape(-1, 1)

    return train_fold_predict, test_predict_mean

if __name__ == '__main__':
    train_set, train_label, test_set = load_dataset()

    model1 = keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model2 = keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model3 = keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model4 = keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model1_train, model1_test = stacking_ensemble(model1, train_set, train_label, test_set)
    model2_train, model2_test = stacking_ensemble(model2, train_set, train_label, test_set)
    model3_train, model3_test = stacking_ensemble(model3, train_set, train_label, test_set)
    model4_train, model4_test = stacking_ensemble(model4, train_set, train_label, test_set)

    new_train_set = np.concatenate((model1_train, model2_train, model3_train, model4_train), axis=1)
    new_test_set = np.concatenate((model1_test, model2_test, model3_test, model4_test), axis=1)

    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    opt = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
    model.fit(new_train_set, train_label, epochs=100000, batch_size=256, verbose=2)
    prediction = model.predict(new_test_set)

    f = open('prediction.txt', 'w')

    for pred in prediction:
        print(pred[0], file=f)
    
    f.close()