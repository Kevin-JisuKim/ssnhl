import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from setting import setting
from data_loader import load_dataset
from model import ensemble_models

lr, epoch, batch, dropout, gpu_num = setting()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
earlystopping = EarlyStopping(monitor='accuracy', patience=2000)

def stacking(model, train_set, train_label, test_set, n_folds=5):
    kfold = KFold(n_splits=n_folds, shuffle=True ,random_state=0)

    train_fold_predict = np.zeros((train_set.shape[0], 1))
    test_predict = np.zeros((test_set.shape[0], n_folds))

    for count, (train_index, val_index) in enumerate(kfold.split(train_set)):
        x_train = train_set[train_index]
        y_train = train_label[train_index]
        x_val = train_set[val_index]

        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch, callbacks=earlystopping, verbose=1)
        train_fold_predict[val_index, :] = model.predict(x_val).reshape(-1, 1)
        test_predict[:, count] = model.predict(test_set)[:, 0]

    test_predict_mean = np.mean(test_predict, axis=1).reshape(-1, 1)

    return train_fold_predict, test_predict_mean

def ensemble(models, train_set, train_label, test_set):
    model_train = []
    model_test = []

    for model in models:
        m_train, m_test = stacking(model, train_set, train_label, test_set)
        model_train.append(m_train)
        model_test.append(m_test)

    new_train_set = np.concatenate(model_train[:], axis=1)
    new_test_set = np.concatenate(model_test[:], axis=1)

    np.savetxt("new_train_set.txt", new_train_set, fmt='%f', delimiter=' ')
    np.savetxt("new_test_set.txt", new_train_set, fmt='%f', delimiter=' ')

    return new_train_set, new_test_set

if __name__ == '__main__':
    train_set, train_label, test_set = load_dataset()
    models = ensemble_models()
    new_train_set, new_test_set = ensemble(models, train_set, train_label, test_set)