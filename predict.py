import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras.callbacks import EarlyStopping
from setting import setting
from data_loader import load_dataset
from model import main_model

lr, epoch, batch, dropout, gpu_num = setting()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
earlystopping = EarlyStopping(monitor='accuracy', patience=2000)

def predict(main_model, train_label, new_train_set, new_test_set):
    opt = keras.optimizers.Adam(learning_rate=lr)
    main_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    main_model.fit(new_train_set, train_label, epochs=epoch * 10, batch_size=batch, callbacks=earlystopping, verbose=1)
    prediction = model.predict(new_test_set)

    result_name = str(lr) + '_' + str(epoch) + '_' + str(batch) + '_' + str(dropout) + '.txt'

    f = open(result_name, 'w')

    for pred in prediction:
        print(pred[0], file=f)
    
    f.close()

if __name__ == '__main__':
    train_set, train_label, test_set = load_dataset()
    main_model = main_model()
    new_train_set = np.loadtxt("new_train_set.txt", delimiter=' ', dtype=np.float32)
    new_test_set = np.loadtxt("new_test_set.txt", delimiter=' ', dtype=np.float32)

    opt = keras.optimizers.Adam(learning_rate=lr)
    main_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    main_model.fit(new_train_set, train_label, epochs=epoch * 10, batch_size=batch, callbacks=earlystopping, verbose=1)
    prediction = model.predict(new_test_set)

    result_name = str(lr) + '_' + str(epoch) + '_' + str(batch) + '_' + str(dropout) + '.txt'

    f = open(result_name, 'w')

    for pred in prediction:
        print(pred[0], file=f)
    
    f.close()