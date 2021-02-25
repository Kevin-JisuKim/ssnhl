import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.model_selection import KFold
from model import ensemble_models, main_model
from data_loader import load_dataset
from stacking_ensemble import ensemble
from predict import predict

if __name__ == '__main__':
    train_set, train_label, test_set = load_dataset()

    models = ensemble_models()
    new_train_set, new_test_set = ensemble(models, train_set, train_label, test_set)

    main_model = main_model()
    predict(main_model, train_label, new_train_set, new_test_set)