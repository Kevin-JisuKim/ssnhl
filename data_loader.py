import pandas as pd

def load_dataset():
    dataset = pd.read_excel('./SSNHL.xlsx', sheet_name='SSNHL_INPUT')
    label = dataset['(Label)']
    del dataset['(in1)']
    del dataset['(Label)']

    dataset = dataset.values
    label = label.values

    train_set = dataset[0:760]
    train_label = label[0:760]
    test_set = dataset[760:800]

    return train_set, train_label, test_set