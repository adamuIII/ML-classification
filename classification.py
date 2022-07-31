from __future__ import absolute_import, division, print_function, unicode_literals

from numpy.random import RandomState
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# Załadowanie pliku z csv
df = pd.read_csv('C:/Users/adamr/Desktop/ML/2.classification/breast-cancer.csv')
rng = RandomState()
# podzielenie pliku na dane uczace i testowe 
train = df.sample(frac=0.7, random_state=rng)
test = df.loc[~df.index.isin(train.index)]


#zamiana zmiennych na numery
print(train.dtypes)
print(train["age"].value_counts())
##wyciagamy jedna zmienna i zamieniamy
train.age = le.fit_transform(train.age)
train.menopause = le.fit_transform(train.menopause)
train.recurrence = le.fit_transform(train.recurrence)
train.tumor_size = le.fit_transform(train.tumor_size)
train.inv_nodes = le.fit_transform(train.inv_nodes)
train.node_caps = le.fit_transform(train.node_caps)
train.deg_caps = le.fit_transform(train.deg_caps)
train.deg_malig = le.fit_transform(train.deg_malig)
train.breast = le.fit_transform(train.breast)
train.breast_quad = le.fit_transform(train.breast_quad)

test.age = le.fit_transform(test.age)
test.menopause = le.fit_transform(test.menopause)
test.recurrence = le.fit_transform(test.recurrence)
test.tumor_size = le.fit_transform(test.tumor_size)
test.inv_nodes = le.fit_transform(test.inv_nodes)
test.node_caps = le.fit_transform(test.node_caps)
test.deg_caps = le.fit_transform(test.deg_caps)
test.breast = le.fit_transform(test.breast)
test.breast_quad = le.fit_transform(test.breast_quad)
test.deg_malig = le.fit_transform(test.deg_malig)


COLUMN_NAMES = ['recurrence','age','menopause','tumor_size','inv_nodes','node_caps','deg_caps','deg_malig','breast','breast_quad']
RECURRENCE = ['no-recurrence-events','recurrence-events']

print(type(train))

#wyrzucenie zmiennej y do osobnej zmiennej
train_y = train.pop('recurrence')
test_y = test.pop('recurrence')
train.shape

print(train.age)



#funkcja wejsciowa
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)




# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)
#

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3)


classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)


eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))




def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['age','menopause','tumor_size','inv_nodes','node_caps','deg_caps','deg_malig','breast','breast_quad']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    val = input(feature + ": ")
    

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        RECURRENCE[class_id], 100 * probability))
