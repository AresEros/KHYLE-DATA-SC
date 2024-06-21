#impoting neccessary libraries from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

#load the iris dataset
iris = load_iri()

x = iris.data #array of the data
y = iris.terget #arrat of the labels(i.e answer) of each data entry

#getting label name i.e the three flower species
y_names = iris.target_names

#taking ramdom indices to split the dataset into train and test
test_ids = np.random.permutation(len(x))

#splitting data and label into train and test
#keeping last 10 entries for testing, rest for traning

x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]

y_train = y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

#classifying using decision tree
clf = tree.DecissionTreeClassifier()

#training (fitting) the classifier with the training set
clf.fit(x_train, y_train)

#predictions on the test dataset
pred = clf.predict(x_test)

print(pred) #predicted labels i.e flower species
print (y_test) #actual labels
print((accuracy_score(pred, y_test))) * 100 # prediction accuracy 

#reference http://docs.python-guide.org/en/latest/scenerio/ml/ 
