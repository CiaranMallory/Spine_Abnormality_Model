import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

df = pd.read_csv('Dataset_spine.csv')
df.replace('?', -99999, inplace=True)
df = df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Class_att']]
df = df.replace(to_replace='Abnormal', value=1)
df = df.replace(to_replace='Normal', value=2)
#print(df.head)

x = np.array(df.drop(['Class_att'], 1))
y = np.array(df['Class_att'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print("Accuracy: ", accuracy)

example_measures = np.array([60,22,35,44,97,-0.12,0.7,11,13.8,14.3])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
#print(prediction)

if (prediction == 1):
	print("Result: Abnormal Spine")

elif (prediction == 2):
	print("Result: Normal")

