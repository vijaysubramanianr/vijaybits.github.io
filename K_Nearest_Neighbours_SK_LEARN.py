import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df1 = pd.read_csv('breast-cancer-wisconsin.data')
df1.replace('?', -99999, inplace=True)
df1.replace(Nan, 000, inplace=True)
'''
id,clump_thickness,uniform_cell_size,
uniform_cell_shape,marginal_adhesion,
single_epi_cell_size,bare_nuclei,bland_chromation,
normal_nucleoli,mitoses,class
'''
df1.drop(['id'],1, inplace=True)
#define features & Attributes
X = np.array(df1.drop(['class'], 1))
y = np.array(df1['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.3)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('accuracy: ', accuracy)

# example_measures = np.array([4,2,1,1,1,2,3,2,1])
# example_measures_1 = example_measures.reshape(1, -1)

example_measures_list = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1], [6,8,1,2,4,2,3,6,1]])
example_measures_2 = example_measures_list.reshape(len(example_measures_list), -1)

prediction = clf.predict(example_measures_2)
print(prediction)



















