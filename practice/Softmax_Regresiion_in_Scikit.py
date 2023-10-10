import numpy as np
from sklearn.linear_model import LogisticRegression


x_train = np.array([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])

y_train = np.array([2,2,2,1,1,1,0,0])

model = LogisticRegression(penalty='none')

model.fit(x_train, y_train)

x_test = np.array([[1,11,10,9], [1,3,4,3], [1,1,0,1]])

print(model.predict(x_test))