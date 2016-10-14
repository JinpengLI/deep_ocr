from sklearn import datasets
digits = datasets.load_digits()
print digits.data.shape
digits.images[0]

digits.target
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1]) 
clf.predict(digits.data[-1:])



import matplotlib.pyplot as plt
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.data[-1].reshape((8,8)), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

digits.target
