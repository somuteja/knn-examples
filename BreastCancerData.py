#knn algorithm on breast cancer data for classifying into benign and malignant tumour

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
cancer = load_breast_cancer()

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,
                                                 stratify = cancer.target, random_state=12)

training_accuracy = []
testing_accuracy = []
neighbor_settings = range(1,11)

for n_neighbors in neighbor_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)
    training_accuracy.append(knn.score(X_train,y_train))
    testing_accuracy.append(knn.score(X_test,y_test))

#for plotting testing and training accuracies wrt to number of neighbors    
plt.plot(neighbor_settings, training_accuracy, label = "training_accuracy")
plt.plot(neighbor_settings, testing_accuracy, label = "testing_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()    
