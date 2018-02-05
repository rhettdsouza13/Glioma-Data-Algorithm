from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from parser_enc import *
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

inputs, labels = input_inject_GLIOMA_MLP()
set_br = 45
set_br_SVM = 45
set_br_DT = 50

estimator = MLPClassifier(hidden_layer_sizes = (100), batch_size = 5, max_iter=1000, solver='adam')
estimator.fit(inputs[:set_br], labels[:set_br])
print "Trained"
#
# print inputs[60:]
# print labels[60:]
predicted = estimator.predict(inputs[set_br:])
print predicted
correct = 0
for pred,lab in zip(predicted,labels[set_br:]):
    if pred[0] == lab[0] and pred[1] == lab[1]:
        correct += 1
    else:
        continue
accuracy = float(float(correct)/len(labels[set_br:]))
print accuracy
# print estimator.score(inputs[60:], labels[60:])
# print labels[set_br:]
# print accuracy_score(labels[set_br:], predicted.argmax(axis=1))


clf = SVC(kernel='rbf')
labels_SVM = [0 if i[0] == 0 else 1 for i in labels]
print labels_SVM, len(labels_SVM)
clf.fit(inputs[:set_br_SVM], labels_SVM[:set_br_SVM])

predicted_sv = clf.predict(inputs[set_br_SVM:])
print predicted_sv

correct = 0
for pred,lab in zip(predicted_sv, labels_SVM[set_br_SVM:]):
    if pred == lab:
        correct += 1
    else:
        continue
accuracy_sv = float(float(correct)/len(labels_SVM[set_br_SVM:]))
print accuracy_sv


classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(inputs[:set_br_DT],labels_SVM[:set_br_DT])
predicted_dt = classifier.predict(inputs[set_br_DT:])
print predicted_dt

correct = 0
for pred,lab in zip(predicted_dt, labels_SVM[set_br_DT:]):
    if pred == lab:
        correct += 1
    else:
        continue

accuracy_dt = float(float(correct)/len(labels_SVM[set_br_DT:]))
print accuracy_dt
