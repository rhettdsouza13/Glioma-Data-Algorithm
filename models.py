from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from parser_enc import *
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as pl
import numpy
from functions import element_wise_add
from scipy import interp

set_br = 40
set_br_SVM = 40
set_br_DT = 40
set_br_LR = 40
n_runs = 15

metrics = {'ann' : [0 for i in xrange(6)], 'svm' : [0 for i in xrange(6)], 'dt' : [0 for i in xrange(6)], 'lr' : [0 for i in xrange(6)]}
rocs = {'ann' : [[],[]], 'svm' : [[],[]] , 'dt' : [[],[]], 'lr' : [[],[]]}
roc_pl = {'ann' : [[],[]], 'svm' : [[],[]] , 'dt' : [[],[]], 'lr' : [[],[]]}

mean_fpr = numpy.linspace(0,1,101)

for i in xrange(n_runs):

    inputs, labels = input_inject_GLIOMA_MLP()

    print "\nANN\n"
    estimator = MLPClassifier(hidden_layer_sizes = (100,200,), batch_size = 5, max_iter=1000, solver='adam')
    estimator.fit(inputs[:set_br], labels[:set_br])
    # print "Trained"
    y_score = estimator.predict_proba(inputs[set_br:])

    # print y_score
    # print labels
    test_set = numpy.array(labels[set_br:])
    fpr, tpr, _ = roc_curve(test_set[:,1], y_score[:,1])
    tpr_m = interp(mean_fpr, fpr, tpr)
    tpr_m[0] = 0.0

    roc_auc = auc(fpr, tpr)
    # print fpr
    # print tpr
    print "AUC " + str(roc_auc)
    # ann, = pl.plot(fpr, tpr)

    #
    # print inputs[60:]
    # print labels[60:]
    predicted = estimator.predict(inputs[set_br:])
    # print predicted
    correct_1 = 0
    correct_0 = 0
    wrong_1 = 0
    wrong_0 = 0
    correct = 0
    for pred,lab in zip(predicted,labels[set_br:]):
        if pred[0] == lab[0] and pred[1] == lab[1]:
            correct += 1
        if pred[0] == lab[0] and pred[1] == lab[1] and lab[0] == 0 :
            correct_0 += 1
        elif pred[0] == lab[0] and pred[1] == lab[1] and lab[0] == 1 :
            correct_1 += 1
        elif pred[0] != lab[0] and pred[1] != lab[1] and lab[0] == 1 :
            wrong_1 += 1
        elif pred[0] != lab[0] and pred[1] != lab[1] and lab[0] == 0 :
            wrong_0 += 1
        else:
            continue
    accuracy = float(float(correct)/len(labels[set_br:]))
    print accuracy
    print correct_1 + correct_0 + wrong_0 + wrong_1
    print correct_1
    print correct_0
    print wrong_1
    print wrong_0
    metrics['ann'][0] += accuracy
    metrics['ann'][1] += correct_1
    metrics['ann'][2] += correct_0
    metrics['ann'][3] += wrong_0
    metrics['ann'][4] += wrong_1
    metrics['ann'][5] += roc_auc
    # rocs['ann'][0].append(mean_fpr)
    rocs['ann'][1].append(tpr_m)

    # print estimator.score(inputs[60:], labels[60:])
    # print labels[set_br:]
    # print accuracy_score(labels[set_br:], predicted.argmax(axis=1))

    print "\nSVM\n"

    clf = SVC(kernel='rbf', C=100.0, gamma=0.1)
    labels_SVM = [0 if i[0] == 0 else 1 for i in labels]
    # print labels_SVM, len(labels_SVM)
    clf.fit(inputs[:set_br_SVM], labels_SVM[:set_br_SVM])
    y_score = clf.decision_function(inputs[set_br_SVM:])
    # print y_score
    fpr, tpr, _ = roc_curve(labels_SVM[set_br_SVM:], y_score)
    tpr_m = interp(mean_fpr, fpr, tpr)
    tpr_m[0] = 0.0
    roc_auc = auc(fpr, tpr)
    print "AUC " + str(roc_auc)
    # svm, = pl.plot(fpr, tpr)

    # print fpr
    # print tpr
    predicted_sv = clf.predict(inputs[set_br_SVM:])
    # print predicted_sv

    correct_1 = 0
    correct_0 = 0
    wrong_1 = 0
    wrong_0 = 0
    correct = 0
    for pred,lab in zip(predicted_sv, labels_SVM[set_br_SVM:]):
        if pred == lab:
            correct += 1
        if pred == lab and lab == 0 :
            correct_0 += 1
        elif pred == lab and lab == 1 :
            correct_1 += 1
        elif pred != lab and lab == 1 :
            wrong_1 += 1
        elif pred != lab and lab == 0 :
            wrong_0 += 1
        else:
            continue
    accuracy_sv = float(float(correct)/len(labels_SVM[set_br_SVM:]))
    print accuracy_sv
    print correct_1 + correct_0 + wrong_0 + wrong_1
    print correct_1
    print correct_0
    print wrong_1
    print wrong_0
    metrics['svm'][0] += accuracy_sv
    metrics['svm'][1] += correct_1
    metrics['svm'][2] += correct_0
    metrics['svm'][3] += wrong_0
    metrics['svm'][4] += wrong_1
    metrics['svm'][5] += roc_auc
    # rocs['svm'][0].append(mean_fpr)
    rocs['svm'][1].append(tpr_m)


    print "\nDT\n"

    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(inputs[:set_br_DT],labels_SVM[:set_br_DT])

    predicted_dt = classifier.predict(inputs[set_br_DT:])
    y_score = classifier.predict_proba(inputs[set_br_DT:])

    # print y_score

    fpr, tpr, _ = roc_curve(labels_SVM[set_br_SVM:], y_score[:,1])
    tpr_m = interp(mean_fpr, fpr, tpr)
    tpr_m[0] = 0.0
    roc_auc = auc(fpr, tpr)
    # print fpr
    # print tpr
    print "AUC " + str(roc_auc)
    # dt, = pl.plot(fpr, tpr)
    # print predicted_dt

    correct_1 = 0
    correct_0 = 0
    wrong_1 = 0
    wrong_0 = 0
    correct = 0
    for pred,lab in zip(predicted_dt, labels_SVM[set_br_DT:]):
        if pred == lab:
            correct += 1
        if pred == lab and lab == 0 :
            correct_0 += 1
        elif pred == lab and lab == 1 :
            correct_1 += 1
        elif pred != lab and lab == 1 :
            wrong_1 += 1
        elif pred != lab and lab == 0 :
            wrong_0 += 1
        else:
            continue

    accuracy_dt = float(float(correct)/len(labels_SVM[set_br_DT:]))
    print accuracy_dt
    print correct_1 + correct_0 + wrong_0 + wrong_1
    print correct_1
    print correct_0
    print wrong_1
    print wrong_0

    metrics['dt'][0] += accuracy_dt
    metrics['dt'][1] += correct_1
    metrics['dt'][2] += correct_0
    metrics['dt'][3] += wrong_0
    metrics['dt'][4] += wrong_1
    metrics['dt'][5] += roc_auc
    # rocs['dt'][0].append(mean_fpr)
    rocs['dt'][1].append(tpr_m)

    print "\nLR\n"

    classifier = LogisticRegression(C=150.0)
    classifier.fit(inputs[:set_br_LR],labels_SVM[:set_br_LR])
    predicted_lr = classifier.predict(inputs[set_br_LR:])
    # print predicted_lr
    predicted_dt = classifier.predict(inputs[set_br_LR:])
    y_score = classifier.predict_proba(inputs[set_br_LR:])

    # print y_score

    fpr, tpr, _ = roc_curve(labels_SVM[set_br_LR:], y_score[:,1])
    tpr_m = interp(mean_fpr, fpr, tpr)
    tpr_m[0] = 0.0
    roc_auc = auc(fpr, tpr)
    # print fpr
    # print tpr
    print "AUC " + str(roc_auc)
    # lr, = pl.plot(fpr, tpr)

    correct_1 = 0
    correct_0 = 0
    wrong_1 = 0
    wrong_0 = 0
    correct = 0
    for pred,lab in zip(predicted_lr, labels_SVM[set_br_LR:]):
        if pred == lab:
            correct += 1
        if pred == lab and lab == 0 :
            correct_0 += 1
        elif pred == lab and lab == 1 :
            correct_1 += 1
        elif pred != lab and lab == 1 :
            wrong_1 += 1
        elif pred != lab and lab == 0 :
            wrong_0 += 1
        else:
            continue

    accuracy_lr = float(float(correct)/len(labels_SVM[set_br_LR:]))
    print accuracy_lr
    print correct_1 + correct_0 + wrong_0 + wrong_1
    print correct_1
    print correct_0
    print wrong_1
    print wrong_0

    metrics['lr'][0] += accuracy_lr
    metrics['lr'][1] += correct_1
    metrics['lr'][2] += correct_0
    metrics['lr'][3] += wrong_0
    metrics['lr'][4] += wrong_1
    metrics['lr'][5] += roc_auc
    # rocs['lr'][0].append(mean_fpr)
    rocs['lr'][1].append(tpr_m)




metrics['ann'][:] = [x/n_runs for x in metrics['ann']]
metrics['dt'][:] = [x/n_runs for x in metrics['dt']]
metrics['svm'][:] = [x/n_runs for x in metrics['svm']]
metrics['lr'][:] = [x/n_runs for x in metrics['lr']]


rocs['ann'][1] = numpy.mean(rocs['ann'][1], axis=0)
rocs['ann'][1][-1] = 1.0
rocs['dt'][1] = numpy.mean(rocs['dt'][1], axis=0)
rocs['dt'][1][-1] = 1.0
rocs['svm'][1] = numpy.mean(rocs['svm'][1], axis=0)
rocs['svm'][1][-1] = 1.0
rocs['lr'][1] = numpy.mean(rocs['lr'][1], axis=0)
rocs['lr'][1][-1] = 1.0


print metrics
print rocs

ann, = pl.plot(mean_fpr, rocs['ann'][1])
dt, = pl.plot(mean_fpr, rocs['dt'][1])
svm, = pl.plot(mean_fpr, rocs['svm'][1])
lr, = pl.plot(mean_fpr, rocs['lr'][1])

metrics['ann'][5] = auc(mean_fpr, rocs['ann'][1])
metrics['dt'][5] = auc(mean_fpr,rocs['dt'][1])
metrics['svm'][5] = auc(mean_fpr,rocs['svm'][1])
metrics['lr'][5] = auc(mean_fpr,rocs['lr'][1])

pl.plot([0,1],[0,1], linestyle='--')

fig = pl.gcf()
fig.set_size_inches(10,9)
pl.legend([ann,svm,dt,lr], ["Artificial Neural Network", "Support Vector Machine", "Decision Tree Classifier", "Logistic Regression"], loc=4)
pl.title("Receiver Operating Curves for Models")
pl.xlabel("1-Specificity")
pl.ylabel("Sensitivity")
pl.savefig('roc3.png')
# pl.show()
