import csv
import numpy
from datetime import datetime

raw_data_set = []
with open("Glioma Data.xlsx - Sheet1.csv", 'r') as g_d_csv:
    f_reader = csv.reader(g_d_csv, delimiter=' ', quotechar='|')
    for row in f_reader:
        raw_data_set.append(row[0].split(','))

data_set = []
for sample in raw_data_set:
    d1 = sample.pop(0)
    d2 = sample[0]
    d1 = datetime.strptime(d1, "%m/%d/%Y")
    d2 = datetime.strptime(d2, "%m/%d/%Y")
    sample[0] = abs((d2 - d1).days)/365


for i,sample in enumerate(raw_data_set):
    for j,feature in enumerate(sample):
        if feature == 'MALE':
            raw_data_set[i][j] = 1
            continue
        if feature == 'FEMALE':
            raw_data_set[i][j] = 0
            continue
        if feature == 'Y':
            raw_data_set[i][j] = 1
            continue
        if feature == 'N':
            raw_data_set[i][j] = 0
            continue
        if feature == '?':
            raw_data_set[i][j] = 0.5
            continue
        if feature == 'ST':
            raw_data_set[i][j] = 0
            continue
        if feature == 'T':
            raw_data_set[i][j] = 1
            continue
        if feature == 'BX':
            raw_data_set[i][j] = 2
            continue
        if feature == 'GK':
            raw_data_set[i][j] = 3
            continue
        if feature == 'NA':
            raw_data_set[i][j] = 3
            continue
        if feature == 'RIP':
            raw_data_set[i][j] = [0,1]
            continue
        if feature == 'Alive':
            raw_data_set[i][j] = [1,0]
            continue
        raw_data_set[i][j] = float(raw_data_set[i][j])

labels = []
inputs = []
numpy.random.shuffle(raw_data_set)
for sample in raw_data_set:
    labels.append(sample[len(sample)-1])
    inputs.append(sample[:-1])

def input_inject_GLIOMA_MLP():
    return inputs, labels

# inputs, labels = input_inject_GLIOMA()
# print inputs, labels
