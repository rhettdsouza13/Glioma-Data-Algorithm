import csv
import numpy
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

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
    sample[0] = (abs((d2 - d1).days)/365)



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
            raw_data_set[i][j] = 4
            continue
        if feature == 'RIP':
            raw_data_set[i][j] = [0,1]
            continue
        if feature == 'Alive':
            raw_data_set[i][j] = [1,0]
            continue
        # if feature == '0' and j == 22:
        #     raw_data_set[i][j] = [0,1]
        #     continue
        # if feature == '1' and j == 22:
        #     raw_data_set[i][j] = [1,0]
        #     continue
        raw_data_set[i][j] = float(raw_data_set[i][j])



labels = []
inputs = []

numpy.random.shuffle(raw_data_set)

# with open('shuffled_GLIOMA.csv', 'wb+') as file_store:
#     wr = csv.writer(file_store)
#     for row in raw_data_set:
#         wr.writerow(row)

print raw_data_set

for dat in raw_data_set:
    #del dat[4]
    #del dat[4]
    continue

print len(raw_data_set[10])
for sample in raw_data_set:
    labels.append(sample[len(sample)-1])
    inputs.append(sample[:-1])

scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(inputs)
print scaler.data_max_
inputs = scaler.transform(inputs)
inputs = normalize(inputs)
#
# numpy.save("shuffled_GLIOMA_nump_inp.npy", inputs)
# numpy.save("shuffled_GLIOMA_nump_lab.npy", labels)

# inputs = numpy.load("shuffled_GLIOMA_nump_inp.npy")
# labels = numpy.load("shuffled_GLIOMA_nump_lab.npy")

def input_inject_GLIOMA_MLP():
    labels = []
    inputs = []
    numpy.random.shuffle(raw_data_set)

    print len(raw_data_set[10])
    for sample in raw_data_set:
        labels.append(sample[len(sample)-1])
        inputs.append(sample[:-1])

    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(inputs)
    # print scaler.data_max_
    inputs = scaler.transform(inputs)
    inputs = normalize(inputs)
    return inputs, labels

# inputs, labels = input_inject_GLIOMA_MLP()
# print inputs
# print labels
