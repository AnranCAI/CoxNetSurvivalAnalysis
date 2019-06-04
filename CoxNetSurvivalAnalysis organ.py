import csv
import numpy
import random
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt


# converts cells to floats, empty cells become 0
def convertToFloat(x):
    if x == '' or x == '-':
        return 0
    else:
        return float(x)

'''#data transformation for KM
def split_for_kaplan(labels, vars, id):
    event = [[], []];
    time = [[], []];
    for i in range(0, len(labels)):
        event[int(vars[i][id])].append(labels[i][0])
        time[int(vars[i][id])].append(labels[i][1])        #the sample i in the list of variable, index of each sample
    return (event, time);'''

# open data r is read
f = open('base_diabete_ml_sauf_sexe.csv', 'r')
# create reader
reader = csv.reader(f, delimiter=';')
# remove first line (header parts) from reader and save as header
header = next(reader)

# create a list for all features
data = []
# create a list for labels (got sick or not)
label = []

# for every row in the data
for row in reader:
    # convert every feature cell, map to array and apply convert function
    data.append(numpy.array(list(map(convertToFloat, row[2:len(row)]))))
    # get the label and add it to the label list, row 0 is the first cell of row, gives list; if 1 gives true
    label.append(('1' == row[0], convertToFloat(row[1]))) #list of label

# transform label data to required format, zip takes label and data from each subject combine as new list
d = list(zip(data, label));

diabetic = []
non_diabetic = []

#d is list of sample,every sample has a list where the first element is the list of label and the second element is the list of variables
for val in d:  #do for each sample in d
    if(val[0][0]==True): #for each sample,we take first element which is the list of label then the first element of this list which is diabetes
        diabetic.append(val);
    else:
        non_diabetic.append(val);

# shuffle data by random
random.shuffle(diabetic);
random.shuffle(non_diabetic);

# split data and labels
train_data = diabetic[0:int(0.66 * len(diabetic))] + non_diabetic[0:int(0.66 * len(non_diabetic))];
test_data = diabetic[int(0.66 * len(diabetic)) : len(diabetic)] + non_diabetic[int(0.66 * len(non_diabetic)) : len(non_diabetic)];

_train_d, _train_l = zip(*train_data);
_test_d, _test_l = zip(*test_data);

_train_d = list(_train_d)
_test_d = list(_test_d)

#bool means true or false , f4 means 4 bytes,
_train_l = numpy.array(list(_train_l), dtype='bool,f4');
_test_l = numpy.array(list(_test_l), dtype='bool,f4');

'''plot some estimator stuff
_event, _time = split_for_kaplan(_train_l, _train_d, 24)

for i in range(0, len(_event)):
    x, y = kaplan_meier_estimator(_event[i], _time[i])
    plt.step(x, y, where="post", label="CT_group= "+str(i));

plt.legend();
plt.plot();
plt.show();'''



# create and train the coxnet model
clf = CoxnetSurvivalAnalysis(n_alphas=100,l1_ratio=0.5,alpha_min_ratio=0.01,tol=0.1,fit_baseline_model=True).fit(_train_d, _train_l)

ccx = []
event_indicator=[val[0] for val in _test_l]
event_time=[val[1] for val in _test_l]
for val in clf.alphas_:
    res=clf.predict(_test_d,alpha=val)
    ccx.append(concordance_index_censored(event_indicator,event_time,estimate=res))

#curve concordance over alphas
plt.step(clf.alphas_, [val[0] for val in ccx], where="post");
plt.show();

#check mqx concordance index
print("concordance index:" + str(max([egg[0] for egg in ccx])))


best_alpha_index = numpy.array([egg[0] for egg in ccx]).argmax();  

print("best index: " + str(best_alpha_index))
#check best alpha
print("best alphas" + str(clf.alphas_[best_alpha_index]))

    
#prediction
res= clf.predict(_test_d,alpha=clf.alphas_[best_alpha_index]);

# save coeficientos
numpy.savetxt("coefsorgan.txt", list(map(lambda x : [str(x[0]),str(x[1])], zip([val[best_alpha_index] for val in clf.coef_], header[2:len(header)]))), fmt="%s")
numpy.savetxt("resorgan.txt",res,fmt="%s")

#prediction mean curve
pred_curves = clf.predict_survival_function(_test_d,alpha=clf.alphas_[best_alpha_index])

avg_y = []

for i in range(0,len(pred_curves[0].x)):
    val = 0;
    for j in range(0,len(pred_curves)):
        val += pred_curves[j].y[i];
    avg_y.append((pred_curves[0].x[i], float(val)/float(len(pred_curves))));

x, y = zip(*avg_y)
x = list(x)
y = list(y)

plt.step(x, y, where="post");
plt.show()


#print predict result
print(_test_l[res.argmax()])
print(_test_l[res.argmin()])

#print all sick people and healthy people predict result
index_sick = []
index_health = []

for i in range(0, len(_test_l)):
    if _test_l[i][0]==True:
        index_sick.append(i)
    else:
        index_health.append(i)

for i in range (0,len(index_sick)):
    print(res[i])




for i in range (0,len(index_health)):
    print(res[i])



