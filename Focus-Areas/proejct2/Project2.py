from google.colab import drive
drive.mount('/content/drive')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score




# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from pandas.tools import plotting
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

names=['id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
copied_path =  r'C:/Users/MACWAN/Documents/data/wdbc_data.csv' #remove ‘content/’ from path then use 
data = pd.read_csv(copied_path,names=names)



y = data.diagnosis 
data = data.drop(['id'],axis = 1) #remove first column as id


# quick look to data
data.head()


data.shape # (569, 31)
data.describe()

def nullcounts(ser):
    return ser.isnull().sum()


def custom_describe(frame, func=[nullcounts, 'sum', 'mean', 'median', 'max','std'],
                    numeric_only=True, **kwargs):
    if numeric_only:
        frame = frame.select_dtypes(include=np.number)
    return frame.agg(func, **kwargs)
custom_describe(data)

data.columns 

def histogram_gen(feature,data):
  x_m=data[data["diagnosis"] == "M"][feature]
  x_b=data[data["diagnosis"] == "B"][feature]
  m = plt.hist(x_m,bins=30,fc = (1,0,0,0.5),label = "Malignant")
  b = plt.hist(x_b,bins=30,fc = (0,1,0,0.5),label = "Bening")
  plt.legend()
  plt.xlabel(feature)
  plt.ylabel("Frequency")
  plt.title("Histogram of "+feature+" for Bening and Malignant Tumors")
  fig1 = plt.gcf()
  plt.show()
  frequent_malignant = m[0].max()
  index_frequent_malignant = list(m[0]).index(frequent_malignant)
  most_frequent_malignant = m[1][index_frequent_malignant]
  print("Most frequent malignant "+feature+" mean is: ",most_frequent_malignant)
  #fig1.savefig("/content/sample_data/project2_output/"+feature+".png",dpi=100)
for feature in names[2:]:
  histogram_gen(feature,data)

columns_need_to_drop=['concavity_se', 'fractal_dimension_se', 'texture_se']

def outliers(feature,data):
  data_bening = data[data["diagnosis"] == "B"]
  data_malignant = data[data["diagnosis"] == "M"]
  #for class B 
  desc = data_bening[feature].describe()
  Q1 = desc[4]
  Q3 = desc[6]
  IQR = Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  print("For Feature "+feature+" Anything outside this range is an outlier for class B rows: (", lower_bound ,",", upper_bound,")")
  data_bening[data_bening[feature] < lower_bound][feature]
  print("Outliers: ",data_bening[(data_bening[feature] < lower_bound) | (data_bening[feature] > upper_bound)][feature].values)
  #for class M
  desc = data_malignant[feature].describe()
  Q1 = desc[4]
  Q3 = desc[6]
  IQR = Q3-Q1
  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR
  print("For Feature "+feature+" Anything outside this range is an outlier for class M: (", lower_bound ,",", upper_bound,")")
  data_malignant[data_malignant[feature] < lower_bound][feature]
  print("Outliers: ",data_malignant[(data_malignant[feature] < lower_bound) | (data_malignant[feature] > upper_bound)][feature].values)

for feature in names[2:]:
  outliers(feature,data)

def box_plot(data,names):
  melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = names[2:5])
  plt.figure(figsize = (15,10))
  sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
  plt.show()
box_plot(data,names)

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()

corr_matrix=data.corr()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
to_drop=to_drop+columns_need_to_drop
data = data.drop(to_drop,axis = 1) 

data = data.drop(['diagnosis'],axis = 1)
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=42)

ax = sns.countplot(y_test,label="Count")       # M = 212, B = 357
B, M = y_test.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)

#random forest classifier with n_estimators=10 (default)
n_trees=[5,7,10,15,25,30,100]
for n_t in n_trees:
  clf_rf = RandomForestClassifier(n_estimators=n_t, random_state=43)      
  clr_rf = clf_rf.fit(x_train,y_train)
  ac = accuracy_score(y_train,clf_rf.predict(x_train))
  print('Train Data','number of estimators:',n_t,'Accuracy is with : ',ac)

  ac = accuracy_score(y_test,clf_rf.predict(x_test))
  print('Test Data  number of estimators:',n_t,'Accuracy is with : ',ac)
 # cm = confusion_matrix(y_test,clf_rf.predict(x_test))
 # sns.heatmap(cm,annot=True,fmt="d")

def feature_importance(clf_rf,x_train):
  #clf_rf.predict_proba(x_test)
  importances = clf_rf.feature_importances_
  std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
              axis=0)
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  for f in range(x_train.shape[1]):
      print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

  # Plot the impurity-based feature importances of the forest
  plt.figure()
  plt.title("Feature importances")
  plt.bar(range(x_train.shape[1]), importances[indices],
          color="r", yerr=std[indices], align="center")
  plt.xticks(range(x_train.shape[1]), indices)
  plt.xlim([-1, x_train.shape[1]])
  plt.show()

from sklearn.tree import DecisionTreeClassifier

len(data.columns)

a=[119513,'N',31,18.02,27.6,117.5,1013,0.09489,0.1036,0.1086,0.07055,0.1865,0.06333,0.6249,1.89,3.972,71.55,0.004433,0.01421,0.03233,0.009854,0.01694,0.003495,21.63,37.08,139.7,1436,0.1195,0.1926,0.314,0.117,0.2677,0.08113,5,5
]
len(a)

#random forest classifier with n_estimators=10 (default)
clf_tree = DecisionTreeClassifier(random_state=43)      
clr_tree = clf_tree.fit(x_train,y_train)
ac = accuracy_score(y_train,clf_tree.predict(x_train))
print('Train Data','Accuracy is with : ',ac)

ac = accuracy_score(y_test,clf_tree.predict(x_test))
print('Test Data' ,'Accuracy is with : ',ac)
# cm = confusion_matrix(y_test,clf_rf.predict(x_test))
# sns.heatmap(cm,annot=True,fmt="d")


print("Important features by decision tree")
a=[]
for b,c in zip(data.columns,clf_tree.feature_importances_):
  a.append([b,abs(c)])
sort_list=sorted(a,key=lambda x :x[1],reverse=True)
for pos, coff in enumerate(sort_list):
  print(pos,coff[0],coff[1])

import xgboost as xgb

# Train the model, this will take a few minutes to run
bst = xgb.XGBClassifier(
    objective='reg:logistic'
)

bst.fit(x_train, y_train)


print("Important features by Xgboost")
a=[]
for b,c in zip(data.columns,bst.feature_importances_):
  a.append([b,abs(c)])
sort_list=sorted(a,key=lambda x :x[1],reverse=True)
for pos, coff in enumerate(sort_list):
  print(pos,coff[0],coff[1])


# Get predictions on the test set and print the accuracy score
y_pred = bst.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc, '\n')

# Save the model so we can deploy it
bst.save_model('model.bst')

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
ac = accuracy_score(y_train,clf.predict(x_train))
print('Train Data','Accuracy is  : ',ac)

ac = accuracy_score(y_test,clf.predict(x_test))
print('Test Data Accuracy is : ',ac)
#clf.predict_proba(x_test)
y_pred_proba = clf.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label='M')
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logstic regression, auc="+str(auc))
plt.legend(loc=4)
plt.show()

print("Important features by Logistic Regression")
a=[]
for b,c in zip(data.columns,clf.coef_[0]):
  a.append([b,abs(c)])
sort_list=sorted(a,key=lambda x :x[1],reverse=True)
for pos, coff in enumerate(sort_list):
  print(pos,coff[0],coff[1])

from sklearn import metrics

def plot_roc(clsf,x_test,y_test):
    rfc_disp = metrics.plot_roc_curve(clsf, x_test, y_test, ax=plt.gca(), alpha=0.8)
    plt.show()

def report(y_test,y_pred):
    print(metrics.classification_report(y_test, y_pred))  
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print( "Confusion matrix: \n",confusion_matrix(y_test,y_pred))

def eval_confusion_matrix(clsf,x_test,y_test):
    titles_options = [("Confusion matrix", None)]
    for title, normalize in titles_options:
        disp = metrics.plot_confusion_matrix(clsf, x_test, y_test,
                                 display_labels=["M","B"],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

    plt.show()
    
def eval_prob(X_train,X_test,y_train,y_test):
    train_probs = clf_rf.predict_proba(X_train)[:,1] 
    probs = clf_rf.predict_proba(X_test)[:, 1]
    train_predictions = clf_rf.predict(X_train)
    print(f'Train ROC AUC Score: {metrics.roc_auc_score(y_train, train_probs)}')
    print(f'Test ROC AUC  Score: {metrics.roc_auc_score(y_test, probs)}')
    

#ROC curve with auc 
y_pred_probar = clr_rf.predict_proba(x_test)[::,1]#randomforest
rfpr, rtpr, _ = metrics.roc_curve(y_test, y_pred_probar,pos_label='M')
rauc = metrics.roc_auc_score(y_test, y_pred_probar)

y_pred_probax = bst.predict_proba(x_test)[::,1]#xgboost
xfpr, xtpr, _ = metrics.roc_curve(y_test, y_pred_probax,pos_label='M')
xauc = metrics.roc_auc_score(y_test, y_pred_probax)

y_pred_probal = clf.predict_proba(x_test)[::,1]#LogisticRegression
lfpr, ltpr, _ = metrics.roc_curve(y_test, y_pred_probal,pos_label='M')
lauc = metrics.roc_auc_score(y_test, y_pred_probal)

y_pred_probad = clf_tree.predict_proba(x_test)[::,1]#DecisionTreeClassifier
dfpr, dtpr, _ = metrics.roc_curve(y_test, y_pred_probad,pos_label='M')
dauc = metrics.roc_auc_score(y_test, y_pred_probad)

plt.plot(rfpr,rtpr,label="RandomForestClassifier, auc="+str(rauc))
plt.plot(xfpr,xtpr,label="XGBoostClassifier, auc="+str(xauc))
plt.plot(lfpr,ltpr,label="LogisticRegression, auc="+str(lauc))
plt.plot(dfpr,dtpr,label="DecisionTreeClassifier, auc="+str(dauc))
plt.title("ROC Curve")
plt.legend(loc=8)
plt.show()

print("RandomFoestClassifier")
eval_confusion_matrix(clr_rf, x_test, y_test)

print("XGBoostClassifier")
eval_confusion_matrix(bst, x_test, y_test)

report(y_test,clf_rf.predict(x_test))#report for Random forest

report(y_test,bst.predict(x_test))#report for XGClassifier 

