import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

!pip install -U -q PyDrive
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
fileDownloaded = drive.CreateFile({'id':'https://drive.google.com/file/d/1uTHL59_tid1Tu4O9_T2zHZYIYu0D6bit/view?usp=sharing'})
file_list = drive.ListFile({'q': "'1ATVDsYj_YhpzSmYhV5X-q21CYzujQ2As' in parents and trashed=false"}).GetList()
for file1 in file_list:
  print('title: %s, id: %s' % (file1['title'], file1['id']))
data_downloaded_user_likes = drive.CreateFile({'id': '1uTHL59_tid1Tu4O9_T2zHZYIYu0D6bit'})
data_downloaded_user_likes.GetContentFile('users-likes.csv')
data_downloaded_users = drive.CreateFile({'id': '1XRppHA9JDYGw1gio1Y5XpzxT9uwmJiU1'})
data_downloaded_users.GetContentFile('users.csv')
data_downloaded_likes = drive.CreateFile({'id': '1cHKEtaDRtQBFC6MCDcFMP_Wu8AgrwK4D'})
data_downloaded_likes.GetContentFile('likes.csv')

users=pd.read_csv('users.csv')
likes=pd.read_csv('likes.csv')
ul=pd.read_csv('users-likes.csv',error_bad_lines=False,engine='python')

#users=pd.read_csv('./sample_dataset/users.csv')
#likes=pd.read_csv('./sample_dataset/likes.csv')
#ul=pd.read_csv('./sample_dataset/users-likes.csv')
users.head()

ul.head()

likes.head()

df=ul.iloc[:50957]

sparse_matrix=df.groupby(['userid', 'likeid']).size().unstack(fill_value=0)

sparse_matrix.head()

user_ids=sparse_matrix.index.values.tolist()

sparse_matrix_n=np.array(sparse_matrix)

#Trimming 
sparse_matrix_n1=sparse_matrix_n[:,np.sum(sparse_matrix_n, axis=0)>1]
print(sparse_matrix_n1.shape)
sparse_matrix_n2=sparse_matrix_n1[np.sum(sparse_matrix_n, axis=1)>1]
print(sparse_matrix_n2.shape)

#print(user_ids.shape)
user_ids=np.array(user_ids)
user_ids=user_ids[np.sum(sparse_matrix_n, axis=1)>1]

sparse_matrix_n2

random.seed(68)

n_components=5
svd = TruncatedSVD(n_components=5)
X_reduced = svd.fit_transform(sparse_matrix_n2)
df_svd = pd.DataFrame(data=X_reduced, index=[i for i in range(len(user_ids))], columns=['svd_'+str(i+1) for i in range(n_components)])
df_svd['userid']=user_ids
df_svd.head()

left = users.set_index('userid')
right = df_svd.set_index('userid')
combined_table=left.join(right, how='inner')
combined_table.head()

combined_table.corr()

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(combined_table.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
# save heatmap as .png file
# dpi - sets the resolution of the saved image in dots/inches
# bbox_inches - when set to 'tight' - does not allow the labels to be cropped
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')

#X_reduced[0]
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)

n_components_svd=50
response_columns=8
result=np.zeros((n_components_svd,response_columns))
titles = ['Gender','Age','Political','ope','con','ext','agr','neu']
for n_c in range(1,n_components_svd+1):
    #print('model learning in progres for components',n_c)
    svd = TruncatedSVD(n_components=n_c)
    X_reduced = svd.fit_transform(sparse_matrix_n2)
    df_svd = pd.DataFrame(data=X_reduced, index=[i for i in range(len(user_ids))], columns=['svd_'+str(i+1) for i in range(n_c)])
    df_svd['userid']=user_ids
    left = users.set_index('userid')
    right = df_svd.set_index('userid')
    combined_table=left.join(right, how='inner')
    #for all y response variable except political one
    c_t=np.array(combined_table)
    X,y=c_t[:,response_columns:],c_t[:,:response_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #for political response variable
    c_t1=np.array(combined_table.dropna())
    X1,y1=c_t1[:,response_columns:],c_t1[:,2]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33, random_state=42)
    for y_col in range(response_columns):
        #for categorical columns
        if y_col in [0,2]:
            if y_col==2:
                clf = LogisticRegression().fit(X_train1, y_train1)
                auc=roc_auc_score(y_test1, clf.predict_proba(X_test1)[:, 1])
                result[n_c-1,y_col]=auc
                if(n_c==50):
                    print('no of component',n_c,'Political response variable auc',auc)
            else:
                clf = LogisticRegression().fit(X_train, y_train[:,y_col])
                auc=roc_auc_score(y_test[:,y_col], clf.predict_proba(X_test)[:, 1])
                result[n_c-1,y_col]=auc
                if(n_c==50):
                    print('no of component',n_c,'Gender response variable auc',auc)
            
        else:
            reg = LinearRegression().fit(X_train, y_train[:,y_col])
            y_pred=reg.predict(X_test)
            r=np.corrcoef(y_test[:,y_col],y_pred)[0][1]
            result[n_c-1,y_col]=r
            if(n_c==50):
                print('Number of component:',n_c,' y_col:',titles[y_col],'Accuracy(correlation)',r)
    

titles = ['Gender','Age','Political','ope','con','ext','agr','neu'] #title
fig, axs = plt.subplots(2,4,figsize=(10, 10))
fig.suptitle('AUC, Accuracy(Correlation) vs Number of components used', fontsize=16)
for kk, (ax,yy) in enumerate(zip(axs.reshape(-1),zip(*result))):
    #print(yy)
    ax.plot([i+1 for i in range(50)],yy)
    ax.set_title(titles[kk])
    ax.set_xlabel('number of components')
    if kk in [0,2]:
      ax.set_ylabel('AUC')
    else:
      ax.set_ylabel('Accuracy(correlation')
#fig.delaxes(axs[1][1])
plt.show()
plt.savefig('result.png', dpi=300, bbox_inches='tight')