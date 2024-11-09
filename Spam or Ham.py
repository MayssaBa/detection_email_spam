import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score
from sklearn.preprocessing import StandardScaler
#from sklearn.naive_bayes import


colonnes = ['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our','word_freq_over','word_freq_remove','word_freq_internet',
    'word_freq_order','word_freq_mail','word_freq_receive','word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free',
    'word_freq_business','word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money',
    'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet','word_freq_857','word_freq_data',
    'word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting',
    'word_freq_original','word_freq_project','word_freq_re','word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[',
    'char_freq_!','char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','class'
]
file=pd.read_csv("spambase.data",names=colonnes)
df=pd.DataFrame(data=file)
print(df.head())
print(df.shape)

X=df.drop(['class'], axis=1)
y=df['class']
#print(X.head(),"\n",y.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#logistic regression
model_log=LogisticRegression()
model_log.fit(X_train,y_train)
#prediction:
y_pred=model_log.predict(X_test)

#confusion matrix:
matrix_confusion=confusion_matrix(y_test,y_pred)
plt.imshow(matrix_confusion,interpolation='nearest',cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("y_predicted")
plt.ylabel("y_real")
plt.colorbar()
plt.xticks(np.arange(2), ['Spam', 'Ham'])
plt.yticks(np.arange(2), ['Spam', 'Ham'])
plt.show()

#metrics:
prec=precision_score(y_test,y_pred)
print("precision_logistic_reg=",prec)

recall=recall_score(y_test,y_pred)
print("recall_logistic_reg=",recall)

F1=f1_score(y_test,y_pred)
print("f1_score_logistic_reg=",F1)


#svm:
svm=SVC(kernel='linear')
svm.fit(X_train,y_train)
y_svm_pred=svm.predict(X_test)


plt.scatter(X_train[y_train==0][:,0],X_train[y_train==0][:,1],c='blue',s=50,edgecolors='k',label='Ham')
plt.scatter(X_train[y_train==1][:,0],X_train[y_train==1][:,1],c='red',s=50,edgecolors='k',label='Spam')
plt.xlabel('word_freq:make')
plt.ylabel('word_freq:address')
plt.legend()
plt.show()
#metrics:
prec_svm=precision_score(y_test,y_svm_pred)
print("\nprecision_svm=",prec)

recall_svm=recall_score(y_test,y_svm_pred)
print("recall_svm=",recall)

F1_svm=f1_score(y_test,y_svm_pred)
print("f1_score_SVM=",F1_svm)


#naive_bayes:









