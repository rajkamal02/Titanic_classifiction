#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[4]:


df1 = pd.read_csv("D:\\Bharat intern\\titanic_train.csv")
df1.head()


# In[6]:


df1.isnull().any()


# In[7]:


df1.info()


# In[10]:


x=df1.describe()
print(x)


# In[11]:


import matplotlib.pyplot as plt


# In[14]:


df=x.T
df.plot(kind='barh',color='green',y='count',grid=True)
plt.title('Count_of_Features')


# In[15]:


# General visualization of the passengers survived and not-survived according to different aspects.
import seaborn as sns
sns.set_style('whitegrid')
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(18,7))
sns.countplot(data=df1,x ='Survived', ax=ax1)
sns.countplot(data=df1,ax=ax2,x='Survived',hue='Pclass')
sns.countplot(data=df1,ax=ax3,x='Survived',hue='Sex')
ax1.set_title('Total Passengers',fontsize=15)
ax1.set_xlabel('Survived',fontsize=14)
ax1.set_ylabel('Count',fontsize=14)
ax2.set_title('Total Passengers by Pclass',fontsize=15)
ax2.set_xlabel('Survived',fontsize=14)
ax2.set_ylabel('Count',fontsize=14)
ax3.set_title('Total Passengers by Sex',fontsize=15)
ax3.set_xlabel('Survived',fontsize=14)
ax3.set_ylabel('Count',fontsize=14)
plt.show()


# In[16]:


Frac_of_Pas_Sur=df1.groupby('Pclass')['Survived'].mean()
Frac_of_Pas_Sur=pd.DataFrame(Frac_of_Pas_Sur)
Frac_of_Pas_Sur


# In[17]:


Frac_of_Pas_Sur.plot(kind='barh',grid=True)
plt.title('Fraction_of_Passengers_Survived_per_Class')


# In[18]:


df1.head(5)


# In[19]:


fig,ax00=plt.subplots(figsize=(10,6))
df1.Age.hist(bins=35,color='blue',alpha=0.8,ax=ax00)
ax00.set_xlabel('Passenger_Age',fontsize=14)
ax00.set_ylabel('Count',fontsize=14)
ax00.set_title('Age Distribution of All Passengers',fontsize=18)
fig,(ax10,ax11)=plt.subplots(ncols=2,figsize=(12,6),sharex=True,sharey=True)
df1.loc[df1["Sex"]=='male','Age'].hist(bins=35,color='green',alpha=0.8,ax=ax10)
df1.loc[df1["Sex"]=='female','Age'].hist(bins=35,color='yellow',alpha=0.8,ax=ax11)
ax10.set_title('Age Distribution of Males',fontsize=15)
ax11.set_title('Age Distribution of Females',fontsize=15)
ax10.set_xlabel('Age_of_Male_Passengers')
ax11.set_xlabel('Age_of_Female_Passengers')
ax11.set_ylabel('Count')
ax10.set_ylabel('Count')
ax11.tick_params(axis='y',which='both',labelleft=False,labelright=True)
plt.show()


# In[20]:


fig,AX=plt.subplots(figsize=(8,5))
df1.groupby(['Survived']).get_group((1))['Age'].hist(bins=35,alpha=0.8,color='darkgreen',ax=AX)
AX.set_title('Age Distribution of Surviving males and femles',fontsize=16)
AX.set_xlabel('Age',fontsize=14)
AX.set_ylabel('Count',fontsize=14)
fig,(ax21,ax22)=plt.subplots(ncols=2,figsize=(15,6))
df1.groupby(['Survived','Sex']).get_group((1,'male'))['Age'].hist(bins=35,alpha=0.8,color='red',ax=ax21)
df1.groupby(['Survived','Sex']).get_group((1,'female'))['Age'].hist(bins=35,alpha=0.8,color='cyan',ax=ax22)
ax21.set_title('Age Distribution of Surviving Men',fontsize=15)
ax21.set_xlabel("Age of Men(Survived)",fontsize=13)
ax21.set_ylabel('Count',fontsize=13)
ax22.set_xlabel("Age of Women(Survived)",fontsize=13)
ax22.set_ylabel('Count',fontsize=13)
ax22.set_title('Age Distribution of Surviving Women',fontsize=15)
plt.show()
fig,AX1=plt.subplots(figsize=(8,5))
df1.groupby(['Survived']).get_group((0))['Age'].hist(bins=35,alpha=0.8,color='yellow',ax=AX1)
AX1.set_title('Age Distribution of non-surviving males and femles',fontsize=16)
AX1.set_xlabel('Age',fontsize=14)
AX1.set_ylabel('Count',fontsize=14)
fig,(ax31,ax32)=plt.subplots(ncols=2,figsize=(15,6))
df1.loc[(df1['Survived']==0) & (df1['Sex']=='male'),'Age'].hist(bins=30,alpha=0.8,color='darkred',ax=ax31)
df1.query("Survived==0 and Sex=='female'")['Age'].hist(bins=30,alpha=0.8,color='blue',ax=ax32)
ax31.set_title('Age Distribution of Non-Surviving Men',fontsize=15)
ax31.set_xlabel("Age of Men(not-Survived)",fontsize=13)
ax31.set_ylabel('Count',fontsize=13)
ax32.set_title('Age Distribution of Non-Surviving Women',fontsize=15)
ax32.set_xlabel("Age of Women(not-Survived)",fontsize=13)
ax32.set_ylabel('Count',fontsize=13)


# In[21]:


data = df1.groupby(['Sex','Pclass'])['Survived'].count()
P_data= data.reset_index().pivot(index='Pclass',columns='Sex',values='Survived')
fig,aX1=plt.subplots(figsize=(10,6))
P_data.plot(kind='bar',stacked=True,ax=aX1)
aX1.set_title('Total Male-Female in every Passenger Class',fontsize=16)
aX1.set_xlabel('Passenger Class',fontsize=14)
aX1.set_ylabel('Count',fontsize=14)
fig,(aX11,aX12)=plt.subplots(ncols=2, figsize=(10,6),sharex=True,sharey=True)
data1=df1[df1['Survived']==1].groupby(['Sex','Pclass'])['Survived'].count().reset_index().pivot(index='Pclass',columns='Sex',values='Survived')
data2=df1[df1['Survived']==0].groupby(['Sex','Pclass'])['Survived'].count().reset_index().pivot(index='Pclass',columns='Sex',values='Survived')
data1.plot(kind='bar',stacked=True,ax=aX11)
data2.plot(kind='bar',stacked=True,ax=aX12)
aX11.set_title('Surviving Passengers from every P-Class',fontsize=13)
aX11.set_xlabel('Passenger Class',fontsize=11)
aX11.set_ylabel('Count',fontsize=11)
aX12.set_title('Non-Surviving Passengers from every P-Class',fontsize=13)
aX12.set_xlabel('Passener Class',fontsize=11)
aX12.set_ylabel('Count',fontsize=11)
aX12.tick_params(axis='y',which='both',labelleft=False,labelright=True)
plt.show()


# In[22]:


M_S=df1.loc[(df1['Sex']=='male') & (df1['Survived']==1)]
F_S=df1.loc[lambda x:(x['Sex']=='female') & (x['Survived']==1)]
fig,(AX11,AX12)=plt.subplots(ncols=2,figsize=(14,5),sharex=True,sharey=True)
sns.violinplot(data=M_S,x='Pclass',y='Age',palette='colorblind',ax=AX11)
sns.violinplot(data=F_S,x='Pclass',y='Age',palette='bright',ax=AX12)
AX11.set_title('Age distribution of Males')
AX12.set_title('Age distribution of Females')
AX12.tick_params(axis='y',which='both',labelleft=False,labelright=True)
M_n=df1.query("Sex=='male' and Survived==0")
F_n=df1.groupby(['Sex','Survived']).get_group(('female',0))
fig,(AX21,AX22)=plt.subplots(ncols=2,figsize=(14,5),sharex=True,sharey=True)
sns.boxplot(data=M_n,x='Pclass',y='Age',palette='pastel',ax=AX21,showmeans=True)
sns.boxplot(data=F_n,x='Pclass',y='Age',palette='dark',ax=AX22,showmeans=True)
AX21.set_title('Age distribution of dead Males')
AX22.set_title('Age distribution of dead Females')
AX12.tick_params(axis='y',which='both',labelleft=False,labelright=True)


# In[24]:


fig,(AA,BB)=plt.subplots(ncols=2,figsize=(14,6))
sns.violinplot(data=df1.query("Survived==1"),x='Pclass',y='Age',palette='dark',ax=AA)
sns.boxplot(data=df1.query("Survived==0"),x='Pclass',y='Age',palette='bright',ax=BB)
AA.set_title('Age distribution of all Survived Passengers')
BB.set_title('Age distribution of all dead Passengers')
M_NA=df1.groupby(['Sex','Survived',pd.cut(df1['Age'],[0,18,np.inf],labels=['<=18','>=18'])]).get_group(('female',1,'<=18'))
F_NA=df1[df1['Age']<=18].groupby(['Sex','Survived']).get_group(('female',1))
fig,(CC,DD)=plt.subplots(ncols=2,figsize=(14,6))
sns.boxplot(data=M_NA,x='Pclass',y='Age',showmeans=True,palette='bright',ax=CC)
sns.boxplot(data=F_NA,x='Pclass',y='Age',showmeans=True,palette='bright',ax=DD)
CC.set_title('Underaged males Survived',fontsize=13)
DD.set_title('Underaged females Survived',fontsize=13)


# In[26]:


df1.isnull().sum()


# In[27]:


Male_A_Age=df1[df1['Sex']=='male'].groupby(['Pclass'])['Age'].mean()
Female_A_Age=df1[df1['Sex']=='female'].groupby(['Pclass'])['Age'].mean()
Male_A_Age=pd.DataFrame(Male_A_Age)
Female_A_Age=pd.DataFrame(Female_A_Age)
fig,(QA,QB)=plt.subplots(ncols=2,figsize=(12,5),sharey=True)
Male_A_Age.plot(kind='bar',y='Age',grid=True,color='green',ax=QA)
Female_A_Age.plot(kind='bar',y='Age',grid=True,color='red',ax=QB)
QA.set_title('Average Age of Males per Class',fontsize=13)
QB.set_title('Average Age of Females per Class',fontsize=13)
QB.tick_params(axis='y',which='both',labelleft=False,labelright=True)


# In[28]:


print(Male_A_Age)
print(Female_A_Age)


# In[29]:


age_group=df1.groupby(['Sex','Pclass'])['Age'].mean().to_dict()
age_group
df1['Age']=df1.Age.fillna(df1.apply(lambda x:age_group.get((x['Sex'],x['Pclass'])),axis=1),inplace=False)
df1.isnull().sum()


# In[30]:


pre_data=df1.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=False)
pre_data


# In[31]:


sex=pd.get_dummies(pre_data['Sex'],drop_first=True)
embarked=pd.get_dummies(pre_data['Embarked'],drop_first=True)
pre_data1=pre_data.drop(columns=['Sex','Embarked'],inplace=False)
Data=pd.concat([pre_data1,sex,embarked],axis=1)
Data


# In[32]:


X=Data.drop(columns=['Survived'],axis=1)
Y=Data['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=130)
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
X_train=Scaler.fit_transform(X_train)
X_test=Scaler.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
Regressor=LogisticRegression(random_state=0)
Regressor.fit(X_train,Y_train)
Y_pred=Regressor.predict(X_test)
Y_pred


# In[33]:


from sklearn.metrics import confusion_matrix
CM=confusion_matrix(Y_test,Y_pred)
print(CM)
print('Model Accuracy is : ', (102+50)/(102+50+19+8))


# In[34]:


X1=Data.drop(columns=['Survived'],axis=1)
Y1=Data['Survived']
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,test_size=0.2,random_state=90)
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy')
tree.fit(X1_train,Y1_train)


# In[35]:


from sklearn.tree import plot_tree
plt.figure(figsize=(25,30))
plot_tree(tree,feature_names=['Age','male','SibSp','Parch','Fare','Pclass','Q','S'])
plt.show()


# In[36]:


tree.get_depth()


# In[37]:


tree.get_n_leaves()


# In[38]:


Y1_pred=tree.predict(X1_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y1_test,Y1_pred)


# In[39]:


print('Accuracy Score in Logistic Regression : 0.84916')
print('Accuracy Score in Decision Tree is : 0.79888')


# In[ ]:




