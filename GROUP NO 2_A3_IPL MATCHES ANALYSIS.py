#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[103]:


match=pd.read_csv("matches.csv")
match1=pd.read_csv("matches.csv")
match1.shape


# In[3]:


match1.head()


# In[4]:


match1.isnull().sum()


# In[5]:


#total number of matches in the dataset
match1['id'].max()


# In[6]:


#total number of matches in the dataset
match1['id'].max()


# In[7]:


#matches with no result
match1[pd.isnull(match1.winner)]


# In[9]:


match1['winner'].fillna('Draw', inplace=True)


# In[10]:


match1[pd.isnull(match1.winner)]


# In[11]:


match1.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Rising Pune Supergiant','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','RPS','KTK','PW'],inplace=True)

encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
match1.replace(encode, inplace=True)
match1.head(2)


# In[12]:


match1[pd.isnull(match1.city)]


# In[13]:


match1.city.fillna('city', inplace=True)


# In[14]:


#find key by value search 
code = encode['winner']
print(code['MI']) #key value
print(list(code.keys())[list(code.values()).index(1)]) 


# In[15]:


match2 = match1[['team1','team2','city','toss_decision','toss_winner','venue','winner']]
match2.head()


# In[16]:


match2 = pd.DataFrame(match2)
match2.describe()


# In[17]:


t1=match2['toss_winner'].value_counts(sort=True)
t2=match2['winner'].value_counts(sort=True)
print('No of toss winners by each team')
for idx, val in t1.iteritems():
   print('{} -> {}'.format(list(code.keys())[list(code.values()).index(idx)],val))
print('No of match winners by each team')
for idx, val in t2.iteritems():
   print('{} -> {}'.format(list(code.keys())[list(code.values()).index(idx)],val))


# In[19]:


ypos= np.arange(len(match2.winner))
ypos


# In[20]:


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Team')
ax1.set_ylabel('Count of toss wins')
ax1.set_title("toss winners")
t1.plot(kind='bar')

ax2 = fig.add_subplot(122)
ax2.set_xlabel('Team')
ax2.set_ylabel('count of matches won')
ax2.set_title("Match winners")
t2.plot(kind = 'bar')


# In[24]:


#find out null values
match2.apply(lambda x: sum(x.isnull()),axis=0) 


# In[25]:





# In[ ]:





# In[ ]:





# In[40]:


x = match2.drop(['city','toss_decision','venue'],axis='columns')
y=match2.winner
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[98]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'logistic_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


# In[70]:


#building predictive model
from sklearn.preprocessing import LabelEncoder
var_mod = ['city','toss_decision','venue']
le = LabelEncoder()
for i in var_mod:
    match2[i] = le.fit_transform(match2[i])
match2.dtypes 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold as kf  #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[80]:


#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  model.fit(data[predictors],data[outcome])
  predictions = model.predict(data[predictors])
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print('Accuracy : %s' % '{0:.3%}'.format(accuracy))


# In[81]:


from sklearn.ensemble import RandomForestRegressor
outcome_var=['winner']
predictor_var = ['team1','team2','toss_winner']
model = LogisticRegression()
classification_model(model, match2,predictor_var,outcome_var)


# In[82]:


match2.head(3)


# In[83]:


model = RandomForestClassifier(n_estimators=100)
outcome_var = ['winner']
predictor_var = ['team1', 'team2', 'venue', 'toss_winner','city','toss_decision']
classification_model(model, df,predictor_var,outcome_var)


# In[86]:


#'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'
team1='RCB'
team2='KKR'
toss_winner='RCB'
input=[code[team1],code[team2],'14',code[toss_winner],'2','1']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(code.keys())[list(code.values()).index(output)]) #find key by value search output


# In[87]:


#'team1', 'team2', 'venue', 'toss_winner','city','toss_decision'


team1='DC'
team2='DD'
toss_winner='DC'
input=[code[team1],code[team2],'23',code[toss_winner],'14','0']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(code.keys())[list(code.values()).index(output)]) #find key by value search output


# In[88]:


#okay from the above prediction on features, we notice toss winner has least chances of winning matches
#but does the current stats shows the same result
#df.count --> 577 rows
#Previously toss_winners were about 50.4%, with 2017 IPL season, it has reached 56.7%. As data matures, so does
# the changes in the predictions
import matplotlib.pyplot as mlt
mlt.style.use('fivethirtyeight')
df_fil=df[df['toss_winner']==df['winner']]
slices=[len(df_fil),(577-len(df_fil))]
mlt.pie(slices,labels=['Toss & win','Toss & lose'],startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%',colors=['r','g'])
fig = mlt.gcf()
fig.set_size_inches(6,6)
mlt.show()
# Toss winning does not gaurantee a match win from analysis of current stats and thus 
#prediction feature gives less weightage to that 


# In[92]:


#top 2 team analysis based on number of matches won against each other and how venue affects them?
#Previously we noticed that CSK won 79, RCB won 70 matches
#now let us compare venue against a match between CSK and RCB
#we find that CSK has won most matches against RCB in MA Chidambaram Stadium, Chepauk, Chennai
#RCB has not won any match with CSK in stadiums St George's Park and Wankhede Stadium, but won matches
#with CSK in Kingsmead, New Wanderers Stadium.
#It does prove that chances of CSK winning is more in Chepauk stadium when played against RCB.
# Proves venue is important feature in predictability
import seaborn as sns
team1=code['MI']
team2=code['CSK']
mtemp=match2[((match2['team1']==team1)|(match2['team2']==team1))&((match2['team1']==team2)|(match2['team2']==team2))]
sns.countplot(x='venue', hue='winner',data=mtemp,palette='Set2')
mlt.xticks(rotation='vertical')
leg = mlt.legend( loc = 'upper right')
fig=mlt.gcf()
fig.set_size_inches(10,6)
mlt.show()


# In[93]:


le.classes_[34]


# In[95]:


match1.city


# In[96]:


match2.city


# In[97]:


team1='CSK'
team2='MI'
toss_winner='CSK'
input=[code[team1],code[team2],'34',code[toss_winner],'2','1']
input = np.array(input).reshape((1, -1))
output=model.predict(input)
print(list(code.keys())[list(code.values()).index(output)]) #find key by value search output


# In[100]:


sns.countplot(x='season', data=match1)
plt.show()


# In[104]:


sns.countplot(y='winner', data = match)
plt.show()


# In[106]:


#toss winning has helped?
ss = match2['toss_winner'] == match2['winner']

ss.groupby(ss).size()


# In[107]:


#%of winning after toss winning
round(ss.groupby(ss).size() / ss.count() * 100,2)


# In[108]:


sns.countplot(ss);


# In[ ]:




