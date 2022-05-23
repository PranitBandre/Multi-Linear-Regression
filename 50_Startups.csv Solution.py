#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[7]:


data=pd.read_csv('50_Startups.csv')
data


# In[8]:


data1=data.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
data1


# In[9]:


data1[data1.duplicated()]


# In[10]:


data1.describe()


# In[11]:


data1.corr()


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(data1)


# In[13]:


model=smf.ols("Profit~RDS+ADMS+MKTS",data=data1).fit()


# In[14]:


model.params


# In[15]:


model.tvalues , np.round(model.pvalues,5)


# In[16]:


model.rsquared , model.rsquared_adj


# In[17]:


slr_a=smf.ols("Profit~ADMS",data=data1).fit()
slr_a.tvalues , slr_a.pvalues


# In[18]:


slr_m=smf.ols("Profit~MKTS",data=data1).fit()
slr_m.tvalues , slr_m.pvalues


# In[19]:


mlr_am=smf.ols("Profit~ADMS+MKTS",data=data1).fit()
mlr_am.tvalues , mlr_am.pvalues


# In[20]:


rsq_r=smf.ols("RDS~ADMS+MKTS",data=data1).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMS~RDS+MKTS",data=data1).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTS~RDS+ADMS",data=data1).fit().rsquared
vif_m=1/(1-rsq_m)


d1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[21]:


sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[22]:


list(np.where(model.resid<-30000))


# In[23]:


def standard_values(vals) : return (vals-vals.mean())/vals.std()


# In[24]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# In[25]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'RDS',fig=fig)
plt.show()


# In[26]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'ADMS',fig=fig)
plt.show()


# In[27]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'MKTS',fig=fig)
plt.show()


# In[28]:


(c,_)=model.get_influence().cooks_distance
c


# In[30]:


fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[31]:


np.argmax(c) , np.max(c)


# In[33]:


data2=data1.drop(data1.index[[49]],axis=0).reset_index(drop=True)
data2


# In[34]:


while np.max(c)>0.5 :
    model=smf.ols("Profit~RDS+ADMS+MKTS",data=data2).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    data2=data2.drop(data2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    data2
else:
    final_model=smf.ols("Profit~RDS+ADMS+MKTS",data=data2).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[35]:


final_model.rsquared


# In[36]:


data2


# In[37]:


new_data=pd.DataFrame({'RDS':70000,"ADMS":90000,"MKTS":140000},index=[0])
new_data


# In[38]:


final_model.predict(new_data)


# In[39]:


pred_y=final_model.predict(data2)
pred_y


# In[40]:


d2={'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_model.rsquared]}
table=pd.DataFrame(d2)
table


# In[ ]:




