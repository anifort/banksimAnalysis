
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
import graphviz, pydot
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from IPython.display import display, HTML
import time
display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))


# In[408]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
     
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    '''    
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_cm(y,y_pred, class_names):
    print('Confusion Matrix')
    cnf_matrix = metrics.confusion_matrix(y, y_pred)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Without normalization')
    
    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                      title='Normalized')

    plt.show()
    
def export_tree(dt, filename):
    dot_data = tree.export_graphviz(dt, out_file=None,
                         feature_names=df2.columns.values,  
                         class_names=class_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
    graph = graphviz.Source(dot_data) 
    output = graph.render(filename) 
    print("results saved: ", output)

    # (graph,) = pydot.graph_from_dot_file(filename)
    # graph.write_png(filename+'.png')


# In[376]:


df = pd.read_csv('bs140513_032310.csv')



# In[377]:


df.head()


# In[378]:


df.age=df.age.map(lambda x: x.lstrip("'").rstrip("'"))
df.merchant=df.merchant.map(lambda x: x.lstrip("'").rstrip("'"))
df.customer=df.customer.map(lambda x: x.lstrip("'").rstrip("'"))
df.category=df.category.map(lambda x: x.lstrip("'").rstrip("'"))


# In[379]:


df['age']= df['age'].str.replace('U','-1')


# In[380]:


df.age = pd.to_numeric(df.age)


# In[381]:


df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)


# In[382]:


df.loc[df.gender == "'M'", 'gender'] = 0
df.loc[df.gender == "'F'", 'gender'] = 1
df.loc[df.gender == "'E'", 'gender'] = -1
df.loc[df.gender == "'U'", 'gender'] = -1
df.gender = pd.to_numeric(df.gender) # ensure there are no str left


# In[383]:


df.merchant=df.merchant.map(lambda x: x.lstrip("M"))
df.customer=df.customer.map(lambda x: x.lstrip("C"))

df.customer = pd.to_numeric(df.customer)
df.merchant = pd.to_numeric(df.merchant)


# # Run model without parameter tunning

# In[ ]:


df2 = df
y = df2["fraud"]
df2 = df2.drop(["fraud", 'customer', 'step'], axis=1)
df2=pd.get_dummies(df2, columns=["category"])

X = df2
dt = DecisionTreeClassifier()

cv=10
class_names = [
    'not fraud', 'fraud']

"""
# ### Train Model with 10 cross validations and default parameters, in all cores

# In[400]:


dt.fit(X, y)
y_pred_default = cross_val_predict(dt, X, y, cv=cv, n_jobs=-1)


# ### Classification Report

# In[398]:


print(metrics.classification_report(y, y_pred_default))


# ### Confusion Matrix

# In[399]:


plot_cm(y,y_pred_default,class_names)


# ### Export tree graph in pdf

# In[409]:


export_tree(dt,'dt_default')

"""
# # Run model with parameter tunning (grid search)

# In[ ]:


params = {
        'class_weight':[None], 
        'criterion':['gini','entropy'], 
        'max_depth':range(8,40,1),
        'min_samples_leaf':range(22,40,1), 
        'max_features':[None], 
        'max_leaf_nodes':[None],
        'min_impurity_decrease':[0.0], 
        'min_impurity_split':[None],
        'min_samples_split':range(2,13,2),
        'min_weight_fraction_leaf':[0.0], 
        'presort':[False], 
        'random_state':[42],
        'splitter':['best']}

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
gs = GridSearchCV(dt,n_jobs=-1, 
                  param_grid=params, 
                  scoring=scoring, 
                  return_train_score=True, 
                  cv=cv, 
                  refit='Accuracy',verbose=1)
start_time = time.time()
gs.fit(X, y)
elapsed_time = time.time() - start_time
print("ellapsed time", elapsed_time)
#gs.cv_results_


# In[411]:


print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_Accuracy']
stds = gs.cv_results_['std_test_Accuracy']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


# In[352]:


dt = gs.best_estimator_


# ### Train Model with optimal parameters,10 cross validations and default parameters, in all cores

# In[353]:


y_pred_param_tune = cross_val_predict(dt, X, y, cv=10, n_jobs=-1)


# ### Classification Report

# In[398]:


print(metrics.classification_report(y, y_pred_param_tune))


# ### Confusion Matrix

# In[399]:


plot_cm(y,y_pred_param_tune,class_names)


# ### Export tree graph in pdf

# In[409]:


export_tree(dt,'dt_param_tune')


# # Feature Engineering : create additional features

# In[71]:


df['cust_tl_count_trans'] = df['amount'].groupby(df['customer']).transform('count')
df['cust_tl_mean_amount'] = df['amount'].groupby(df['customer']).transform('mean')
df['cust_tl_median_amount'] = df['amount'].groupby(df['customer']).transform('median')
df['cust_tl_std_amount'] = df['amount'].groupby(df['customer']).transform('std')
df['cust_tl_max_amount'] = df['amount'].groupby(df['customer']).transform('max')
df['cust_tl_min_amount'] = df['amount'].groupby(df['customer']).transform('min')


# In[70]:


df

