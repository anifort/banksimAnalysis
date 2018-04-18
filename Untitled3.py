
# coding: utf-8

# In[523]:

import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
import graphviz, pydot
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))


# In[561]:


class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0}  onload="this.width=screen.width;this.height=screen.height/3;"></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

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
    return output
    # (graph,) = pydot.graph_from_dot_file(filename)
    # graph.write_png(filename+'.png')
    


def plot_score_3d(results):
    fig = plt.figure(figsize=(13, 13))
    
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    max_depth = np.array(results['param_max_depth'].data, dtype=float)
    min_samples_leaf = np.array(results['param_min_samples_leaf'].data, dtype=float)
    score = results['mean_test_Accuracy']
    ax.scatter(max_depth, min_samples_leaf, score)

    ax.set_xlabel('max_depth')
    ax.set_ylabel('min_samples_leaf')
    ax.set_zlabel('Score')

    plt.show()


def plot_score(results):
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_max_depth'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()


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

# In[427]:


dt.fit(X, y)
get_ipython().run_line_magic('timeit', 'y_pred_default = cross_val_predict(dt, X, y, cv=cv, n_jobs=-1)')


# ### Classification Report

# In[398]:


print(metrics.classification_report(y, y_pred_default))


# ### Confusion Matrix

# In[399]:


plot_cm(y,y_pred_default,class_names)


# ### Export tree graph in pdf

# In[466]:


out = export_tree(dt,'dt_default')


# In[472]:


PDF(out)


# # Run model with parameter tunning (grid search)

# In[496]:

"""

params = {
        'class_weight':[None], 
        'criterion':['entropy'], 
        'max_depth':range(13,16,1),
        'min_samples_leaf':range(29,32,1), 
        'max_features':[None], 
        'max_leaf_nodes':[None],
        'min_impurity_decrease':[0.0], 
        'min_impurity_split':[None],
        'min_samples_split':[2,3],
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


# In[538]:


print("Best parameters set found on development set:")
print()
print(gs.best_params_)
#print(gs.cv_results_)
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_Accuracy']
stds = gs.cv_results_['std_test_Accuracy']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


# In[562]:


plot_score_3d(gs.cv_results_)


# In[509]:


dt = gs.best_estimator_


# ### Train Model with optimal parameters,10 cross validations and default parameters, in all cores

# In[499]:


y_pred_param_tune = cross_val_predict(dt, X, y, cv=10, n_jobs=-1)


# ### Classification Report

# In[500]:


print(metrics.classification_report(y, y_pred_param_tune))


# ### Confusion Matrix

# In[501]:


plot_cm(y,y_pred_param_tune,class_names)


# ### Export tree graph in pdf

# In[502]:


out = export_tree(dt,'dt_param_tune')


# In[508]:


PDF(out)


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

