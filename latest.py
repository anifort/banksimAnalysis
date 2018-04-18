
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from random import shuffle
import time
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree, metrics, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score
import graphviz, pydot
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,accuracy_score, roc_auc_score, roc_curve, auc
from IPython.display import display, HTML
import warnings
from scipy import interp
from scipy.stats import pearsonr

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
    
display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))


# In[2]:


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

    plt.xlabel("max depth")
    plt.ylabel("Score")
    plt.grid()

    ax = plt.axes()
    #ax.set_xlim(0, 402)
    ax.set_ylim(0.85, 1)

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
    
def report(y,y_pred,class_names):
    plot_cm(y,y_pred,class_names)
    print(metrics.classification_report(y, y_pred))
    print('ROC_AUC score', roc_auc_score(y,y_pred))
    
def reset_data():
    df = pd.read_csv('bs140513_032310.csv')
    df.age=df.age.map(lambda x: x.lstrip("'").rstrip("'"))
    df.merchant=df.merchant.map(lambda x: x.lstrip("'").rstrip("'"))
    df.customer=df.customer.map(lambda x: x.lstrip("'").rstrip("'"))
    df.category=df.category.map(lambda x: x.lstrip("'").rstrip("'"))
    
    df['age']= df['age'].str.replace('U','-1')
    df.age = pd.to_numeric(df.age)
    
    df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)
    
    df.loc[df.gender == "'M'", 'gender'] = 0
    df.loc[df.gender == "'F'", 'gender'] = 1
    df.loc[df.gender == "'E'", 'gender'] = -1
    df.loc[df.gender == "'U'", 'gender'] = -1
    df.gender = pd.to_numeric(df.gender) # ensure there are no str left
    
    df.merchant=df.merchant.map(lambda x: x.lstrip("M"))
    df.customer=df.customer.map(lambda x: x.lstrip("C"))

    df.customer = pd.to_numeric(df.customer)
    df.merchant = pd.to_numeric(df.merchant)
    return df


df = reset_data()


# In[176]:


df['cust_tl_count_trans'] = df['amount'].groupby(df['customer']).transform('count')
#df['cust_cat_count_trans'] = df.groupby(['customer', 'category'])['amount'].transform('count')
#df['cust_mer_count_trans'] = df.groupby(['customer', 'merchant'])['amount'].transform('count')

df['cust_tl_mean_amount'] = df['amount'].groupby(df['customer']).transform('mean')
df['cat_mean_amount'] = df.groupby(['category'])['amount'].transform('mean')
df['cust_mer_mean_amount'] = df.groupby(['merchant'])['amount'].transform('mean')
#df['cust_cat_mean_amount'] = df.groupby(['customer', 'category'])['amount'].transform('mean')
#df['cust_mer_mean_amount'] = df.groupby(['customer', 'merchant'])['amount'].transform('mean')

df['cust_tl_median_amount'] = df['amount'].groupby(df['customer']).transform('median')
#df['cust_cat_median_amount'] = df.groupby(['customer', 'category'])['amount'].transform('median')
#df['cust_mer_median_amount'] = df.groupby(['customer', 'category'])['amount'].transform('median')

df['cust_tl_std_amount'] = df['amount'].groupby(df['customer']).transform('std')
df['cat_mean_amount'] = df.groupby(['category'])['amount'].transform('std')
df['cust_mer_mean_amount'] = df.groupby(['merchant'])['amount'].transform('std')
#df['cust_cat_std_amount'] = df.groupby(['customer', 'category'])['amount'].transform('std')
#df['cust_mer_std_amount'] = df.groupby(['customer', 'merchant'])['amount'].transform('std')

df['cust_tl_max_amount'] = df['amount'].groupby(df['customer']).transform('max')
df['cat_mean_amount'] = df.groupby(['category'])['amount'].transform('max')
df['cust_mer_mean_amount'] = df.groupby(['merchant'])['amount'].transform('max')
#df['cust_cat_max_amount'] = df.groupby(['customer', 'category'])['amount'].transform('max')
#df['cust_mer_max_amount'] = df.groupby(['customer', 'merchant'])['amount'].transform('max')

df['cust_tl_min_amount'] = df['amount'].groupby(df['customer']).transform('min')
df['cat_mean_amount'] = df.groupby(['category'])['amount'].transform('min')
df['cust_mer_mean_amount'] = df.groupby(['merchant'])['amount'].transform('min')
#df['cust_cat_min_amount'] = df.groupby(['customer', 'category'])['amount'].transform('min')
#df['cust_mer_min_amount'] = df.groupby(['customer', 'merchant'])['amount'].transform('min')

df = df.fillna(0)


# In[177]:


fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(35, 12)
ax = sns.boxplot(x="category", y="amount", data=df[df['category']!='es_travel'], palette="Set3")
ax.set_ylim(-1,1500)


# In[178]:


category_sample = list(set(list(df['category'])))
customer_sample = list(set(list(df['customer'])))



# In[180]:


'''
clf = LocalOutlierFactor()
def outlier_detect(amount, neigh):
    if (len(amount)>neigh):
        clf.n_neighbors = neigh
    elif (len(amount)==1):
        return [1]
    else:
        clf.n_neighbors = len(amount)-1

    yy = clf.fit_predict(np.array(amount).reshape(-1, 1))
    return yy
    

df['ol_cust_cat_amount'] = df.groupby(['customer','category'])['amount'].transform(outlier_detect,30)
df['ol_cust_mer_amount'] = df.groupby(['customer','merchant'])['amount'].transform(outlier_detect,30)
df['ol_cust_tl_amount'] = df.groupby(['customer'])['amount'].transform(outlier_detect,30)
df['ol_cat_amount'] = df.groupby(['category'])['amount'].transform(outlier_detect,30)
df['ol_mer_amount'] = df.groupby(['merchant'])['amount'].transform(outlier_detect,30)
'''

def outlier_detect_boxplot(amount):
    median = np.median(amount)
    std = np.std(amount)
    upper_quartile = np.percentile(amount, 77) #75
    lower_quartile = np.percentile(amount, 23) #25

    iqr = upper_quartile - lower_quartile
    upper_whisker = amount[amount<=upper_quartile+1.5*iqr].max()
    lower_whisker = amount[amount>=lower_quartile-1.5*iqr].min()
    return [0 if (x<upper_whisker and x>lower_whisker) else 1 for x in amount]
   

df['ol_cust_cat_amount'] = df.groupby(['customer','category'])['amount'].transform(outlier_detect_boxplot)
df['ol_cust_mer_amount'] = df.groupby(['customer','merchant'])['amount'].transform(outlier_detect_boxplot)
df['ol_cust_tl_amount'] = df.groupby(['customer'])['amount'].transform(outlier_detect_boxplot)
df['ol_cat_amount'] = df.groupby(['category'])['amount'].transform(outlier_detect_boxplot)
df['ol_mer_amount'] = df.groupby(['merchant'])['amount'].transform(outlier_detect_boxplot)


# In[181]:


df[['fraud','ol_cust_cat_amount','ol_cust_mer_amount','ol_cust_tl_amount','ol_cat_amount','ol_mer_amount']].corr()
#df.corr()



df2 = df
y = df2["fraud"]

# keep original values before standartization
keep_orig_columns = ['ol_cust_cat_amount','ol_cust_mer_amount','ol_cust_tl_amount','ol_cat_amount','ol_mer_amount']
keep_as_is_df = df2[keep_orig_columns]

cat_df=pd.get_dummies(df2["category"], columns=["category"])

df2 = df2.drop(["fraud", 'customer', 'step', 'gender', 'category'], axis=1)
df2 = df2.drop(keep_orig_columns, axis=1)

scaled_features = preprocessing.StandardScaler().fit_transform(df2.values)
df2 = pd.DataFrame(scaled_features, index=df2.index, columns=df2.columns)

X = pd.concat([df2,cat_df,keep_as_is_df], axis=1)
X.head()


# In[211]:



# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=len(X.columns.values), kernel_initializer=init, activation='relu'))
	model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

seed = 7

model = KerasClassifier(build_fn=create_model, verbose=1)
#---------------------------------------------
#optimizers = ['rmsprop', 'adam']
#init = ['glorot_uniform', 'normal', 'uniform']
#epochs = [50, 100, 150]



model.fit(X, y, batch_size = 124, 
          epochs = 5, validation_split = 0.25)


'''
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

scoring = {'AUC': make_scorer(roc_auc_score), 'Accuracy': make_scorer(accuracy_score)}
gs = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                    scoring=scoring, 
                    return_train_score=True, 
                    cv=4, 
                    refit='Accuracy')

start_time = time.time()
gs.fit(X, y,verbose=1)
elapsed_time = time.time() - start_time
print("ellapsed time", elapsed_time)

# summarize results
print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
means = gs.cv_results_['mean_test_Accuracy']
stds = gs.cv_results_['std_test_Accuracy']
params = gs.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
'''