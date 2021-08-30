#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
from collections import defaultdict
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, make_scorer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import *

import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV

# In[2]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def print_confusion_matrix(confusion_matrix, class_names, figsize = (16,9), fontsize=14, modelname=None):
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        confusion_matrix, columns=class_names, index = class_names 
    )
    if modelname:
        df_cm.to_csv('tephra_results/confusionMatrices/'+modelname+'_confmatscores.csv')
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,ha='right', fontsize=fontsize)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#     os.mkdir('confusion')
    if modelname:
        plt.savefig('tephra_results/confusionMatrices/'+modelname+"cm-5f.svg")
    return fig

def print_probability_matrix(confusion_matrix, class_names, figsize = (70,50), fontsize=14, modelname=None):
    confusion_matrix = confusion_matrix
    df_cm = pd.DataFrame(
        confusion_matrix, index=range(0,len(confusion_matrix)), columns=class_names, 
    )
    if modelname:
        df_cm.to_csv('tephra_results/TestProbs/'+modelname+"_test_probabilities.csv")
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    
    plt.ylabel('Data index')
    plt.xlabel('Predicted label')
    if modelname:
        plt.savefig('tephra_results/TestResults/'+modelname+"_TestPlot.svg")
    return fig

def eval_test(model, modelname):
    tstart = time.time()
    model.fit(X, y)
    train = time.time() - tstart
    pstart = time.time()
    preds = model.predict(test)
    predict = time.time() - pstart
    if modelname!="SVM":
        print(modelname)
        predsprob = model.predict_proba(test)
        print_probability_matrix(predsprob, trfdct.keys(), modelname=modelname)
    return preds, modelname, train, predict


# In[3]:


tuned_models = dict()

tuned_models['xgb'] = {
"colsample_bytree": 0.49346938775510206,
"learning_rate": 0.12199931486297749,
"max_depth": 7,
# 'num_leaves': (2**7)-1,
'eval_metric':'merror',
'use_label_encoder':False,
"n_estimators": 250,
"n_jobs": 8,
"random_state": 1337,
"reg_alpha": 0.34897959183673464,
"reg_lambda": 0.8387755102040816,
"subsample": 0.8122448979591836,
    #8123
}

tuned_models['lgb'] = {
'subsample': 0.9242424242424243,
'random_state': 1337,
'num_leaves': (2**7)-1,
'n_estimators': 250,
'max_depth': 7,
'max_bin': 32,
'learning_rate': 0.08633904519421778,
'colsample_bytree': 0.54,
'boosting_type': 'gbdt'
#8160
}
tuned_models['rfc'] = {
    "criterion": 'gini',
    "max_depth": None,
    "max_features": 'log2',
    "min_samples_leaf": 1,
    "min_samples_split": 3,
    "n_estimators": 500,
    "n_jobs": 8,
    "random_state":1337
}
tuned_models['cat'] = {
    "iterations":1000, 
#     "learning_rate":0.02,
#     "task_type":"CPU",
    "loss_function": 'MultiClass',
#     "rsm":0.8,
#     "subsample":0.8,
#     "bootstrap_type":"Poisson",
#     "depth":6,
#     "auto_class_weights": "SqrtBalanced",
    'border_count': 32,
    "random_seed":1337,
    "verbose":False,
#     "task_type": "GPU",
#     "devices":'0:1',
}
tuned_models["svm"] = {"probability": False, "gamma": 0.00001, "C": 10000, "random_state":13}
tuned_models["svmp"] = {"probability": True, "gamma": 0.00001, "C": 10000, "random_state":13}
tuned_models["knn"] = {"n_neighbors": 8} 
tuned_models["lda"] = {"tol": 0.0001, "solver": "svd"}
tuned_models["ann"] = {"solver": "adam", "max_iter": 1000, "learning_rate": "adaptive", "hidden_layer_sizes": [100], "alpha": 0.0001, "activation": "tanh"}


def newListRemove(element, list):
  list.remove(element)
  return list

# In[4]:


RFC = Pipeline([('scaler', RobustScaler()),
               ('RFC', RandomForestClassifier(**tuned_models['rfc']))])

SVM = Pipeline([('scaler', RobustScaler()),
               ('SVM', SVC(**tuned_models["svm"]))])

SVMP = Pipeline([('scaler', RobustScaler()),
               ('SVMP', SVC(**tuned_models["svmp"]))])

LGB = Pipeline([('scaler', RobustScaler()), 
                ('LGB', LGBMClassifier(**tuned_models['lgb']))])
XGB = Pipeline([('scaler', RobustScaler()), 
                ('LGB', xgb.XGBClassifier(**tuned_models['xgb']))])

CAT = Pipeline([('scaler', RobustScaler()),
               ('CAT', CatBoostClassifier(**tuned_models['cat']))])

VTC = Pipeline([('scaler', RobustScaler()), 
                ('Voting', VotingClassifier(estimators=[('CAT', CAT), ('xgb', XGB), ('LGB', LGB)], voting='soft'))])
            
KNN = Pipeline([('scaler', RobustScaler()),
               ('KNN', KNeighborsClassifier(**tuned_models["knn"]))])  

NB = Pipeline([('scaler', MinMaxScaler()),
               ('NB', ComplementNB(fit_prior=False))])
               
MLP = Pipeline([('scaler', RobustScaler()),
               ('ANN', MLPClassifier(**tuned_models["ann"]))])

LDA = Pipeline([('scaler', RobustScaler()),
               ('LDA', LinearDiscriminantAnalysis(**tuned_models["lda"]))])     

# In[5]:


def get_models():
    models, names = list(), list()
#     SVC
    models.append(SVMP)
    names.append("SVMP")
#     SVCP
    models.append(SVM)
    names.append("SVM")
    RFC
    models.append(RFC)
    names.append('RFC')
#     xgb
    models.append(XGB)
    names.append('XGB')
#     lgbm
    models.append(LGB)
    names.append("LGBM")
#     CAT
    models.append(CAT)
    names.append("CAT")
#     Voting - xgb bag rfc
    models.append(VTC)
    names.append("CAT-XGB-LGB")
#     KNN
    models.append(KNN)
    names.append("KNN")
#     NaiveBayes
    models.append(NB)
    names.append("NB")
#     ArtificialNN
    models.append(MLP)
    names.append("ANN")
#     LinearDis
    models.append(LDA)
    names.append("LDA")    
    return models, names


if __name__ == "__main__":
    print("####### PROCESSING DATA #######\n")
#     trfdct = {'AEGINA':0,'ANTIPAROS':1,'CHIOS':2,'CHRISTIANA ISLANDS':3,'KIMOLOS':4,'KOS':5,'LESBOS':6,'LICHADES ISLANDS':7,'LIMNOS (LEMNOS)':8,'METHANA':9,'MILOS':10,'NISYROS':11,'NWAVA':12,'PAROS':13,'PATMOS':14,'SAMOS':15,'SANTORINI':16,'YALI':17}
    trfdct = {'AEGINA':0,'ANTIPAROS':1,'KOS':2,'METHANA':3,'MILOS':4,'NISYROS':5,'SANTORINI':6,'YALI':7}
    train = pd.read_excel('preprocessed_train2.xls')
    test = pd.read_excel('preprocessed_test1.xls')
    train.drop('Unnamed: 0', axis=1, inplace=True)
    test.drop('Unnamed: 0', axis=1, inplace=True)
#     train = reduce_mem_usage(train)
#     test = reduce_mem_usage(test)
    y = train['LOCATION'].apply(lambda x: trfdct[x])
    X = train
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    test.replace([np.inf, -np.inf], np.nan,inplace=True)
    X.fillna(0, inplace=True)
    test.fillna(0, inplace=True)
    X = X.drop(['LOCATION'], axis=1)
    X = X.reset_index(drop = True)
    y = y.reset_index(drop = True)
    X = np.ascontiguousarray(X)
    
    
#     elapsed_time_df = pd.DataFrame(columns=["model_name", "training_time", ""])
    models, names = get_models()
    results = defaultdict(lambda: defaultdict(int))
    scoring = {
               'kappa': make_scorer(cohen_kappa_score),
               'acc': 'accuracy',
               'f1_m': 'f1_macro',
               'f1_w': 'f1_weighted',
               'auc_ovr': 'roc_auc_ovr',
              }
    print("####### CROSS VALIDATING #######\n")
    start = time.time()
    for rndint in np.random.randint(0,10000, 10):
        print("loop1: " + str(time.time() - start))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rndint)
        for i in range(len(models)):
            scores = []
            if names[i] not in ["CAT", "CAT-XGB-LGB", "LGB-CAT", "RFC-CAT", "NB", "SVM"]:
                scores = cross_validate(models[i], X, y, scoring=scoring, cv=cv, n_jobs=-1)
            if names[i] == "SVM":
#                 print("geldi yine")
                sc={
               'kappa': make_scorer(cohen_kappa_score),
               'acc': 'accuracy',
               'f1_m': 'f1_macro',
               'f1_w': 'f1_weighted',
#                'auc_ovr': 'roc_auc_ovr',
              }
                scores = cross_validate(models[i], X, y, scoring=sc, cv=cv, n_jobs=-1)
            else:
                scores = cross_validate(models[i], X, y, scoring=scoring, cv=cv)
            acc = scores["test_acc"].mean()
            f1_m = scores["test_f1_m"].mean()
            f1_w = scores["test_f1_w"].mean()
            kappa = scores["test_kappa"].mean()
            
            if names[i] == "SVM":
                scores = [acc, f1_m, f1_w, kappa]
            else:
                auc = scores["test_auc_ovr"].mean()
                scores = [acc, f1_m, f1_w, auc, kappa]
            name=names[i]
            results[name][rndint] = scores     
            print(name +", " + str(rndint) +" : acc, f1_m, f1_w : " + str((acc, f1_m, f1_w, auc)))
    elapsed = time.time() - start
    print("cv took: ", elapsed)
    results = pd.DataFrame(results)
    
    acc  = [results["SVMP"].apply(lambda x: x[0]),results["SVM"].apply(lambda x: x[0]), results["RFC"].apply(lambda x: x[0]),results["XGB"].apply(lambda x: x[0]),results["LGBM"].apply(lambda x: x[0]),results["CAT"].apply(lambda x: x[0]),results["CAT-XGB-LGB"].apply(lambda x: x[0]),results["KNN"].apply(lambda x: x[0]),results["ANN"].apply(lambda x: x[0]),results["LDA"].apply(lambda x: x[0]),results["NB"].apply(lambda x: x[0]),]
    f1_m = [results["SVMP"].apply(lambda x: x[1]),results["SVM"].apply(lambda x: x[1]),results["RFC"].apply(lambda x: x[1]),results["XGB"].apply(lambda x: x[1]),results["LGBM"].apply(lambda x: x[1]),results["CAT"].apply(lambda x: x[1]),results["CAT-XGB-LGB"].apply(lambda x: x[1]),results["KNN"].apply(lambda x: x[1]),results["ANN"].apply(lambda x: x[1]),results["LDA"].apply(lambda x: x[1]),results["NB"].apply(lambda x: x[1]),]
    f1_w = [results["SVMP"].apply(lambda x: x[2]),results["SVM"].apply(lambda x: x[2]),results["RFC"].apply(lambda x: x[2]),results["XGB"].apply(lambda x: x[2]),results["LGBM"].apply(lambda x: x[2]),results["CAT"].apply(lambda x: x[2]),results["CAT-XGB-LGB"].apply(lambda x: x[2]),results["KNN"].apply(lambda x: x[2]),results["ANN"].apply(lambda x: x[2]),results["LDA"].apply(lambda x: x[2]),results["NB"].apply(lambda x: x[2]),]
    auc  = [results["SVMP"].apply(lambda x: x[3]),results["RFC"].apply(lambda x: x[3]),results["XGB"].apply(lambda x: x[3]),results["LGBM"].apply(lambda x: x[3]),results["CAT"].apply(lambda x: x[3]),results["CAT-XGB-LGB"].apply(lambda x: x[3]),results["KNN"].apply(lambda x: x[3]),results["ANN"].apply(lambda x: x[3]),results["LDA"].apply(lambda x: x[3]),results["NB"].apply(lambda x: x[3]),]
    kappa  = [results["SVMP"].apply(lambda x: x[4]),results["SVM"].apply(lambda x: x[3]),results["RFC"].apply(lambda x: x[4]),results["XGB"].apply(lambda x: x[4]),results["LGBM"].apply(lambda x: x[4]),results["CAT"].apply(lambda x: x[4]),results["CAT-XGB-LGB"].apply(lambda x: x[4]),results["KNN"].apply(lambda x: x[4]),results["ANN"].apply(lambda x: x[4]),results["LDA"].apply(lambda x: x[4]),results["NB"].apply(lambda x: x[4]),]
    
    print("################### ACCURACY ############")
    print("SVMP ACCURACY : {}".format(results["SVMP"].apply(lambda x: x[0]).mean()))
    print("SVM ACCURACY : {}".format(results["SVM"].apply(lambda x: x[0]).mean()))
    print("RFC ACCURACY : {}".format(results["RFC"].apply(lambda x: x[0]).mean()))
    print("XGB ACCURACY : {}".format(results["XGB"].apply(lambda x: x[0]).mean()))
    print("LGBM ACCURACY : {}".format(results["LGBM"].apply(lambda x: x[0]).mean()))
    print("CAT ACCURACY : {}".format(results["CAT"].apply(lambda x: x[0]).mean()))
    print("CAT-XGB-LGB ACCURACY : {}".format(results["CAT-XGB-LGB"].apply(lambda x: x[0]).mean()))
    print("KNN : {}".format(results["KNN"].apply(lambda x: x[0]).mean()))
    print("NaiveBayes : {}".format(results["NB"].apply(lambda x: x[0]).mean()))
    print("ANN : {}".format(results["ANN"].apply(lambda x: x[0]).mean()))
    print("LDA : {}".format(results["LDA"].apply(lambda x: x[0]).mean()))
    
    print("################### F1-MACRO ############")
    print("SVMP F1-MACRO : {}".format(results["SVMP"].apply(lambda x: x[1]).mean()))
    print("SVM F1-MACRO : {}".format(results["SVM"].apply(lambda x: x[1]).mean()))
    print("RFC F1-MACRO : {}".format(results["RFC"].apply(lambda x: x[1]).mean()))
    print("XGB F1-MACRO : {}".format(results["XGB"].apply(lambda x: x[1]).mean()))
    print("LGBM F1-MACRO : {}".format(results["LGBM"].apply(lambda x: x[1]).mean()))
    print("CAT F1-MACRO : {}".format(results["CAT"].apply(lambda x: x[1]).mean()))
    print("CAT-XGB-LGB F1-MACRO : {}".format(results["CAT-XGB-LGB"].apply(lambda x: x[1]).mean()))
    print("KNN : {}".format(results["KNN"].apply(lambda x: x[1]).mean()))
    print("NaiveBayes : {}".format(results["NB"].apply(lambda x: x[1]).mean()))
    print("ANN : {}".format(results["ANN"].apply(lambda x: x[1]).mean()))
    print("LDA : {}".format(results["LDA"].apply(lambda x: x[1]).mean()))

    print("################### F1-WEIGTHED ############")
    print("SVMP F1-WEIGHTED : {}".format(results["SVMP"].apply(lambda x: x[2]).mean()))
    print("SVM F1-WEIGHTED : {}".format(results["SVM"].apply(lambda x: x[2]).mean()))
    print("RFC F1-WEIGHTED : {}".format(results["RFC"].apply(lambda x: x[2]).mean()))
    print("XGB F1-WEIGHTED : {}".format(results["XGB"].apply(lambda x: x[2]).mean()))
    print("LGBM F1-WEIGHTED : {}".format(results["LGBM"].apply(lambda x: x[2]).mean()))
    print("CAT F1-WEIGHTED : {}".format(results["CAT"].apply(lambda x: x[2]).mean()))
    print("CAT-XGB-LGB F1-WEIGHTED : {}".format(results["CAT-XGB-LGB"].apply(lambda x: x[2]).mean()))
    print("KNN : {}".format(results["KNN"].apply(lambda x: x[2]).mean()))
    print("NaiveKNNBayes : {}".format(results["NB"].apply(lambda x: x[2]).mean()))
    print("ANN : {}".format(results["ANN"].apply(lambda x: x[2]).mean()))
    print("LDA : {}".format(results["LDA"].apply(lambda x: x[2]).mean()))

    print("################### ROC-AUC ############")
    print("SVMP ROC-AUC : {}".format(results["SVMP"].apply(lambda x: x[3]).mean()))
#     print("SVM ROC-AUC : {}".format(results["SVM"].apply(lambda x: x[3]).mean()))
    print("RFC ROC-AUC : {}".format(results["RFC"].apply(lambda x: x[3]).mean()))
    print("XGB ROC-AUC : {}".format(results["XGB"].apply(lambda x: x[3]).mean()))
    print("LGBM ROC-AUC : {}".format(results["LGBM"].apply(lambda x: x[3]).mean()))
    print("CAT ROC-AUC : {}".format(results["CAT"].apply(lambda x: x[3]).mean()))
    print("CAT-XGB-LGB ROC-AUC : {}".format(results["CAT-XGB-LGB"].apply(lambda x: x[3]).mean()))
    print("KNN : {}".format(results["KNN"].apply(lambda x: x[3]).mean()))
    print("NaiveBayes : {}".format(results["NB"].apply(lambda x: x[3]).mean()))
    print("ANN : {}".format(results["ANN"].apply(lambda x: x[3]).mean()))
    print("LDA : {}".format(results["LDA"].apply(lambda x: x[3]).mean()))
    
    print("################### KAPPA ############")
    print("SVMP KAPPA-SCORE : {}".format(results["SVMP"].apply(lambda x: x[4]).mean()))
    print("SVM KAPPA-SCORE : {}".format(results["SVM"].apply(lambda x: x[3]).mean()))
    print("RFC KAPPA-SCORE : {}".format(results["RFC"].apply(lambda x: x[4]).mean()))
    print("XGB KAPPA-SCORE : {}".format(results["XGB"].apply(lambda x: x[4]).mean()))
    print("LGBM KAPPA-SCORE : {}".format(results["LGBM"].apply(lambda x: x[4]).mean()))
    print("CAT KAPPA-SCORE : {}".format(results["CAT"].apply(lambda x: x[4]).mean()))
    print("CAT-KAPPA-SCORE ROC-AUC : {}".format(results["CAT-XGB-LGB"].apply(lambda x: x[4]).mean()))
    print("KNN : {}".format(results["KNN"].apply(lambda x: x[4]).mean()))
    print("NaiveBayes : {}".format(results["NB"].apply(lambda x: x[4]).mean()))
    print("ANN : {}".format(results["ANN"].apply(lambda x: x[4]).mean()))
    print("LDA : {}".format(results["LDA"].apply(lambda x: x[4]).mean()))
    
    print("################### SAVING SCORE PLOTS ############\n")
    names = ['SVMP','SVM','RFC','XGB','LGBM','CAT','CAT-XGB-LGB','KNN','ANN','LDA','NB',]
    names2 = ['SVMP','RFC','XGB','LGBM','CAT','CAT-XGB-LGB','KNN','ANN','LDA','NB',]
    plt.figure(figsize=(16,9))
    fig = plt.figure()
    #fig.suptitle('Accuracy Comparison')
    ax1 = fig.add_subplot(221)
    # plt.yticks(yticks)
    plt.boxplot(acc,showfliers=False,vert=False,autorange=True)
    # ax.set_yticks(yticks)
    ax1.text(0.95, 0.7, 'A', verticalalignment='bottom', horizontalalignment='left', fontweight='bold', fontsize=15)
    ax1.set_yticklabels(names)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_xlim(0.65,1)
    #plt.savefig('acc.svg')
    #plt.show()

    #plt.figure(figsize=(16,9))
    #fig = plt.figure()
    #fig.suptitle('F1-Weighted Score Comparison')
    ax2 = fig.add_subplot(222)
    plt.boxplot(f1_w,showfliers=False,vert=False,autorange=True)
    #ax2.set(ylabel=None)
    ax2.set_xlim(0.65,1)
    ax2.text(0.98, 0.7, 'B', verticalalignment='bottom', horizontalalignment='left', fontweight='bold', fontsize=15)
    # ax.set_yticks(yticks)
    ax2.set_yticklabels(names)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #plt.savefig('f1-w.svg')
    #plt.show()

    #plt.figure(figsize=(16,9))
    #fig = plt.figure()
    #fig.suptitle('ROC-AUC Score Comparison')
    ax3 = fig.add_subplot(223)
    plt.boxplot(auc,showfliers=False,vert=False,autorange=True)
    # ax.set_yticks(yticks)
    ax3.set_yticklabels(names2)
    ax3.set_xlim(0.85,1)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.text(0.98, 0.86, 'C', verticalalignment='bottom', horizontalalignment='left', fontweight='bold', fontsize=15)
    #plt.savefig('roc-auc.svg')
    #plt.show()
    
    #plt.figure(figsize=(16,9))
    #fig = plt.figure()
    #fig.suptitle('Kappa Score Comparison')
    ax4 = fig.add_subplot(224)
    plt.boxplot(kappa,showfliers=False,vert=False, autorange=True)
    #ax4.set(ylabel=None)
    #ax4.set_yticks(yticks)
    ax4.set_yticklabels(names)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.set_xlim(0.55,1)
    ax4.text(0.98, 0.6, 'D', verticalalignment='bottom', horizontalalignment='left', fontweight='bold', fontsize=15)
    plt.savefig('acc_test.svg')
    plt.show()
                                                                          
    print("####### CREATING CROSS-VALIDATION CONFUSION MATRICES #######\n")
    cms1 = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    scoring = {'acc': 'accuracy',
               'f1_m': 'f1_macro',
               'f1_w': 'f1_weighted',
               'auc_ovr': 'roc_auc_ovr',
               'kappa': make_scorer(cohen_kappa_score)
              }
    for i in range(len(models)):
        if names[i] not in ["CAT", "CAT-XGB-LGB", "LGB-CAT"]:
            preds = cross_val_predict(models[i], X, y, cv=cv, n_jobs=-1)
        else:
            preds = cross_val_predict(models[i], X, y, cv=cv)
        cm = confusion_matrix(y, preds)
        cms1[names[i]] = cm

    for i in range(len(models)):
        print_confusion_matrix(cms1[names[i]], trfdct, modelname=names[i])

    time_list = []
    print("####### PREDICTING TEST DATA #######\n")
    for i in range(len(models)):
        print("Running for: ", names[i])
        preds, modelname, train, predict = eval_test(models[i], names[i])
        time_list.append([modelname,train,predict])
#         os.mkdir('predictions')
        with open('tephra_results/TestResults/' + names[i]+"new_test_predictions.txt", "w") as text_file:
            text_file.write((str(preds)))
    time_df = pd.DataFrame(time_list, columns=["model_name", "training_time", "evaluation_time"])
    time_df.to_csv('tephra_results/model_times.csv', index=False)
    print("####### FINISHED!  #######\n")
