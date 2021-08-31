# model tuning playground

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import json
from collections import defaultdict

from tephro_script import reduce_mem_usage

xgb_model = xgb.XGBClassifier()
rfc_model = RandomForestClassifier()
lgb_model = LGBMClassifier()
svc_model = SVC()
cat_model = CatBoostClassifier()
svcp_model = SVC()
knn_model = KNeighborsClassifier()
lda_model = LinearDiscriminantAnalysis()
nb_comp = ComplementNB()
ann_model = MLPClassifier()

xgb_parames = {'nthread':[-1], #when use hyperthread, xgboost may become slower,
              'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)), #so called `eta` value
              'max_depth': list(range(0,10)),
              'min_child_weight': list(range(0, 10)),
              'subsample': list(np.linspace(0, 1, 1)),
              'colsample_bytree': list(np.linspace(0, 1, 1)),
              'n_estimators': list(range(20, 150, 10)), #number of trees, change it to 1000 for better results
              'missing':[-999],
              'reg_alpha': list(np.linspace(0, 1, 1)),
              'reg_lambda': list(np.linspace(0, 1, 1)),
              'seed': [1337]}

rfc_params = {'criterion': ['entropy'],
            #     'max_depth': list(range(0,10)),
            'max_features': list(np.linspace(0.5, 1)),
#             'max_leaf_nodes': list(range(0,10)),
#             'max_samples': list(np.linspace(0.4,1)),
#             'min_impurity_decrease': list(np.linspace(0,1)),
            'min_samples_leaf': list(range(1,20)),
            'min_samples_split': list(range(1,40)),
#             'min_weight_fraction_leaf': list(np.linspace(0,1)),
            'n_estimators': [100],
            'n_jobs': [-1]
            }

lgb_params = {
            'boosting_type': ['gbdt', 'goss', 'dart'],
            'num_leaves': list(range(20, 150)),
            'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
            'subsample_for_bin': list(range(20000, 300000, 20000)),
            'min_child_samples': list(range(20, 500, 5)),
        #     'reg_alpha': list(np.linspace(0, 1)),
        #     'reg_lambda': list(np.linspace(0, 1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'subsample': list(np.linspace(0.5, 1, 100)),
            'is_unbalance': [True, False],
            'n_estimators':[100],
            'num_thread': [-1]
}

svc_params = {'C': [0.1,1, 10, 100,1000,10000], 'gamma': [100,10,1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid'],'tol': [1e-3,1e-4,1e-5,1e-2,1e-1],'probability': ["True]}

svcp_params = {'C': [0.1,1, 10, 100,1000,10000], 'gamma': [100,10,1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid'],'tol': [1e-3,1e-4,1e-5,1e-2,1e-1],'probability': ["False]}

xgb_params = {
            'num_leaves': list(range(10, 150)),
            'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
#             'subsample_for_bin': list(range(20000, 300000, 20000)),
            'min_child_samples': list(range(1, 500, 1)),
            'reg_alpha': list(np.linspace(0, 1)),
            'reg_lambda': list(np.linspace(0, 1)),
            'colsample_bytree': list(np.linspace(0.6, 1, 10)),
            'subsample': list(np.linspace(0.5, 1, 100)),
            'is_unbalance': [True, False],
            'n_estimators': [100],
            'num_thread': [-1]
}

cat_params = {
    "iterations":[5000], 
#     "learning_rate":list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
#     "l2_leaf_reg":list(np.linspace(0, 10, 20)),
#     "depth":list(range(4, 12)),
    "loss_function": ['MultiClass'],
#     'border_count': [32],
#     "use_best_model":True,
#     "random_seed":[1337],
#     "silent":True,
#     "verbose":[5000],
#     "task_type": ["GPU"],
#     "leaf_estimation_iterations":[1],
#     "thread_count":[1]
}

lda_params = {"solver" : ["svd"],
              "tol" : [0.0001,0.0002,0.0003]
             }

knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11] # usually odd numbers
             }

nb_params = {
    
            }

ann_params = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (32,64,128), (100,100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.0001, 0.0005, 0.001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [500,650,800,1000]
             }

if __name__ == "__main__":
    print("####### PROCESSING DATA #######\n")

    trfdct = {'AEGINA':0,'ANTIPAROS':1,'KOS':2,'METHANA':3,'MILOS':4,'NISYROS':5,'SANTORINI':6,'YALI':7}
    train = pd.read_excel('preprocessed_train2.xls')
    test = pd.read_excel('preprocessed_test1.xls')
    train.drop('Unnamed: 0', axis=1, inplace=True)
    test.drop('Unnamed: 0', axis=1, inplace=True)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
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
    
    cv = StratifiedKFold(5)
    models = [rfc_model, lgb_model, svc_model,  svcp_model, knn_model, lda_model, nb_model, ann_model]
    names = ["rfc", "lgb", "svc", "svcp", "knn", "lda", "nb", "ann"]
    params = [rfc_params, lgb_params, svc_params,  svcp_params, knn_params, lda_params, nb_params, ann_params]
    tuned_models=defaultdict()
    for i in range(len(models)):
        rs_clf = RandomizedSearchCV(models[i], params[i], n_iter=250,
                                    n_jobs=1, verbose=2, cv=cv,
                                    scoring='f1_macro', refit=False, random_state=1337)


        rs_clf.fit(X, y)
        best_score = rs_clf.best_score_
        best_params = rs_clf.best_params_
        print("Best score for {}: {}".format(names[i], best_score))
        print("Best params for {}: ".format(names[i]))
        for param_name in sorted(best_params.keys()):
            print('%s: %r' % (param_name, best_params[param_name]))
            tuned_models[names[i]] = best_params
    
    with open('tuned_models3.txt', 'w') as f:
        f.write(json.dumps(tuned_models))