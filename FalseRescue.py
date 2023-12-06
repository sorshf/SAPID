#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""Performing machine learning experiments with data sampling as outlined in Chapter 4 of my Master's thesis."""
# =============================================================================
# Imports
from numpy.core.fromnumeric import size
import feature_names
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, precision_score, f1_score, recall_score
LogTransformer = FunctionTransformer(np.log1p)
import itertools as iter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as ss
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import pingouin as pg
from pingouin import ttest
plt.rcParams.update({'font.size': 15})
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from Constants import decoy_prefix

import matplotlib.patches as mpatches


def get_new_classification(clf_scheme, original_df, downsampled_df, testing=True):
    """Returns X, y values according to the False Negative rescue (obj2.1) or False Positive Rescue (obj2.2) problem

    Args:
        clf_scheme ([string]): "Obj2.1" for the False Negative Rescue, "Obj2.2" for the False Positive Rescue
        original_df ([pandas.dataframe]): Dataframe of the original non-downsampled dataset
        downsampled_df ([pandas.dataframe]): Dataframe of the downsampled dataset
        testing (bool, optional): Whether the dataset is for testing or not. Defaults to True.

    Returns:
        [pandas.dataframe, pandas.series]: The X and y corresponding to the new dataset and labels, respectively.
    """

    #Format index by adding pixel name to the accession numbers so each protein at each pixel is identified
    def format_index(df):
        df1 = df.copy()
        df1.index = df1.apply(lambda x: str(x.POI_name) +":"+ x.accession, axis=1)
        return df1
    
    omega_df = format_index(original_df).copy()
    omega = set(omega_df.index)
    P_original = set(omega_df[omega_df.target==1].index)
    
    downsampled_df = format_index(downsampled_df).copy()
    omega_ds = set(downsampled_df.index)
    P_ds = set((downsampled_df[downsampled_df.target==1].index))
    
    if clf_scheme == "Obj2.1":
        downsampled_df = downsampled_df[~downsampled_df.index.isin(P_ds)]
        downsampled_df["target"] = np.where(downsampled_df.index.isin(P_original), 1, 0)

    elif clf_scheme == "Obj2.2":
        downsampled_df = downsampled_df[downsampled_df.index.isin(P_ds)]
        downsampled_df["target"] = np.where(downsampled_df.index.isin(P_original), 0, 1)

    X, y = pre_process_features(downsampled_df, spatial="All_features",
                            testing=testing, target_column="target", preprocessing="MinMaxScale")
    return X, y
    

def figure_4_2_and_4_3():
    """Perform hierarchical clustering on the False Negative and False Positive rescue problems
    """
    directory="C:/TPP/data/soroush/output_spatial_data/"
    original_training_data_name = "training_0.00_BIG_.csv"

    original_data_training = pd.read_csv(directory+original_training_data_name)
    ds_data_training = pd.read_csv(directory+"training_"+"0.50"+"_BIG_.csv")

    for obj in ["Obj2.1","Obj2.2"]:

        X_train, y_train = get_new_classification(obj, original_data_training, ds_data_training)

        # Get around 40000 samples with all the positives and rest negatives
        X_hc_positive = X_train[y_train == 1]
        X_hc_negative = X_train[y_train == 0].sample(frac=0.25, random_state=1)
        X_train = pd.concat([X_hc_positive,X_hc_negative])
        y_train = y_train[X_train.index]

        lut = dict(zip(y_train.unique(), ["#5c1a33","#749dae"]))

        row_colors = y_train.map(lut)
        col_colors = ["#876D57" if col in feature_names.POI_features_only else "#f7dc6a" for col in X_train.columns]

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10})
        fig = plt.figure()
        fig.patch.set_facecolor('white')

        cluster_map = sns.clustermap(X_train,row_colors=row_colors, col_colors=col_colors, figsize=(24,13.5))
        cluster_map.fig.subplots_adjust(right=0.7)
        #cluster_map.ax_cbar.set_position((0.8, 1, .03, .4))
        POI_features_patch = mpatches.Patch(color='#876D57', label='POI features')
        spatial_features_patch = mpatches.Patch(color='#f7dc6a', label='Spatial features')
        P_present_patch = mpatches.Patch(color='#5c1a33', label='FP by ProtProph')
        P_not_present_patch = mpatches.Patch(color='#749dae', label='TP by ProtProph')
        plt.legend(handles=[POI_features_patch,spatial_features_patch, P_present_patch, P_not_present_patch],bbox_to_anchor=(3.5, 0.5, 1.5, 0.5),fontsize=16)

        plt.savefig(f"Hierarchical_clustering_{obj}.png",facecolor=fig.get_facecolor(), edgecolor='none',dpi=150)

def pre_process_features(dataset_name, spatial, testing, target_column, preprocessing):
    """Preprocess the dataset before machine learning analysis

    Args:
        dataset_name ([string]): Complete path and name of the csv dataset outputted from 3_CreateDataset.py module
        spatial ([string]): The choices are "All_features","POI_features_only","Spatial_features_only"
        testing ([boolean]): Is this the testing dataset (If testing, the accession and protein prophet prob will be kept in the dataset
        target_column ([string]): Name of the column containing the labels (0 and 1) for the instances as determined by PP
        preprocessing ([string]): Type of preprocessing: "MinMaxScale","StandardScale","QuantileTransform","LogTransform", or nothing

    Returns:
        [(pandas.dataframe, pandas.series)]: X, y  for machine learning analysis
    """

    #Read the dataset
    if type(dataset_name) == str:
        data = pd.read_csv(dataset_name)
    else:
        data = dataset_name
    
    #Save the accession column, and PP prob if this is the testing set
    if testing:
        accession = data["accession"]
        pp_prob = data["protein_proph_prob"]
        pixel_instance_name = data.apply(lambda x: str(x.POI_name) +":"+ x.accession, axis=1)

    #Bad columns we won't use for classification
    not_usable_columns = ["accession","POI_name", "target", "protein_proph_prob"]
    if "New_Class" in data.columns:
        not_usable_columns.append("New_Class")
    
    #One nominal column that needs to be one-hot-encoded
    nominal_columns = ["mode_chargeState_POI"]
    
    #Get the example, remove the unusable columns
    X = data.drop(not_usable_columns, axis=1).copy()
    y = data[target_column].copy()
    
    #One-hot-encode the charges
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int)
    OHed_data = pd.DataFrame(OH_encoder.fit_transform(X[nominal_columns]),
                            index=X.index)
                            
    if OHed_data.shape[1] == 4:
        OHed_data.columns = ["Charge_2_POI","Charge_3_POI","Charge_4_POI","Charge_5_POI"]
        OHed_data["Charge_6_POI"] = 0
    else:
        OHed_data.columns = ["Charge_2_POI","Charge_3_POI","Charge_4_POI","Charge_5_POI","Charge_6_POI"]
      
    #Add the one-hot-encoded data
    X = X.join(OHed_data)
    
    #Drop the charge column
    X = X.drop(nominal_columns, axis=1)
        
        
    #Keep the features of interest
    X = keep_necessary_features(X, spatial)

    
    if preprocessing == "MinMaxScale":
        print("MinMaxScale")
        X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)
    elif preprocessing == "StandardScale":
        print("StandardScaler")
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    elif preprocessing == "QuantileTransform":
        print("QuantileTransform")
        X = pd.DataFrame(QuantileTransformer().fit_transform(X), columns=X.columns, index=X.index)
    elif preprocessing == "LogTransform":
        print("LogTransform")
        X = X.astype(float)
        X = pd.DataFrame(LogTransformer.transform(X), columns=X.columns, index=X.index)
    else:
        print("No transformation")
        pass



    #Add the protein prophet probabilities if this is a testing dataset
    if testing:
        X['protein_proph_prob'] = pp_prob
        X['accession'] = accession
        X["pixel_instance_name"] = pixel_instance_name

    return X, y


def keep_necessary_features(dataset, spatial):
    """Keep only the necessary features (columns) in the dataset

    Args:
        dataset ([pandas.dataframe]): Pandas dataframe containing the dataset
        spatial ([string]): The  choices are "All_features","POI_features_only","Spatial_features_only"

    Returns:
        [pandas.dataframe]: Pandas dataframe with appropriate features
    """
    if spatial == "All_features_old":
        return dataset[feature_names.All_features_old]

    if spatial == "All_features":
        return dataset[feature_names.All_features]

    elif spatial == "POI_features_only":
        return dataset[feature_names.POI_features_only]
    
    elif spatial == "Spatial_features_only":
        return dataset[feature_names.Spatial_features_only]
    
    elif spatial =="Non_binary_features":
        return dataset[feature_names.Non_binary_features]
    
    else:
        return dataset[[spatial]]


def validation_X_fold_obj(training_filename, testing_filename, files_directory, X_fold, classifiers_list, sampling_methods,  objective, decoy_string):
    """Perform training testing scheme (x-fold) with different classifiers and data sampling methods for both objectives for the 50% ds ratio dataset

    Args:
        training_filename ([string]): Name of the CSV dataset of the original training dataset
        testing_filename ([string]): Name of the CSV dataset of the original testing dataset
        files_directory ([string]): The path to the training and testing datasets (they should be in the same folder)
        X_fold ([int]): Number of folds that should be used for training testing scheme
        classifiers_list ([list]): List of classifiers name as defined in the funciton get_best_classifier()
        sampling_methods ([dict]): A dictionary with sampling names as keys and sampling objects as values
        objective ([string]): Either objective "Obj2.1" for the False Negative Rescue, or "Obj2.2" for the False Positive Rescue
        decoy_string ([string]): The decoy string used in database search decoy_prefix

    Returns:
        [(pandas.dataframe, dict)]: The performance data is in a dataframe. The ROC PR info (which we don't use) are in the second dictionary.
    """
    
    #We record the performance data in a dataframe
    performance_data = pd.DataFrame(columns=["value", "clf_type", "feature_type", "sampling_name", "metric", "fold"])

    #The TPR, FPR, and PRecision goes toa  dic for each fold and each set of training and testing
    ROC_PR_data = dict()

    #Create stratified classfication object 
    skf = StratifiedKFold(n_splits=X_fold)


    #Just reading the original csv datasets (training and testing files)
    original_data_training = pd.read_csv(files_directory+training_filename)
    original_data_testing = pd.read_csv(files_directory+testing_filename)

    #Downsampling ratio file we want to use
    #We chose to only consider the 0.50 ds ratio, it could be changed if necessary
    ds_ratio = "0.50"

    #FEature type
    feature_type = "All_features"

    #Reading the downsampled datasets
    ds_data_training = pd.read_csv(files_directory+"training_"+ds_ratio+"_BIG_.csv")
    ds_data_testing = pd.read_csv(files_directory+"testing_"+ds_ratio+"_BIG_.csv")

    #Defining feature types

    for sampling in sampling_methods:

        X_train, y_train = get_new_classification(objective, original_data_training, ds_data_training, testing=False)
        X_test, y_test = get_new_classification(objective, original_data_testing, ds_data_testing, testing=True)
        
        ros = sampling_methods[sampling]
        sampling_name= sampling
        if ros != 0:
            X_train, y_train = ros.fit_resample(X_train, y_train)

        #defining folds from training set for evaluation
        for fold, train_indexes, test_indexes in zip(range(skf.n_splits), skf.split(X_train, y_train), skf.split(X_test, y_test)):
            train_index = train_indexes[1]
            test_index = test_indexes[1]

            X_train_fold = X_train.iloc[train_index].copy()
            y_train_fold = y_train.iloc[train_index].copy()

            X_test_fold = X_test.iloc[test_index].copy()
            y_test_fold = y_test.iloc[test_index].copy()

      
            
            #defining classifiers to train test
            for clf_type in classifiers_list:

                trained_classifier = get_best_classifier(clf_type, X=X_train_fold, y=y_train_fold, scoring="f1")

                y_pred = trained_classifier.predict(X_test_fold.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))


                performance_dic = dict()
                performance_dic["test PRAUC"] = average_precision_score(y_test_fold, y_pred)
                performance_dic["test ROCAUC"] = roc_auc_score(y_test_fold, y_pred)
                performance_dic["test Precision"] = precision_score(y_test_fold, y_pred)
                performance_dic["test Recall"] = recall_score(y_test_fold, y_pred)
                performance_dic["test F1"] = f1_score(y_test_fold, y_pred)
                performance_dic["test MCC"] = matthews_corrcoef(y_test_fold, y_pred)

                tn, fp, _, _ = confusion_matrix(y_test_fold, y_pred).ravel()
                specificity = tn / (tn+fp)
                performance_dic["test Specificity"] = specificity



                TPR_list_clf, FPR_list_clf, precision_list_clf = get_ROC_PR_curve(trained_classifier, X_test_fold, y_test_fold, 100)


                ROC_PR_data[f"{clf_type}_{feature_type}_{sampling_name}_{fold+1}_TPR"] = TPR_list_clf
                ROC_PR_data[f"{clf_type}_{feature_type}_{sampling_name}_{fold+1}_FPR"] = FPR_list_clf
                ROC_PR_data[f"{clf_type}_{feature_type}_{sampling_name}_{fold+1}_Precision"] = precision_list_clf

                
                for key in performance_dic:
                    plot_data = pd.Series([performance_dic[key], clf_type, feature_type, sampling_name, key, fold+1], index=["value", "clf_type", "feature_type", "sampling_name", "metric", "fold"])
                    performance_data = performance_data.append(plot_data, ignore_index=True)
                
    return performance_data, ROC_PR_data



def get_best_classifier(clf_type, X, y, scoring):
    """Function that trains a specified type of classifier using grid search, and then returns the hyperparametertuned classifier.

    Args:
        clf_type ([string]): The type of classifier: "Logistic\nRegression", "LDA", "Gaussian\nNB", "Decision\nTree", "Dummy\nCLF"
        X ([pandas.dataframe]): Pandas dataframe with the training examples (X)
        y ([pandas.series]): Pandas series with the labels to the training examples
        scoring ([string]): The scoring function by whihc the best hyperparameter combinations are chosen (e.g. "f1", "accuracy". etc)

    Returns:
        [object]: A hyperparameter tuned classifier object that is already trained
    """

    if clf_type == "Logistic\nRegression":

        #Logistic Regression classifier
        clf = LogisticRegression(solver= "saga", random_state=1, max_iter=1000, n_jobs=-1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'penalty': ["l2", "elasticnet"],'C':[0.01, 0.1, 1.0, 10], 'l1_ratio': [0.25, 0.5, 0.75],
                        'class_weight':[None, "balanced"]}]
    
    elif clf_type == "LDA":
        
        #Logistic Regression classifier
        clf = LinearDiscriminantAnalysis(solver= "svd")

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'tol': [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5]}]
        
    elif clf_type == "QDA":
        
        #Logistic Regression classifier
        clf = QuadraticDiscriminantAnalysis()

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'tol': [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5]}]
        
        
    elif clf_type == "Gaussian\nNB":
        
        #Guassian naive base
        clf = GaussianNB()
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'var_smoothing': [1e-8, 1e-9, 1e-10]}]
        
    elif clf_type == "Multinomial\nNB":
        
        #Guassian naive base
        clf = MultinomialNB()
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'alpha': [0, 0.5, 1.0, 1.5, 2.0],
                       'fit_prior': [True, False]}]

    elif clf_type == "Linear\nSVM":

        clf = SVC(probability=True, random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'kernel': ['linear'], 'gamma': ["auto"],
                     'C': [0.01, 0.1, 1.0, 10],'class_weight':[None, "balanced"]}
                    ]
        
    elif clf_type == "RBF\nSVM":

        clf = SVC(probability=True, random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = [{'kernel': ['rbf'], 'gamma': ["auto"],
                     'C': [0.01, 0.1, 1.0, 10],'class_weight':[None, "balanced"]}
                    ]


    elif clf_type == "Decision\nTree":

        clf = DecisionTreeClassifier(random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)

        #Perform Grid search
        grid_values = {'criterion':['gini','entropy'],
                       'max_depth':[1,2,3,4,5, None],
                       'min_samples_split': [2,4,6,8],
                       'min_samples_leaf' : [1,2,3,5],
                       'class_weight':[None, "balanced"]}

    elif clf_type == "Random\nForest":

        clf = RandomForestClassifier(random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

        n_estimators = [50, 100, 200]
        max_depth = [2, 5,  10]
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [2, 5, 10] 

        grid_values = dict(n_estimators = n_estimators, max_depth = max_depth,  
            min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf,
             class_weight = [None, "balanced"])
    
    elif clf_type == "Neural\nNetwork":

        clf = MLPClassifier(random_state=1,solver="adam", verbose=False, early_stopping=True, activation="relu")

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)


        #Perform Grid search
        grid_values = [{'hidden_layer_sizes': [(5,5,5),(10,10,10),(5,10,10),(10,5,10),(10,10,5), 
                                               (5,5,10), (5,10,5), (10,5,5), (10,10,10,10), 
                                               (5,5,5,5), (10,5,10,5), (5,10,5,10), (20,10, 5, 5)], 
                        'alpha': [0.0001, 0.001, 0.01, 1, 10],
                       
                       "learning_rate_init": [0.001, 0.01, 0.1]}
                    ]

    elif clf_type == "Dummy\nCLF":
        
        clf = DummyClassifier()
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=2)
        
        grid_values = {"strategy": ["stratified"]}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring = scoring, refit = True, cv=skf, verbose=1,return_train_score=True, n_jobs=-1)
    grid_clf.fit(X, y)

    # Best parameters obtained with GridSearchCV:
    print(clf_type," Best parameters are: ",grid_clf.best_params_)
    print(clf_type," Best score is: ", grid_clf.best_score_)

    return grid_clf.best_estimator_


def train_test_many_clfs():
    """Perform 10 fold training testing with different data sampling and classifier types for both objectives, and store the results.
    """

    #A dictionary with the data sampling methods
    sampling_methods = {
    "Raw": 0,
    "Under\nSample\n1": RandomUnderSampler(random_state=1, sampling_strategy=1),
    "Under\nSample\n0.5": RandomUnderSampler(random_state=1, sampling_strategy=0.5),
    "Under\nSample\n0.1": RandomUnderSampler(random_state=1, sampling_strategy=0.1),
    
    "Over\nSample\n1": RandomOverSampler(random_state=1, sampling_strategy=1),
    "Over\nSample\n0.5": RandomOverSampler(random_state=1, sampling_strategy=0.5),
    "Over\nSample\n0.1": RandomOverSampler(random_state=1, sampling_strategy=0.1),
    
    "SMOTE\n1": SMOTE(random_state=1, sampling_strategy=1),
    "SMOTE\n0.5": SMOTE(random_state=1, sampling_strategy=0.5),
    "SMOTE\n0.1": SMOTE(random_state=1, sampling_strategy=0.1),
    
    "SMOTE\nENN": SMOTEENN(random_state=1),
    "SMOTE\nTomek": SMOTETomek(random_state=1),
    
    }

    #List of classifiers that needs to be run
    classifiers_list = [
        "Logistic\nRegression",
        "LDA",
        "QDA",
        "Gaussian\nNB",
        "Multinomial\nNB",
        "Linear\nSVM",
        "RBF\nSVM",
        "Decision\nTree",
        "Random\nForest",
        "Neural\nNetwork",
        "Dummy\nCLF"
    ]



    #perform the 10 fold training testing
    #Directory and name of the original datasets
    files_directory = "C:/TPP/data/soroush/output_spatial_data/"
    training_dataset_name = "training_0.00_BIG_.csv"
    testing_dataset_name = "testing_0.00_BIG_.csv"


    #Perform the datasampling techniques and training testing as explained in thesis

    for objective in ["Obj2.1", "Obj2.2"]:
        performance_data, ROC_PR_data = validation_X_fold_obj(training_dataset_name, testing_dataset_name, files_directory, X_fold=10,
                                                                classifiers_list=classifiers_list, sampling_methods=sampling_methods, 
                                                                objective=objective, decoy_string="_DECOY")


        pd.to_pickle(performance_data, f"{objective}_fold_validation.pkl")

        with open(f'{objective}_ROC_PR.pkl', 'wb') as handle:
            pickle.dump(ROC_PR_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_ROC_PR_curve(model, dataset, target, num_thresholds):
    """Calcualtes the ROC and PR curves x-axis and y-axis for a classifier.

    Args:
        model ([object]): Sklearn trained classifier
        dataset ([pandas.dataframe]): The dataset that the trained model needs to be tested on. It must contain the columns "protein_proph_prob",'accession','pixel_instance_name'.
        target ([pandas.series]): The labels for the examples in the dataset
        num_thresholds ([int]): Number of thresholds from 0 to 1 which should be used to calculate TPR, FPR, and Precision.

    Returns:
        [(list, list, list)]: TPR_list_clf, FPR_list_clf, precision_list_clf
    """
    thresholds = np.linspace(0,1,num_thresholds)
    TPR_list_clf = []
    FPR_list_clf = []
    precision_list_clf = []
    for thr in thresholds:
        TP = len([x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] >= thr, target, -1)) if x==1]) 
        FN = len([x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] < thr, target, -1)) if x==1]) 
        FP = len([x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] >= thr, target, -1)) if x==0]) 
        TN = len([x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] < thr, target, -1)) if x==0])

        #Sensitivity, recall, TPR
        TPR = TP/(TP+FN)

        #1-specifiity, FPR
        FPR = 1 - (TN/(TN+FP))

        #Precision
        if (TP+FP == 0):
            precision = 1
        else:
            precision = TP/(TP+FP)

        TPR_list_clf.append(TPR)
        FPR_list_clf.append(FPR)
        precision_list_clf.append(precision)
    
    return TPR_list_clf, FPR_list_clf, precision_list_clf


def draw_Friedmann(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose):
    """Calculates Friedman p-value as a non-parametric test to compare more than three groups, and plots the result as boxplot

    Args:
        classifier_data ([pandas.dataframe]): Dataframe containing the metrics for classifiers or feature sets
        dependent_var_name ([type]): [description]
        subject_name ([type]): [description]
        within_name ([type]): [description]
        ax ([object]): ax object from matplotlib that can be graphed on
        xlabel ([string]): x-label of the graph
        ylabel ([string]): y-label of the graph
        verbose ([boolean]): Whether the result of friedman test should be displayed on console.

    Returns:
        [object]: ax object from matplotlib with complete graph on it
    """
    #fig, ax = plt.subplots(figsize=(10,8))
    
    #Create wide table from long-format table
    #This is for Friedmann test
    wide_table = classifier_data.pivot(index=subject_name, columns=within_name, values=dependent_var_name)
    
    #Perform Friedmann test
    friedmann_test = ss.friedmanchisquare(*np.array(wide_table).T)
    
    #Perform post-hoc nemenyi test
    ph = sp.posthoc_nemenyi_friedman(classifier_data, y_col=dependent_var_name, block_col=subject_name, group_col=within_name, melted=True)
    if verbose:
        print(friedmann_test)
        print(ph)

    #Get the data from the post-hoc analysis
    new_stat_data = []
    for coord in itertools.combinations(range(len(ph)), 2):
        new_stat_data.append([max(classifier_data[dependent_var_name]), coord[0], coord[1], ph.iloc[coord[0], coord[1]]])
        
    x_values_of_plot = ph.columns
        
    
    #Plot the data
    plot=sns.boxplot(data=classifier_data, x=within_name, y=dependent_var_name, ax=ax, order=x_values_of_plot)
    sns.swarmplot(x=within_name , y=dependent_var_name, data=classifier_data, ax=ax, order=x_values_of_plot, color="#5d6166", size=5)

    
    min_value = min(classifier_data[dependent_var_name])
    max_value = max(classifier_data[dependent_var_name])
    
    tick_coef = (max_value - min_value)/10


    def get_lines_for_stat(data, ax):
        # max_y, x1, x2, string 
        counter = 0
        for comp in data:
            if list(comp)[3] < 0.05:
                y = comp[0] +counter*tick_coef*2 + tick_coef*2
                x1 = comp[1]
                x2 = comp[2]
                string = round(float(comp[3]),4)
                ax.hlines(y,x1,x2, color="black")
                ax.vlines(x1, y-tick_coef*0.5,y, color="black")
                ax.vlines(x2, y-tick_coef*0.5,y, color="black")
                ax.text((x2+x1)/2-0.2, y+tick_coef/2, string)
                ax.set_ylim(top=y+0.05)
                counter = counter+1
        if counter == 0:
            y = max_value + tick_coef*5
        ax.text(-0.35, y+tick_coef*4, f"Friedman p-value: {friedmann_test.pvalue:.3e}", size=15, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min_value -tick_coef*2, y+tick_coef*6)
        return ax
    
    #Basically we are diabling the get_lines_for_stat by making the threshold super big
    if friedmann_test.pvalue > 5000000:
        get_lines_for_stat(new_stat_data, ax)
    else:
        y = max_value
        pval_formatted = float(f"{friedmann_test.pvalue:.3e}")
        symbol = "NS"
        if 0.01 <= pval_formatted <0.05:
            symbol = "*"
        elif 0.001 <= pval_formatted <= 0.01:
            symbol = "**"
        elif pval_formatted < 0.001:
            symbol = "***"
        ax.text(-0.25, y+tick_coef, f"{pval_formatted} ({symbol})", size=15, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min_value -tick_coef, y+tick_coef*2)
        

def draw_ANOVA_or_Friedmann(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose, sort_alphabetical):
    """Performs ANOVA (parametric) or Friedman (non-parametric) depending on the normality and homogenuity and graph it on ax
       IT ONLY PERFORMS FRIEDMAN AS DECIDED

    Args:
        classifier_data ([pandas.dataframe]): Dataframe containing the metrics for classifiers or feature sets
        dependent_var_name ([string]): The column name containing the values (e.g. "value")
        subject_name ([string]): The column name containing the iterations (e.g. "fold")
        within_name ([string]): The column name containing the name of independent variables (e.g. "clf_type")
        ax ([object]): ax object from matplotlib that can be graphed on
        xlabel ([string]): The xlabel for the graph
        ylabel ([string]): The ylabel for the graph
        verbose ([boolean]): Whether the result of statistical tests should be displayed on console
        sort_alphabetical ([string or list]): If "True", the x-axis tick labels will be sorted alphabetically. Otherwise uses the list to order 
                            the items.
    """
    if sort_alphabetical  == "True":
        classifier_data = classifier_data.sort_values(within_name)
    else:
        classifier_data[within_name] = classifier_data[within_name].astype("category")
        classifier_data[within_name].cat.set_categories(sort_alphabetical, inplace=True)
        classifier_data = classifier_data.sort_values([within_name])
        
    spher = pg.sphericity(classifier_data, dv=dependent_var_name, subject=subject_name, within=[within_name])
    
    homogenuity = pg.homoscedasticity(classifier_data, method="bartlett", alpha=.05,dv=dependent_var_name, group=within_name)

    
    normality = pg.normality(classifier_data, method="shapiro", alpha=.05, dv=dependent_var_name, group=within_name)
    
    if verbose:
        print(spher)
        print("\n")
        print(homogenuity)
        print("\n")
        print(normality)
        print("\n")
        fig, ax1 = plt.subplots(figsize = (10,10))
        pg.qqplot(classifier_data[dependent_var_name], dist='norm', ax=ax1, confidence=False)
    
     #If the assumptions met
    if spher.spher & list(homogenuity["equal_var"])[0] & (False not in list(normality.normal))  &  False:
        print("Assumptions met for ANOVA")
        
        #Perform ANOVA
        draw_ANOVA(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose)
    
    else:
        
        #Perform Friedmann
        print("Assumption for ANOVA not met,", "doing Friedman instead")
        draw_Friedmann(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose)
    

def draw_ANOVA(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose):
    """Calculates ANOVA p-value as a non-parametric test to compare more than three groups, and plots the result as boxplot 

    Args:
        classifier_data ([pandas.dataframe]): Dataframe containing the metrics for classifiers or feature sets
        dependent_var_name ([string]): The column name containing the values (e.g. "value")
        subject_name ([string]): The column name containing the iterations (e.g. "fold")
        within_name ([string]): The column name containing the name of independent variables (e.g. "clf_type")
        ax ([object]): ax object from matplotlib that can be graphed on
        xlabel ([string]): The xlabel for the graph
        ylabel ([string]): The ylabel for the graph
        verbose ([boolean]): Whether the result of statistical tests should be displayed on console

    Returns:
        [object]: ax object from matplotlib with the content completed
    """    
    #Perform ANOVA on the data
    ANOVA = AnovaRM(data=classifier_data, depvar=dependent_var_name, subject=subject_name, within=[within_name]).fit()

    #Print ANOVA
    if verbose:
        print(ANOVA)

    #Perform Tukey
    tukey = pairwise_tukeyhsd(endog=classifier_data[dependent_var_name],
                          groups=classifier_data[within_name],
                          alpha=0.05)

    #Print Tukey
    if verbose:
        print(pd.DataFrame(tukey.summary()))



    #Get the x values of each dataset
    x_values_of_plot = {str(x):float(i) for i, x in enumerate(classifier_data[within_name].unique())}
    
    x_values_of_plot_sorted = dict()
    for clf_name in classifier_data[within_name].unique():
        x_values_of_plot_sorted[str(clf_name)] = x_values_of_plot[str(clf_name)]
        
    x_values_of_plot = x_values_of_plot_sorted


    #Get the tukey data as pandas df
    df = pd.DataFrame(tukey.summary())
    df.columns = df.loc[0,:]
    df = df.loc[1:,:]
    

    df.iloc[:,3] = tukey.pvalues
    #The tukey data as [max_y, x1, x2, string] 
    stat_data_plot = []

    #Populate the stat_data
    for ind in range(len(df)):
        # max_y, x1, x2, string 
        stat_data_plot.append([max(classifier_data[dependent_var_name]),x_values_of_plot[str(df.iloc[ind,0])], x_values_of_plot[str(df.iloc[ind,1])], float(df.iloc[ind,3])])

    coord_dic = {(comp[1],comp[2]):comp for comp in stat_data_plot}  


    unique_xs = []
    for i in [1,2]:
        for comp in coord_dic:
            unique_xs.append(coord_dic[comp][i])
    unique_xs = set(unique_xs)    


    sorted_list = []
    for coord in itertools.combinations(unique_xs, 2):
        poss_coord1 = coord
        poss_coord2 = (coord[1], coord[0])

        if poss_coord1 in coord_dic:
            sorted_list.append(coord_dic[poss_coord1])
        else:
            sorted_list.append(coord_dic[poss_coord2])

    #Plot the data


    plot=sns.boxplot(data=classifier_data, x=within_name, y=dependent_var_name, ax=ax, order= list(x_values_of_plot.keys()))
    sns.swarmplot(x=within_name , y=dependent_var_name, data=classifier_data, ax=ax,order= list(x_values_of_plot.keys()), color="#5d6166", size=5)


    min_value = min(classifier_data[dependent_var_name])
    max_value = max(classifier_data[dependent_var_name])

    tick_coef = (max_value - min_value)/10

    def get_lines_for_stat(data, ax):
        # max_y, x1, x2, string 
        counter = 0
        for comp in data:
            if list(comp)[3] < 0.05:
                y = comp[0] +counter*tick_coef*3 + tick_coef*2
                x1 = comp[1]
                x2 = comp[2]
                string = round(float(comp[3]),4)
                ax.hlines(y,x1,x2, color="black")
                ax.vlines(x1, y-tick_coef*0.2,y, color="black")
                ax.vlines(x2, y-tick_coef*0.2,y, color="black")
                ax.text((x2+x1)/2-0.2, y+tick_coef/2, string)
                ax.set_ylim(top=y+tick_coef*4)
                counter = counter+1
        if counter == 0:
            y = max_value + tick_coef*5
        ax.set_ylim(min_value -tick_coef*2, y+tick_coef*6)
        ax.text(-0.35, y+tick_coef*3, f"ARM p-value: {ANOVA.anova_table.iloc[0,3]:.3e}", size=12, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    if ANOVA.anova_table.iloc[0,3] < 0.05:
        get_lines_for_stat(sorted_list, ax)
    else:
        y = max_value + tick_coef*5
        ax.set_ylim(min_value -tick_coef*2, y+tick_coef*5)
        ax.text(-0.25, y+tick_coef*4, f"ARM p-value: {ANOVA.anova_table.iloc[0,3]:.3e}", size=12, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)



def figure_4_4():
    """Summary of the metrics for 10 different classifiers in distinguishing False Negative and True Negative protein identifications for the 0.5 downsampling ratio dataset.
    """
    #Read the dataset generated from the function train_test_many_clfs()
    performance_data = pd.read_pickle("obj2_1_fold_validation_Aug17.pkl")

    #Note that the experiment was performed in two seperate batches (SVM done seperately) 
    #Therefore, it needs to be added to the dataset
    performance_data_svm = pd.read_pickle("obj2_1_fold_validation_Aug18_SVM.pkl")
    performance_data = pd.concat([performance_data, performance_data_svm])

    #Get rid of Multinomial Naive Bayse
    performance_data.drop(performance_data.index[performance_data['clf_type'] == "Multinomial\nNB"], inplace = True)

    print(performance_data['clf_type'].unique())

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=20

    fig, axs = plt.subplots(3,2, figsize=(20,30),  gridspec_kw={'hspace': 0.15, 'wspace': 0.2})
    fig.set_facecolor('white')

    #The order by which I want the x-axis labels to appear
    sorter = ["Logistic\nRegression","LDA","QDA","Gaussian\nNB",
                "Decision\nTree","Random\nForest","Linear\nSVM", "RBF\nSVM", "Neural\nNetwork", "Dummy\nCLF"]

    for i, metric in enumerate(["test Recall", "test Specificity", 'test Precision','test F1', 'test ROCAUC','test PRAUC' ]):
        
        data_for_plotting = performance_data[(performance_data["metric"] == metric) & (performance_data["sampling_name"] == "Raw")]
        draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="clf_type", 
                                ax=axs.flat[i], xlabel="Classifiers", ylabel=metric, verbose=False, sort_alphabetical=sorter)

        
        axs.flat[i].set_xticklabels(axs.flat[i].get_xticklabels(), rotation=40, size=13)
        axs.flat[i].tick_params(axis='x', which='major', labelsize=13)
        axs.flat[i].set_ylabel(axs.flat[i].get_ylabel(), size=20)
        
        axs.flat[i].set_xlabel("")
        
    plt.savefig("Stat_figure_Obj2_1.pdf", bbox_inches='tight')


def figure_4_5():
    """Summary of the metrics for 10 different classifiers in distinguishing False Positives and True Positives for the 0.5 downsampling ratio.
    """
    #Read the dataset generated from the function train_test_many_clfs
    performance_data = pd.read_pickle("obj2_2_fold_validation_Aug19.pkl")

    #Note that the experiment was performed in two seperate batches (SVM done seperately) 
    #Therefore, it needs to be added to the dataset
    performance_data_svm = pd.read_pickle("obj2_2_fold_validation_Aug19_SVM.pkl")
    performance_data = pd.concat([performance_data, performance_data_svm])

    performance_data.drop(performance_data.index[performance_data['clf_type'] == "Multinomial\nNB"], inplace = True)


    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['ytick.labelsize']=20

    fig, axs = plt.subplots(3,2, figsize=(20,30),  gridspec_kw={'hspace': 0.15, 'wspace': 0.2})
    fig.set_facecolor('white')

    #The order by which I want the x-axis labels to appear
    sorter = ["Logistic\nRegression","LDA","QDA","Gaussian\nNB",
                "Decision\nTree","Random\nForest","Linear\nSVM","RBF\nSVM", "Neural\nNetwork", "Dummy\nCLF"]

    for i, metric in enumerate(["test Recall", "test Specificity", 'test Precision','test F1', 'test ROCAUC','test PRAUC' ]):
        
        data_for_plotting = performance_data[(performance_data["metric"] == metric) & (performance_data["sampling_name"] == "Raw")]
        draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="clf_type", 
                                    ax=axs.flat[i], xlabel="Classifiers", ylabel=metric, verbose=False, sort_alphabetical=sorter)

        
        axs.flat[i].set_xticklabels(axs.flat[i].get_xticklabels(), rotation=40, size=13)
        axs.flat[i].tick_params(axis='x', which='major', labelsize=13)
        axs.flat[i].set_ylabel(axs.flat[i].get_ylabel(), size=20)
        
        axs.flat[i].set_xlabel("")
        
    plt.savefig("Stat_figure_Obj2_2.pdf",bbox_inches='tight')


def figure_4_6():
    """Comparing different data sampling approaches on the False Negative rescue classification task with LDA, random forest, and SVM for the 0.5 downsampling ratio dataset.
    """
    #Read the dataset generated from the function train_test_many_clfs
    performance_data = pd.read_pickle("obj2_1_fold_validation_Aug17.pkl")

    #Note that the experiment was performed in two seperate batches (SVM done seperately) 
    #Therefore, it needs to be added to the dataset
    performance_data_svm = pd.read_pickle("obj2_1_fold_validation_Aug18_SVM.pkl")
    performance_data = pd.concat([performance_data, performance_data_svm])

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.labelsize']=9
    plt.rcParams['ytick.labelsize']=15

    metrics_list = ["test Recall", "test Specificity", 'test ROCAUC','test PRAUC']
    clf_list = ["LDA", "Random\nForest","Linear\nSVM"]

    #The order by which I want the x-axis labels to appear
    sorter = ["Raw", "Over\nSample\n0.1","Over\nSample\n0.5","Over\nSample\n1",
                "Under\nSample\n0.1","Under\nSample\n0.5","Under\nSample\n1",
                "SMOTE\n0.1","SMOTE\n0.5","SMOTE\n1",
                "SMOTE\nENN", "SMOTE\nTomek"]

    fig, axs = plt.subplots(4,3, figsize=(24,31),  gridspec_kw={'hspace': 0.12, 'wspace': 0.11})
    fig.set_facecolor('white')

    counter = 0
    for metric in metrics_list:
        for clf in clf_list:
            data_for_plotting = performance_data[(performance_data["metric"] == metric) & (performance_data["clf_type"] == clf)]
            
            draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="sampling_name", 
                                        ax=axs.flat[counter], xlabel="", ylabel=metric, verbose=False, sort_alphabetical=sorter)
            

        
            axs.flat[counter].set_xticklabels(axs.flat[counter].get_xticklabels(), rotation=30)
            
            if counter in [0,1,2]:
                axs.flat[counter].set_title(clf, size=20)
            
            if counter in [0,3,6,9]:
                axs.flat[counter].set_ylabel(axs.flat[counter].get_ylabel(), size=20)
            else:
                axs.flat[counter].set_ylabel("", size=20)
            
            counter += 1
        
    #     axs.flat[i].set_ylabel(axs.flat[i].get_ylabel(), size=20)
        
    #     axs.flat[i].set_xlabel("")
        
    plt.savefig("Sampling_Obj2_1_Aug21.pdf",bbox_inches='tight')


def figure_4_7():
    """Comparing different data sampling approaches on the False Positive rescue classification task with LDA, random forest, and SVM for the 0.5 downsampling ratio dataset
    """
    #Read the dataset generated from the function train_test_many_clfs
    performance_data = pd.read_pickle("obj2_2_fold_validation_Aug19.pkl")

    
    #Note that the experiment was performed in two seperate batches (SVM done seperately) 
    #Therefore, it needs to be added to the dataset
    performance_data_svm = pd.read_pickle("obj2_2_fold_validation_Aug19_SVM.pkl")
    performance_data = pd.concat([performance_data, performance_data_svm])

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['xtick.labelsize']=9
    plt.rcParams['ytick.labelsize']=15

    metrics_list = ["test Recall", "test Specificity", 'test ROCAUC','test PRAUC']
    clf_list = ["LDA", "Random\nForest","RBF\nSVM"]

    #The order by which I want the x-axis labels to appear
    sorter = ["Raw", "Over\nSample\n0.1","Over\nSample\n0.5","Over\nSample\n1",
                "Under\nSample\n0.1","Under\nSample\n0.5","Under\nSample\n1",
                "SMOTE\n0.1","SMOTE\n0.5","SMOTE\n1",
                "SMOTE\nENN", "SMOTE\nTomek"]

    fig, axs = plt.subplots(4,3, figsize=(24,31),  gridspec_kw={'hspace': 0.12, 'wspace': 0.11})
    fig.set_facecolor('white')

    counter = 0
    for metric in metrics_list:
        for clf in clf_list:
            data_for_plotting = performance_data[(performance_data["metric"] == metric) & (performance_data["clf_type"] == clf)]
            
            draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="sampling_name", 
                                        ax=axs.flat[counter], xlabel="", ylabel=metric, verbose=False, sort_alphabetical=sorter)
            

        
            axs.flat[counter].set_xticklabels(axs.flat[counter].get_xticklabels(), rotation=30)
            
            if counter in [0,1,2]:
                axs.flat[counter].set_title(clf, size=20)
            
            if counter in [0,3,6,9]:
                axs.flat[counter].set_ylabel(axs.flat[counter].get_ylabel(), size=20)
            else:
                axs.flat[counter].set_ylabel("", size=20)
            
            counter += 1
        
    #     axs.flat[i].set_ylabel(axs.flat[i].get_ylabel(), size=20)
        
    #     axs.flat[i].set_xlabel("")
        
    plt.savefig("Sampling_Obj2_2_Aug21.pdf",bbox_inches='tight')



def main():

    figure_4_2_and_4_3()

    #This function needs to be run to generate datasets that will be later used for plotting
    train_test_many_clfs()

    #Generate figures
    figure_4_4()
    figure_4_5()
    figure_4_6()
    figure_4_7()


if __name__=="__main__":
    main()