#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""Performing machine learning experiments as outlined in Chapter 3 of my Master's thesis."""
# =============================================================================
# Imports
from numpy.core.fromnumeric import size
import feature_names
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, FunctionTransformer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score, precision_score, f1_score, recall_score
import numpy as np
LogTransformer = FunctionTransformer(np.log1p)
import pandas as pd
import numpy as np
import itertools as iter
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
#%matplotlib inline
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import random
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
import os
import matplotlib.patches as mpatches
import matplotlib
from Constants import decoy_prefix
plt.rcParams.update({'font.size': 15})



def get_percolator_data(ds_ratio, percolator_output_dir):
    """Returns the Percolator data for a given ds_ratio for the LCM4 experiment (testing dataset)

    Args:
        ds_ratio ([string]): Downsampling ratio (e.g "0.50")
        percolator_output_dir ([string]): The path to the folder containing Percolator's runs

    Returns:
        [pandas.dataframe]: Pandas dataframe containing the Percolator's protein assessment for a given ds_ratio
    """
    all_data = pd.DataFrame()
    counter = 0
    for filename in os.listdir(percolator_output_dir):
        if "proteins" in filename and ds_ratio in filename and "LCM4" in filename:
            a_table = pd.read_table(percolator_output_dir+filename).drop(["peptideIds","ProteinGroupId"], axis=1)
            a_table["pixel"] = filename.rsplit(sep="_No")[1][0:2]
            all_data = pd.concat([all_data, a_table])
            counter = counter+1
    if counter != 48:
        print("There is a problem")
        
    all_data["pixel_accession"] = all_data.apply(lambda x: str(x.pixel) +":"+ x.ProteinId, axis=1)   
    return all_data

def get_FDR_Percolator(percolator_output_dir, downsampled_ratio, target_FDR):
    """Returns FDR information for a given ds_ratio dataset analyzed by Percolator

    Args:
        percolator_output_dir ([string]): The path to the folder containing Percolator's runs
        downsampled_ratio ([string]): Downsampling ratio (e.g "0.50")
        target_FDR ([type]): [description]

    Returns:
        [(dict, dict)]: FDR dictionary contains the calculated Gygi's FDR, # of target matches, and threshold.
                        Protein dict contains the accession number of decoy and protein matches 
    """
    all_data = get_percolator_data(downsampled_ratio, percolator_output_dir)
    
    column_of_interest = "posterior_error_prob"
    threshold_list = sorted(set(all_data[column_of_interest]))

    #trim the threshold list with the closest FDR to target_FDR
    #This step is to acquire the threshold that results in FDR < the target_FDR
    while len(threshold_list) != 1:
        #Choose a random threshold
        rd_thr = random.choice(threshold_list)

        #Calculate FDR at that thr
        #Get the length of decoy matches
        decoy_matches = len(all_data[(all_data[column_of_interest] <= rd_thr) & (all_data["pixel_accession"].str.contains(decoy_prefix))].pixel_accession)
        protein_matches = len(all_data[(all_data[column_of_interest] <= rd_thr) & (~all_data["pixel_accession"].str.contains(decoy_prefix))].pixel_accession)   

        if ((protein_matches)+(decoy_matches)) != 0:
            FDR = (2*(decoy_matches))/((protein_matches)+(decoy_matches))
        else:
            FDR = 0

        if FDR < target_FDR:
            threshold_list = threshold_list[threshold_list.index(rd_thr):len(threshold_list)]
        elif FDR >= target_FDR and (len(threshold_list) != 2):
            threshold_list = threshold_list[0:threshold_list.index(rd_thr)+1]
        elif FDR >= target_FDR and (len(threshold_list) == 2):
            #threshold_list = threshold_list[0:threshold_list.index(rd_thr)]
            threshold_list = [min(threshold_list)]


        #print(rd_thr, FDR, protein_matches, len(threshold_list))
    
    #Dictionary containing the actual protein information
    protein_dic = dict()

    #Dictionary containing the FDR information with decoy and protein matches stored in a dicionary
    FDR_dic = dict()
    for threshold in threshold_list:
        decoy_matches = all_data[(all_data[column_of_interest] <= threshold) & (all_data["pixel_accession"].str.contains(decoy_prefix))].pixel_accession
        protein_matches = all_data[(all_data[column_of_interest] <= threshold) & (~all_data["pixel_accession"].str.contains(decoy_prefix))].pixel_accession  

        Target_Match = len(protein_matches)
        if (len(protein_matches)+len(decoy_matches)) != 0:
            FDR = (2*len(decoy_matches))/(len(protein_matches)+len(decoy_matches))
        else:
            FDR = 0
        FDR_dic[threshold] = (FDR, Target_Match)

        protein_dic[threshold] = [decoy_matches, protein_matches]
        print("Percolator in get_FDR_classifier", "thr", threshold, "FDR", FDR, "length of protein matches raw and set", len(protein_matches), len(set(protein_matches)))
        #print(Target_Match, len(decoy_matches))

    return FDR_dic, protein_dic

def percolator_calc_FDR_plot(percolator_output_dir, downsampled_ratio, number_of_points):
    """This function generates the FDR vs. number_of_protein_matches for a given DS_ratio for the percolator output

    Args:
        percolator_output_dir ([string]): The path to the folder containing Percolator's runs
        downsampled_ratio ([string]): Downsampling ratio (e.g "0.50")
        number_of_points ([int]): Number of thresholds that the FDR needs to be evaluated

    Returns:
        [(list, list)]: List of FDRs (x-axis), and list of numbr of protein matches (y-axis) at that FDR
    """
    all_data = get_percolator_data(downsampled_ratio, percolator_output_dir)
    column_of_interest="posterior_error_prob"
    
    FDR_list=[]
    protein_mathced_list = []
    #Create a list of thresholds with equal spaces apart, Also, making sure we have the highest threshold in
    thresholds = sorted(set(all_data[column_of_interest]))
    thresholds_filtered = thresholds[::int(len(thresholds)/(number_of_points -2))]
    if thresholds[len(thresholds)-1] not in thresholds_filtered:
        thresholds_filtered.append(thresholds[len(thresholds)-1])

    for thr in thresholds_filtered:
        decoy_matches = all_data[(all_data[column_of_interest] <= thr) & (all_data["pixel_accession"].str.contains(decoy_prefix))].pixel_accession
        protein_matches = all_data[(all_data[column_of_interest] <= thr) & (~all_data["pixel_accession"].str.contains(decoy_prefix))].pixel_accession
        Target_Match = len(protein_matches)
        if (len(protein_matches)+len(decoy_matches)) != 0:
            FDR = (2*len(decoy_matches))/(len(protein_matches)+len(decoy_matches))
        else:
            FDR = 0
        FDR_list.append(FDR)
        protein_mathced_list.append(Target_Match)
        #print(thr, len(decoy_matches), len(protein_matches))
        #thr_list.append(thr)
        
    return FDR_list, protein_mathced_list

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
                             columns=["Charge_2_POI","Charge_3_POI","Charge_4_POI","Charge_5_POI","Charge_6_POI"],
                            index=X.index)
      
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
    """Keep only the necessary features (columns) in the dataset (i.e. which feature sets to uses)

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

def validation_X_fold(training_filename, testing_filename, X_fold, decoy_string):
    """Perform X-fold training-testing scheme explained in the thesis (Chp3) with three different feature sets 
       and 5 machine learning models

    Args:
        training_filename ([string]): Complete path and name of the csv training dataset outputted from 3_CreateDataset.py module
        testing_filename ([type]): Complete path and name of the csv testing dataset outputted from 3_CreateDataset.py module
        X_fold ([int]): Number of folds for the training and testing scheme
        decoy_string ([string]): Decoy string defines decoy sequences in the database (decoy_prefix)

    Returns:
        [(pandas.dataframe, dict)]: The dataframe contains the performance info of the ML models. The dictionary contains the 
                                    ROC and PR curve information for plotting.
    """
    
    #We record the performance data in a dataframe
    performance_data = pd.DataFrame(columns=["value", "clf_type", "feature_type","metric", "fold"])

    #The TPR, FPR, and Precision goes to a dic for each fold and each set of training and testing
    ROC_PR_data = dict()

    #Create stratified classfication object 
    skf = StratifiedKFold(n_splits=X_fold)
    
    #Defining feature types
    for feature_type in ["All_features", "POI_features_only", "Spatial_features_only"]:
        
        #Define the training and testing datasets, then preprocess them
        X_train, y_train = pre_process_features(training_filename, spatial=feature_type, testing=False, target_column="target", preprocessing="MinMaxScale")
        X_test, y_test = pre_process_features(testing_filename, spatial=feature_type, testing=True, target_column="target", preprocessing="MinMaxScale")
    
        #defining folds from training set for evaluation
        for fold, train_indexes, test_indexes in zip(range(skf.n_splits), skf.split(X_train, y_train), skf.split(X_test, y_test)):
            train_index = train_indexes[1]
            test_index = test_indexes[1]

            X_train_fold = X_train.iloc[train_index].copy()
            y_train_fold = y_train.iloc[train_index].copy()

            X_test_fold = X_test.iloc[test_index].copy()
            y_test_fold = y_test.iloc[test_index].copy()
            
            #defining classifiers to train test
            for clf_type in ["LDA","Gaussian\nNB", "Decision\nTree","Logistic\nRegression", "Dummy\nCLF"]:

                #Train the classifier with grid search
                trained_classifier = get_best_classifier(clf_type, X=X_train_fold, y=y_train_fold, scoring="f1")
                
                #Get FDR information of the classifier
                FDR_dic_trained_classifier, _ = get_FDR_classifier(trained_classifier, X_test_fold, target_FDR=0.01, decoy_string=decoy_string)
                FDR_dic_trained_classifier = list(FDR_dic_trained_classifier.values())[0]
                print("FDR calculation", clf_type, FDR_dic_trained_classifier[0], FDR_dic_trained_classifier[1])
                
                #use the trained classifier to predict instances in the testing set
                y_pred = trained_classifier.predict(X_test_fold.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))

                #Store the performance of the classifier in a dic which will then populate a dataframe
                performance_dic = dict()
                performance_dic["test PRAUC"] = average_precision_score(y_test_fold, y_pred)
                performance_dic["test ROCAUC"] = roc_auc_score(y_test_fold, y_pred)
                performance_dic["test Precision"] = precision_score(y_test_fold, y_pred)
                performance_dic["test Recall"] = recall_score(y_test_fold, y_pred)
                performance_dic["test F1"] = f1_score(y_test_fold, y_pred)
                performance_dic["test MCC"] = matthews_corrcoef(y_test_fold, y_pred)
                performance_dic["test TargetMatch"] = FDR_dic_trained_classifier[1]

                #Confusion matrix is used to calc specificity
                tn, fp, _, _ = confusion_matrix(y_test_fold, y_pred).ravel()
                specificity = tn / (tn+fp)
                performance_dic["test Specificity"] = specificity

                #We use the trained model and the testing set to draw ROC and PR curves
                TPR_list_clf, FPR_list_clf, precision_list_clf = get_ROC_PR_curve(trained_classifier, X_test_fold, y_test_fold, 100)

                #Store the ROC and PR info in a dic
                ROC_PR_data[f"{clf_type}_{feature_type}_{fold+1}_TPR"] = TPR_list_clf
                ROC_PR_data[f"{clf_type}_{feature_type}_{fold+1}_FPR"] = FPR_list_clf
                ROC_PR_data[f"{clf_type}_{feature_type}_{fold+1}_Precision"] = precision_list_clf

                #Populate the pandas dataframe with the dic
                for key in performance_dic:
                    plot_data = pd.Series([performance_dic[key], clf_type, feature_type, key, fold+1], index=["value", "clf_type", "feature_type","metric", "fold"])
                    performance_data = performance_data.append(plot_data, ignore_index=True)
                    
                print(clf_type, feature_type, fold+1, "is DONE!")

                
    return performance_data, ROC_PR_data

def get_protein_hits_at_FDR(training_filename, testing_filename, target_FDR, decoy_string, clf_type, spatial, over_under_sample, grid_search_scoring, get_PP=True):
    """Function that trains a classifier, and then it outputs the FDR results for ProteinProphet and SAPID

    Args:
        training_filename ([string]): Complete path and name of the training file name
        testing_filename ([string]): Complete path and name of the testing file name
        target_FDR ([float]): The target FDR (e.g. 0.01)
        decoy_string ([string]): The decoy string used in the databse search
        clf_type ([string]): The type of classifier: "Logistic\nRegression", "LDA", "Gaussian\nNB", "Decision\nTree", "Dummy\nCLF"
        spatial ([string]): The choices are "All_features","POI_features_only","Spatial_features_only"
        over_under_sample ([string]): Different data sampling: "None","SMOTE_Over","Random_Over","Random_Under", "SMOTEENN"
        grid_search_scoring ([string]): The scoring function by which the best hyperparameter combinations are chosen (e.g. "f1", "accuracy". etc)
        get_PP (bool, optional): Whether to get the FDR info for ProteinProphet. Defaults to True.

    Returns:
        [(dict,dict,dict,dict)]: FDR_classifier_info, protein_classifier_info, FDR_PP_dic, protein_PP_dic
    """
    
    print("Training", clf_type)

    #Train a classifier using the training set
    trained_classifier = train_a_classifier(training_file=training_filename, clf_type=clf_type, 
                                            spatial=spatial, over_under_sample=over_under_sample, scoring=grid_search_scoring, 
                                            target_column="target", preprocessing="MinMaxScale")
    
    
    #Preprocess the testing set
    testing_set, target = pre_process_features(testing_filename, spatial=spatial, testing=True, target_column="target", preprocessing="MinMaxScale")
    print(f"shape of testing set is {testing_set.shape}")

    #Get the FDR information of ProteinProphet
    if get_PP:
        FDR_PP_dic, protein_PP_dic = get_FDR_PP(testing_set, target_FDR=target_FDR, decoy_string=decoy_string)

    #Get FDR information of the trained classifier (SAPID)
    FDR_classifier_info, protein_classifier_info = get_FDR_classifier(trained_classifier, testing_set, target_FDR=target_FDR, decoy_string=decoy_string)
    
    if get_PP:
        return FDR_classifier_info, protein_classifier_info, FDR_PP_dic, protein_PP_dic
    
    return FDR_classifier_info, protein_classifier_info

def analyse_file_plotting(training_filename, testing_filename, num_threshold, decoy_string, clf_type, spatial, over_under_sample, grid_search_scoring):
    """Train a classifier, and then use the testing set to calculate FPR, TPR, Precision, and FDR vs Number of protein matches information
       For both SAPID and ProteinProphet

    Args:
        training_filename ([string]): Complete path and name of the training file name
        testing_filename ([string]): Complete path and name of the testing file name
        num_threshold ([int]): Number of thresholds from 0 to 1 which we find to calculated FDR at
        decoy_string ([string]): The decoy string used in the databse search
        clf_type ([string]): The type of classifier: "Logistic\nRegression", "LDA", "Gaussian\nNB", "Decision\nTree", "Dummy\nCLF"
        spatial ([string]): The choices are "All_features","POI_features_only","Spatial_features_only"
        over_under_sample ([string]): Different data sampling: "None","SMOTE_Over","Random_Over","Random_Under", "SMOTEENN"
        grid_search_scoring ([string]): The scoring function by which the best hyperparameter combinations are chosen (e.g. "f1", "accuracy". etc)

    Returns:
        [dict]: Dictionary with information on TPR, FPR, Precision, classifier_FDR_dic, PP_FDR_dic
    """

    print("Training", clf_type)

    #Train a classifier on the training dataset
    trained_classifier = train_a_classifier(training_file=training_filename, clf_type=clf_type, 
                                            spatial=spatial, over_under_sample=over_under_sample, scoring=grid_search_scoring, 
                                            target_column="target", preprocessing="MinMaxScale")
    
    #Pre-process the testing set
    testing_set, target = pre_process_features(testing_filename, spatial=spatial, testing=True, target_column="target", preprocessing="MinMaxScale")
    print(f"shape of testing set is {testing_set.shape}")


    #Get the ROC and PR curve info for the classifier and the test test
    TPR_list_clf, FPR_list_clf, precision_list_clf = get_ROC_PR_curve(trained_classifier, testing_set, target, 100)
    print("ROC PR Done")
    
    
    #Get the ProteinProphet and classifier FDR info 
    PP_info = get_FDR_PP_For_plotting(testing_set, num_threshold=num_threshold, decoy_string=decoy_string)
    classifier_info = get_FDR_classifier_For_plotting(trained_classifier, testing_set, num_threshold=num_threshold, decoy_string=decoy_string)
        
    return {"TPR":TPR_list_clf, "FPR":FPR_list_clf, "Precision":precision_list_clf, "classifier_FDR_dic":classifier_info, "PP_FDR_dic":PP_info}
    
def train_a_classifier(training_file, clf_type, spatial, over_under_sample, scoring, target_column, preprocessing):
    """Train a classifier with hyperparameter tuning using grid search

    Args:
        training_file ([string]): Complete path and name to the training file CSV file
        clf_type ([string]): The type of classifier: "Logistic\nRegression", "LDA", "Gaussian\nNB", "Decision\nTree", "Dummy\nCLF"
        spatial ([string]): The choices are "All_features","POI_features_only","Spatial_features_only"
        over_under_sample ([string]): Type of datasampling (if any) that needs to be performed
        scoring ([string]): The scoring function by whihc the best hyperparameter combinations are chosen (e.g. "f1", "accuracy". etc)
        target_column ([string]): Name of the column containing the labels for the training dataset (e.g. "target")
        preprocessing ([string]): Type of preprocessing: "MinMaxScale","StandardScale","QuantileTransform","LogTransform", or nothing

    Returns:
        [type]: [description]
    """

    #Read the training data, and preprocess it
    X, y = pre_process_features(training_file, spatial=spatial,
                            testing=False, target_column=target_column, preprocessing=preprocessing)
    
    print(f"shape of training set is {X.shape}")

    
    if over_under_sample != "None":
        print("Values before ",over_under_sample, " Sampling" ,"\n",y.value_counts())

    #Data sampling
    X, y = over_under_sample_data(over_under_sample, X, y)
    
    print(f"training dataset has {len(X.columns)} columns")
    
    #Oversample
    if over_under_sample != "None":
        print("Values after ",over_under_sample, " Sampling" ,"\n", pd.Series(y).value_counts())


    return get_best_classifier(clf_type, X, y, scoring)

def over_under_sample_data(over_under_sample, X, y):
    """Perform data sampling on the datasample

    Args:
        over_under_sample ([string]): "None","SMOTE_Over","Random_Over","Random_Under", "SMOTEENN"
        X ([pandas.dataframe]): Pandas dataframe which contains the examples (X)
        y ([pandas.series]): Pandas series containing the labels of the examples (y)

    Returns:
        [(pandas.dataframe, pandas.series)]: Modified X and y
    """
    if over_under_sample == "None":
        return X, y
    elif over_under_sample == "SMOTE_Over":
        ros = SMOTE(random_state=1)
        return ros.fit_resample(X, y)
    elif over_under_sample == "Random_Over":
        ros = RandomOverSampler(random_state=1)
        return ros.fit_resample(X, y)
    elif over_under_sample == "Random_Under":
        ros = RandomUnderSampler(random_state=1)
        return ros.fit_resample(X, y)
    elif over_under_sample == "SMOTEENN":
        ros = SMOTEENN(random_state=1)
        return ros.fit_resample(X, y)

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
        skf = StratifiedKFold(n_splits=5)

        #Perform Grid search
        grid_values = [{'penalty': ["l2", "elasticnet"],'C':[0.01, 0.1, 1.0, 10], 'l1_ratio': [0.25, 0.5, 0.75],
                        'class_weight':[None, "balanced"]}]
    
    elif clf_type == "LDA":
        
        #LDA
        clf = LinearDiscriminantAnalysis(solver= "svd")

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=5)

        #Perform Grid search
        grid_values = [{'tol': [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-5]}]
        
    
    elif clf_type == "KNeighborsClassifier":
        
        clf = KNeighborsClassifier(n_jobs=-1)
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=10, shuffle=False)
        
        #Perform Grid search
        grid_values = [{'n_neighbors': [2, 5 ,20],'weights':["uniform", "distance"],
                        'leaf_size':[30]}]
        
    elif clf_type == "Gaussian\nNB":
        
        #Guassian naive base
        clf = GaussianNB()
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=5)

        #Perform Grid search
        grid_values = [{'var_smoothing': [1e-8, 1e-9, 1e-10]}]
        
    elif clf_type == "Multinomial\nNB":
        
        #Multinomial naive base
        clf = MultinomialNB()
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=10, shuffle=False)

        #Perform Grid search
        grid_values = [{'alpha': [0, 0.5, 1.0, 1.5, 2.0],
                       'fit_prior': [True, False]}]

    elif clf_type == "SVC":

        clf = SVC(probability=True, random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

        #Perform Grid search
        grid_values = [{'kernel': ['linear'], 'gamma': ["auto"],
                     'C': [0.01, 0.1, 1.0, 10],'class_weight':[None, "balanced"]}
                    ]


    elif clf_type == "Decision\nTree":

        clf = DecisionTreeClassifier(random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=5)

        #Perform Grid search
        grid_values = {'criterion':['gini','entropy'],
                       'max_depth':[1,2,3,4,5, None],
                       'min_samples_split': [2,4,6,8],
                       'min_samples_leaf' : [1,2,3,5],
                       'class_weight':[None, "balanced"]}

    elif clf_type == "Random\nForest":

        clf = RandomForestClassifier(random_state=1)

        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        n_estimators = [100, 300, 500]
        max_depth = [5,  10]
        min_samples_split = [2, 5, 10, 15, 50]
        min_samples_leaf = [1, 2, 5, 10] 

        grid_values = dict(n_estimators = n_estimators, max_depth = max_depth,  
            #   min_samples_split = min_samples_split, 
            #  min_samples_leaf = min_samples_leaf,
             class_weight = ["None", "balanced"])
    
    elif clf_type == "Dummy\nCLF":
        
        clf = DummyClassifier()
        
        #Number of splits and cross validation
        skf = StratifiedKFold(n_splits=5)
        
        grid_values = {"strategy": ["stratified"]}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring = scoring, refit = True, cv=skf, verbose=1,return_train_score=True, n_jobs=-1)
    grid_clf.fit(X, y)

    # Best parameters obtained with GridSearchCV:
    print(clf_type," Best parameters are: ",grid_clf.best_params_)
    print(clf_type," Best score is: ", grid_clf.best_score_)

    return grid_clf.best_estimator_

def get_FDR_PP(dataset, target_FDR, decoy_string=decoy_prefix):
    """Find the ProteinProphet prob threshold that results in target FDR value (Gygi's formula), then returns the protein and decoy matches accessions.

    Args:
        dataset ([pandas.dataframe]): Dataframe with the ProteinProphet probabilities and protein accession values in the column.
        target_FDR ([float]): The target FDR (e.g. 0.01)
        decoy_string (str, optional): The decoy string thta was used in the database search. Defaults to decoy_prefix.

    Returns:
        [(dict, dict)]: FDR dictionary with the threshold and number of protein matches and calculated Gygi's FDR. 
                        The protein dictionary containing the list of decoy_matches, protein_matches, protein_not_matches as lists.
    """
    
    #Threshold List 
    #threshold_list=list(np.linspace(0,1,num_threshold))
    threshold_list=sorted(set((list(dataset["protein_proph_prob"]))),reverse=False)
    
    #trim the thr list with the closest FDR to target_FDR
    while len(threshold_list) != 1:
        #Choose a random threshold
        rd_thr = random.choice(threshold_list)

        #Calculate FDR at that thr
        #Get the length of decoy matches
        decoy_matches= len(list(dataset[(dataset["protein_proph_prob"] >= rd_thr) & (dataset["accession"].str.contains(decoy_string))].pixel_instance_name))
        #Get the length of protein matches
        protein_matches= len(list(dataset[(dataset["protein_proph_prob"] >= rd_thr) & (~dataset["accession"].str.contains(decoy_string))].pixel_instance_name))
        if ((protein_matches)+(decoy_matches)) != 0:
            FDR = (2*(decoy_matches))/((protein_matches)+(decoy_matches))
        else:
            FDR = 0

        if FDR > target_FDR:
            threshold_list = threshold_list[threshold_list.index(rd_thr)+1:len(threshold_list)]
        elif FDR <= target_FDR:
            threshold_list = threshold_list[0:threshold_list.index(rd_thr)+1]

    #Dictionary containing the actual protein information
    protein_dic = dict()
    
    #Dictionary containing the FDR information
    FDR_dic = dict()

    #Get the list of decoy matches, protein matches, and protein unmatches at each threshold and keep them in a dictionary
    for threshold in threshold_list:

        #Get the decoy matches
        decoy_matches= list(dataset[(dataset["protein_proph_prob"] >= threshold) & (dataset["accession"].str.contains(decoy_string))].pixel_instance_name)

        #Get the protein matches
        protein_matches= list(dataset[(dataset["protein_proph_prob"] >= threshold) & (~dataset["accession"].str.contains(decoy_string))].pixel_instance_name)

        #Get the protein not_matches
        protein_not_matches = list(dataset[(dataset["protein_proph_prob"] < threshold) & (~dataset["accession"].str.contains(decoy_string))].pixel_instance_name)

        Target_Match = len(protein_matches)
        if (len(protein_matches)+len(decoy_matches)) != 0:
            FDR = (2*len(decoy_matches))/(len(protein_matches)+len(decoy_matches))
        else:
            FDR = 0
        FDR_dic[threshold] = (FDR, Target_Match)

        protein_dic[threshold] = [decoy_matches, protein_matches, protein_not_matches]
        print("PP in get_FDR_classifier", "thr", threshold, "length of protein matches raw and set", len(protein_matches), len(set(protein_matches)))

        
    return FDR_dic, protein_dic

def fit_polynomial(x, y):
    """Extrapolates 1% FDR by fitting a third, fourth, and fifth degree polynomial and then averaging the value at 1% FDR

    Args:
        x ([list]): Calculated FDR values
        y ([list]): Number of protein matches correponding to the calcualted FDR values

    Returns:
        [float]: Extrapolated number of protein matches at 1% FDR
    """

    #fig, ax = plt.subplots(figsize=(10,6))
    xp = np.linspace(0, 1, 100)
    degrees = [3,4,5]
    #plt.plot(x, y, '.',label="Actual Points", markersize=20)
    values_at_1percent_fdr = []
    for degree in degrees:
        z = np.polyfit(x, y, degree)
        p = np.poly1d(z)
        values_at_1percent_fdr.append(p(0.01))
        #plt.plot(xp, p(xp), '-', label = degree)
    # plt.scatter(0.01, np.mean(values_at_1percent_fdr), s=100, label="predicted value", color="red")
    # plt.legend()
    # plt.show()

    return np.mean(values_at_1percent_fdr)
    
def get_FDR_classifier(model, dataset, target_FDR, decoy_string=decoy_prefix):
    """Calculates the classifier's threshold that results in target FDR value (Gygi's formula), then returns the protein and decoy matches accessions.

    Args:
        model ([Object]): Sklearn's trained classifier object (which must have the predict_proba() function)
        dataset ([pandas.dataframe]): The dataset that the trained model needs to be tested on.It must contain the columns ("protein_proph_prob",'accession','pixel_instance_name')
        target_FDR ([float]): The target FDR (e.g. 0.01)
        decoy_string (str, optional): The decoy string used in database search. Defaults to decoy_prefix.

    Returns:
        [(dict, dict)]: FDR dictionary with the threshold and number of protein matches and calculated Gygi's FDR. 
                        The protein dictionary containing the list of decoy_matches, protein_matches, protein_not_matches as lists.
    """
    
    #Threshold List 
    threshold_list=sorted(set((list(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1]))),reverse=False)
    
    #trim the thr list with the closest FDR to target_FDR
    while len(threshold_list) > 1:
        #Choose a random threshold
        rd_thr = random.choice(threshold_list)

        #Calculate FDR at that thr
        decoy_matches = len([x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession','pixel_instance_name'],axis=1))[:,1] >= rd_thr, dataset.pixel_instance_name, 0)) if x!=0 and decoy_string in x]) 

        protein_matches = len([x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession','pixel_instance_name'],axis=1))[:,1] >= rd_thr, dataset.pixel_instance_name, 0)) if x!=0 and decoy_string not in x]) 

        if ((protein_matches)+(decoy_matches)) != 0:
            FDR = (2*(decoy_matches))/((protein_matches)+(decoy_matches))
        else:
            FDR = 0

        if FDR > target_FDR:
            threshold_list = threshold_list[threshold_list.index(rd_thr)+1:len(threshold_list)]
        elif FDR <= target_FDR:
            threshold_list = threshold_list[0:threshold_list.index(rd_thr)+1]
    

        
        
    #Dictionary containing the actual protein information
    protein_dic = dict()

    #Dictionary containing the FDR information
    FDR_dic = dict()
    for threshold in threshold_list:
        decoy_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] >= threshold, dataset.pixel_instance_name, 0)) if x!=0 and decoy_string in x] 

        protein_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] >= threshold, dataset.pixel_instance_name, 0)) if x!=0 and decoy_string not in x] 

        protein_not_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', "pixel_instance_name"],axis=1))[:,1] < threshold, dataset.pixel_instance_name, 0)) if x!=0 and decoy_string not in x]

        Target_Match = len(protein_matches)
        if (len(protein_matches)+len(decoy_matches)) != 0:
            FDR = (2*len(decoy_matches))/(len(protein_matches)+len(decoy_matches))
        else:
            FDR = 0
        FDR_dic[threshold] = (FDR, Target_Match)
        
        protein_dic[threshold] = [decoy_matches, protein_matches, protein_not_matches]
        print("SAPID in get_FDR_classifier", "thr", threshold, "length of protein matches raw and set", len(protein_matches), len(set(protein_matches)))
        #If FDR > target_FDR lets extrapolate the value of FDR using polynomial fitting
        
    if FDR > target_FDR:
        FDR_Data = get_FDR_classifier_For_plotting(model, dataset, num_threshold=10, decoy_string=decoy_prefix)
        FDR_values = []
        protein_hits = []

        for thr in FDR_Data:
            FDR_values.append(FDR_Data[thr][0])
            protein_hits.append(FDR_Data[thr][1])

        estimated_value = fit_polynomial(FDR_values, protein_hits)
        
        return {"0.00_estimated": (0.01, estimated_value)}, {}
    else:
        return FDR_dic, protein_dic

def get_FDR_PP_For_plotting(dataset, num_threshold, decoy_string=decoy_prefix):
    """Calcualte the FDR values (Gygi's formula) and number of protein matches at different thresholds for ProteinProphet

    Args:
        dataset ([pandas.dataframe]): Dataframe with the ProteinProphet probabilities.
        num_threshold ([int]): Number of thresholds from 0 to 1 which we find to calculated FDR at.
        decoy_string (str, optional): The decoy string used in the databse search. Defaults to decoy_prefix.

    Returns:
        [dict]: The dicionary containing the thresholds and number of protein matches at different FDR.
    """
    
    #Threshold List 
    threshold_list=set(dataset["protein_proph_prob"].copy())
    
    #Make sure the number of thresholds is less than the number of probabilities
    if num_threshold >= len(threshold_list):
        num_threshold = len(threshold_list) - 1
    
    thresholds = sorted(threshold_list)
    thresholds_filtered = thresholds[::int(len(thresholds)/(num_threshold -2))]
    if thresholds[len(thresholds)-1] not in thresholds_filtered:
        thresholds_filtered.append(thresholds[len(thresholds)-1])
    
    #Dictionary containing the FDR information
    FDR_dic = dict()

    #Get the list of decoy matches, protein matches, and protein unmatches at each threshold and keep them in a dictionary
    for threshold in thresholds_filtered:

        #Get the decoy matches
        decoy_matches= list(dataset[(dataset["protein_proph_prob"] >= threshold) & (dataset["accession"].str.contains(decoy_string))].accession)

        #Get the protein matches
        protein_matches= list(dataset[(dataset["protein_proph_prob"] >= threshold) & (~dataset["accession"].str.contains(decoy_string))].accession)

        #Get the protein not_matches
        #protein_not_matches = list(dataset[(dataset["protein_proph_prob"] < threshold) & (~dataset["accession"].str.contains(decoy_string))].accession)

        Target_Match = len(protein_matches)
        if (len(protein_matches)+len(decoy_matches)) != 0:
            FDR = (2*len(decoy_matches))/(len(protein_matches)+len(decoy_matches))
        else:
            FDR = 0
        FDR_dic[threshold] = (FDR, Target_Match)
        #print( "Threshold", threshold, "FDR", FDR, "Target:", Target_Match, decoy_matches)
        
    return FDR_dic

def get_FDR_classifier_For_plotting(model, dataset, num_threshold=10, decoy_string=decoy_prefix):
    """Calcualte the FDR values (Gygi's formula) and number of protein matches at different thresholds for a classifier

    Args:
        model ([object]): A sklearn trained classifier (which must have the predict_proba() function)
        dataset ([pandas.dataframe]): The dataset that the trained model needs to be tested on. It must contain the columns "protein_proph_prob",'accession','pixel_instance_name'
        num_threshold (int, optional): Number of thresholds from 0 to 1 which we find to calculated FDR at. Defaults to 10.
        decoy_string (str, optional):  The decoy string used in the databse search. Defaults to decoy_prefix.

    Returns:
        [dict]: The dicionary containing the thresholds and number of protein matches at different FDR.
    """
    
    #Threshold List 
    threshold_list=set(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1])
    
    #Make sure the number of thresholds is less than the number of probabilities
    if num_threshold >= len(threshold_list):
        num_threshold = len(threshold_list) - 1
        
    thresholds = sorted(threshold_list)
    thresholds_filtered = thresholds[::int(len(thresholds)/(num_threshold -2))]
    if thresholds[len(thresholds)-1] not in thresholds_filtered:
        thresholds_filtered.append(thresholds[len(thresholds)-1])
    
    #Dictionary containing the actual protein information
    protein_dic = dict()

    #Dictionary containing the FDR information
    FDR_dic = dict()
    for threshold in thresholds_filtered:
        decoy_not_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] < threshold, dataset.accession, 0)) if x!=0 and decoy_string in x] 

        decoy_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] >= threshold, dataset.accession, 0)) if x!=0 and decoy_string in x] 

        protein_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] >= threshold, dataset.accession, 0)) if x!=0 and decoy_string not in x] 

        protein_not_matches = [x for x in list(np.where(model.predict_proba(dataset.drop(["protein_proph_prob",'accession', 'pixel_instance_name'],axis=1))[:,1] < threshold, dataset.accession, 0)) if x!=0 and decoy_string not in x]

        Target_Match = len(protein_matches)
        if (len(protein_matches)+len(decoy_matches)) != 0:
            FDR = (2*len(decoy_matches))/(len(protein_matches)+len(decoy_matches))
        else:
            FDR = 0
        FDR_dic[threshold] = (FDR, Target_Match)
        #print( "Threshold", threshold, "FDR", FDR, "Target:", Target_Match, len(decoy_matches))

        protein_dic[threshold] = [decoy_matches, protein_matches, protein_not_matches]

    return FDR_dic

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
    
    #A very large p-value forces the function to not bother plotting the post-hoc Nemenyi test
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
           
def draw_ANOVA_or_Friedmann(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose):
    """Performs ANOVA (parametric) or Friedman (non-parametric) depending on the normality and homogenuity and graph it on ax
       ***IT ONLY PERFORMS FRIEDMAN AS DECIDED (remove "False" from second if statement if you want the intended behaviour)

    Args:
        classifier_data ([pandas.dataframe]): Dataframe containing the metrics for classifiers or feature sets
        dependent_var_name ([string]): The column name containing the values (e.g. "value")
        subject_name ([string]): The column name containing the iterations (e.g. "fold")
        within_name ([string]): The column name containing the name of independent variables (e.g. "clf_type")
        ax ([object]): ax object from matplotlib that can be graphed on
        xlabel ([string]): The xlabel for the graph
        ylabel ([string]): The ylabel for the graph
        verbose ([boolean]): Whether the result of statistical tests should be displayed on console
    """

    classifier_data = classifier_data.sort_values(within_name)
        
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
    #** Remove the "& False" at the end to make the function perform ANOVA
    if spher.spher & list(homogenuity["equal_var"])[0] & (False not in list(normality.normal))  &  False:
        print("Assumptions met for ANOVA")
        
        #Perform ANOVA
        draw_ANOVA(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose)
    
    else:
        
        #Perform Friedmann
        print("Assumption for ANOVA not met,", "doing Friedman instead")
        draw_Friedmann(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose)
    
def t_test_or_wilcoxon(classifier_data, dependent_var_name, within_name, ax, xlabel, ylabel, verbose):
    """Performs t-test (parametric) or wilcoxon (non-parametric) depending on the normality and homogenuity and graph it on ax
       **IT ONLY PERFORMS WILCOXON AS DECIDED (remove the "False" in second if statement to make it work as intended)

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
        [object]: ax object from matplotlib with complete graph on it
    """
    clf_names = list(set(classifier_data[within_name]))
    data1 = list(classifier_data[classifier_data[within_name] == clf_names[0]][dependent_var_name])
    data2 = list(classifier_data[classifier_data[within_name] == clf_names[1]][dependent_var_name])
        
    homogenuity = pg.homoscedasticity(classifier_data, method="bartlett", alpha=.05,dv=dependent_var_name, group=within_name)

    
    normality = pg.normality(classifier_data, method="shapiro", alpha=.05, dv=dependent_var_name, group=within_name)
    
    if verbose:
        print(homogenuity)
        print("\n")
        print(normality)
        print("\n")
        fig, ax1 = plt.subplots(figsize = (10,10))
        pg.qqplot(classifier_data[dependent_var_name], dist='norm', ax=ax1, confidence=False)
    
     #If the assumptions are met
    if list(homogenuity["equal_var"])[0] & (False not in list(normality.normal) & (False)):
        print("Assumptions meet", "Doing t-test")
        
        
        
        ttest_res = ttest(data1, data2, paired=True, correction='auto')
        cohed_d = float(ttest_res["cohen-d"])
        pvalue = float(ttest_res["p-val"])
        if pvalue < 0.05:
            test_type = "t-test"
            text = f"* {pvalue:.3}\nCohen-d: {cohed_d:.3}"
        else:
            test_type = "t-test"
            text = f"ns {pvalue:.3}"
            
    else:
        
        #Perform Wilcoxon
        print("Assumptions not meet", "Doing Wilcoxon test")
        
        wilcoxon_res = pg.wilcoxon(data1, data2, tail='two-sided')
        pvalue = float(wilcoxon_res["p-val"])
        RBC = float(wilcoxon_res["RBC"])
        
        if pvalue < 0.05:
            test_type = "Wilcoxon test"
            text = f"* {pvalue:.3}\nRBC: {RBC:.3}"
        else:
            test_type = "Wilcoxon test"
            text = f"ns {pvalue:.3}"
    

  
    #Plot the data
    plot=sns.boxplot(data=classifier_data, x=within_name, y=dependent_var_name, ax=ax, order= clf_names)
    sns.swarmplot(x=within_name , y=dependent_var_name, data=classifier_data, ax=ax, order= clf_names, color="#5d6166", size=5)


    min_value = min(classifier_data[dependent_var_name])
    max_value = max(classifier_data[dependent_var_name])
    
    sorted_list = [[max_value, 0, 1, text]]

    tick_coef = (max_value - min_value)/10

    def get_lines_for_stat(data, ax):
        # max_y, x1, x2, string 
        counter = 0
        for comp in data:
            y = comp[0] +counter*tick_coef*3 + tick_coef*2
            x1 = comp[1]
            x2 = comp[2]
            string = comp[3]
            ax.hlines(y,x1,x2, color="black")
            ax.vlines(x1, y-tick_coef*0.2,y, color="black")
            ax.vlines(x2, y-tick_coef*0.2,y, color="black")
            ax.text((x2+x1)/2-0.2, y+tick_coef/2, string)
            ax.set_ylim(top=y+tick_coef*4)
            counter = counter+1
        if counter == 0:
            y = max_value + tick_coef*5
        ax.set_ylim(min_value -tick_coef*2, y+tick_coef*6)
        ax.text(-0.35, y+tick_coef*3, test_type, size=12, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    get_lines_for_stat(sorted_list, ax)

def draw_pairwise_wilcoxon(classifier_data, dependent_var_name, subject_name, within_name, ax, xlabel, ylabel, verbose):
    """Performs pairwise wilcoxon, and graph it on ax

    Args:
        classifier_data ([pandas.dataframe]): Dataframe containing the metrics for classifiers or feature sets
        dependent_var_name ([string]): The column name containing the values (e.g. "value")
        subject_name ([string]): The column name containing the iterations (e.g. "fold")
        within_name ([string]): The column name containing the name of independent variables (e.g. "clf_type")
        ax ([object]): ax object from matplotlib that can be graphed on
        xlabel ([string]): The xlabel for the graph
        ylabel ([string]): The ylabel for the graph
        verbose ([boolean]): Whether the result of statistical tests should be displayed on console
    """
    
    wide_table = classifier_data.pivot(index=subject_name, columns=within_name, values=dependent_var_name)
    num_comparisons = len(list(itertools.combinations(range(wide_table.shape[1]), 2)))

    #Get the data from the repeated wilcoxon test
    new_stat_data = []
    for coord in itertools.combinations(range(wide_table.shape[1]), 2):
        
        pvalue = (float(pg.wilcoxon(wide_table[wide_table.columns[coord[0]]], wide_table[wide_table.columns[coord[1]]], tail='two-sided')["p-val"])) 
        corrected_pvalue = pvalue * num_comparisons
        #print(pvalue, corrected_pvalue, num_comparisons)
        if corrected_pvalue > 1:
            corrected_pvalue = 1
        new_stat_data.append([max(classifier_data[dependent_var_name]), coord[0], coord[1], corrected_pvalue])

    x_values_of_plot = wide_table.columns
    
    coord_dic = {(comp[1],comp[2]):comp for comp in new_stat_data}  

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
    plot=sns.boxplot(data=classifier_data, x=within_name, y=dependent_var_name, ax=ax, order=x_values_of_plot)
    sns.swarmplot(x=within_name , y=dependent_var_name, data=classifier_data, ax=ax, order=x_values_of_plot, color="#5d6166", size=5)


    min_value = min(classifier_data[dependent_var_name])
    max_value = max(classifier_data[dependent_var_name])

    tick_coef = (max_value - min_value)/10


    def get_lines_for_stat(data, ax):
        # max_y, x1, x2, string 
        counter = 0
        y = max_value #+ tick_coef*5
        pval_formatted = float(f"{list(data[0])[3]:.3e}")
        symbol = "NS"
        if 0.01 <= pval_formatted <0.05:
            symbol = "*"
        elif 0.001 <= pval_formatted <= 0.01:
            symbol = "**"
        elif pval_formatted < 0.001:
            symbol = "***"
        ax.text(-0.35, y+tick_coef, f"{pval_formatted} ({symbol})", size=15, color="black")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min_value -tick_coef*2, y+tick_coef*3)
        return ax

    get_lines_for_stat(sorted_list, ax)



def figure_3_1():
    """Performing hierarchical clustering on the dataset
    """
    directory = "C:/TPP/data/soroush/output_spatial_data/"
    original_0DS_training_dataset = "training_0.00_BIG_.csv"

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))

    
    #Read the data
    X_train, y_train = pre_process_features(directory+original_0DS_training_dataset, spatial="All_features", 
                                            testing=False, target_column="target", preprocessing="MinMaxScale")


    #Color the target values
    lut = dict(zip(y_train.unique(), ["#5c1a33","#749dae"]))

    #Color the rows
    row_colors = y_train.map(lut)
    col_colors = ["#876D57" if col in feature_names.POI_features_only else "#f7dc6a" for col in X_train.columns]

    #Draw cluster map
    cluster_map = sns.clustermap(X_train.sample(frac=0.20, random_state=1),row_colors=row_colors, col_colors=col_colors, figsize=(24,13.5))
    cluster_map.fig.subplots_adjust(right=0.7)
    #cluster_map.ax_cbar.set_position((0.8, 1, .03, .4))
    POI_features_patch = mpatches.Patch(color='#876D57', label='POI features')
    spatial_features_patch = mpatches.Patch(color='#f7dc6a', label='Spatial features')
    P_present_patch = mpatches.Patch(color='#5c1a33', label='Confidently identified proteins')
    P_not_present_patch = mpatches.Patch(color='#749dae', label='Proteins not confidently identified')
    plt.legend(handles=[POI_features_patch,spatial_features_patch, P_present_patch, P_not_present_patch],bbox_to_anchor=(3.5, 0.5, 1.5, 0.5),fontsize=16)

    plt.savefig("Hierarchical_clustering_Figure15_2.png",dpi=150)

def figure_3_2():
    """Create the correlation heatmap of the features in the dataset
    """
    directory = "C:/TPP/data/soroush/output_spatial_data/"
    original_0DS_training_dataset = "training_0.00_BIG_.csv"

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))

    
    #Read the data
    X_train, y_train = pre_process_features(directory+original_0DS_training_dataset, spatial="All_features", 
                                            testing=False, target_column="target", preprocessing="MinMaxScale")

        
    sns.set_theme(style="white",font_scale=1.6)


    # Compute the correlation matrix
    corr = X_train.corr(method="pearson")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.rcParams['font.size'] = 50

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 30, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0,vmin=-1.0, center=0,ax=ax,
                square=True, linewidths=.5, cbar_kws={"location":"top", "use_gridspec" : False,"shrink": .5, "anchor":(0.5,2)})

    plt.tight_layout()
    plt.savefig("Correlation_matrix_Figure16.png",dpi=250)


def load_or_create_obj1_1_dataset(mode):
    """Perfoming 10 fold training and testing scheme and save (or load) the info as pickle object

    Args:
        mode ([string]): "create" or "load" the info

    Returns:
        [(pandas.dataframe, dict)]: The dataframe with performance info. The dict with ROC PR information.
    """
    if mode == "create":
        #perform the 10 fold training for objective 1.1 (SAPID on the original dataset)
        training_dataset_name = "C:/TPP/data/soroush/output_spatial_data/training_0.00_BIG_.csv"
        testing_dataset_name = "C:/TPP/data/soroush/output_spatial_data/testing_0.00_BIG_.csv"
        performance_data, ROC_PR_data = validation_X_fold(training_dataset_name, testing_dataset_name, X_fold=10 ,decoy_string=decoy_prefix)
        pd.to_pickle(performance_data, "obj1_1_fold_validation_July24.pkl")

        with open('obj1_1_ROC_PR_July24.pkl', 'wb') as handle:
            pickle.dump(ROC_PR_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #Read the dataset for statistical tests
        performance_data = pd.read_pickle("obj1_1_fold_validation_July24.pkl")

        with open('obj1_1_ROC_PR_July24.pkl', 'rb') as handle:
            ROC_data = pickle.load(handle)

    elif mode == "load":
        #Read the dataset for statistical tests
        performance_data = pd.read_pickle("obj1_1_fold_validation_July24.pkl")

        with open('obj1_1_ROC_PR_July24.pkl', 'rb') as handle:
            ROC_data = pickle.load(handle)

        #Get rid of the "_" in the feature type names
        performance_data.feature_type = performance_data.feature_type.apply(lambda x: x.replace("_"," "))


        #Get rid of Decision tree and Guassian NB test_Target_match values
        bad_index = performance_data[(performance_data["metric"] == "test TargetMatch") & (performance_data["clf_type"] == "Decision\nTree")].index
        performance_data = performance_data.drop(bad_index)

        bad_index = performance_data[(performance_data["metric"] == "test TargetMatch") & (performance_data["clf_type"] == "Gaussian\nNB")].index
        performance_data = performance_data.drop(bad_index)

        #Get rid of Dummy data
        bad_index = performance_data[performance_data["clf_type"] == "Dummy\nCLF"].index
        performance_data = performance_data.drop(bad_index)

    return performance_data, ROC_data


def figure_3_3(mode):
    """Performs the Friedman test on the performance of four classifiers and three feature sets

    Args:
        mode ([string]): Different statistical approach that can be done: "ARM_or_Friedman", "Wilcoxon".
                         WE DECIDED ON FRIEDMAN

    Returns:
        [void]: PDF of the graph in the output directory
    """
    performance_data, _ = load_or_create_obj1_1_dataset(mode="load")

    def get_ylim(df, column_name):
        min_value = min(df[column_name])
        max_value = max(df[column_name])

    metric_lists = ["test Recall", "test Specificity", "test Precision", "test F1","test ROCAUC", "test PRAUC"]
    feature_types_list = ["All features", "POI features only", "Spatial features only"]

    #Return the max y-values of the texts inside the axes of the ANOVa and friedman figures
    def get_max_y_value_ax(ax):
        y_values = []
        for i in range(len(ax)):
            y_values.append(ax[i].get_unitless_position()[1])

        return max(y_values)

    fig, axs = plt.subplots(len(metric_lists),len(feature_types_list), figsize=(16.6,23.4),sharex="col", gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.set_facecolor('white')

    counter = 0

    for metric in metric_lists:
        for feature_type in feature_types_list:
            
            data_for_plotting = performance_data[(performance_data.metric == metric) & (performance_data.feature_type == feature_type)]
            ylim = get_ylim(data_for_plotting, "value")

            if mode=="ARM_or_Friedman":
                draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="clf_type", 
                                        ax=axs.flat[counter], xlabel="Classifiers", ylabel=metric, verbose=False)
            elif mode=="Wilcoxon":
                draw_pairwise_wilcoxon(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="clf_type", 
                                        ax=axs.flat[counter], xlabel="Classifiers", ylabel=metric, verbose=False)

            axs.flat[counter].set_ylabel("")
            if counter < 3:
                axs.flat[counter].set_title(feature_type, fontsize=20)
            if counter == 0 or counter%3==0:
                axs.flat[counter].set_ylabel(metric, fontsize=18)
            
            axs.flat[counter].set_xlabel("")
            
            if counter == 0:
                counter_size_adj = 0
                min_y = []
                max_y = []
                
            if counter_size_adj > 2:
                min_y = min(min_y)
                max_y = max(max_y)
                
                for i in [1,2,3]:
                    ax_num = counter - i
                    axs.flat[ax_num].set_ylim(min_y, max_y+0.1)
                    axs.flat[ax_num].axes.texts[len(axs.flat[ax_num].axes.texts)-1].set_y(max_y+0.07)
                    

                
                counter_size_adj = 0
                min_y = []
                max_y = []
            else:
                min_y.append(axs.flat[counter].get_ylim()[0])
                max_y.append(get_max_y_value_ax(axs.flat[counter].axes.texts))
                
            
            
            
            axs.flat[counter].label_outer()
            counter_size_adj += 1
            counter += 1
            
            plt.savefig(f'Obj1_1_stat_correct_final_{mode}.pdf')


def get_metric_ROC_PR(data, ml_name, feature_type, metric):
    """Returns the specified metric from the dictionary that is storing the ROC and PR information

    Args:
        data ([dict]): Dictionary with ROC PR info
        ml_name ([string]): Name of the machine learning algorithm
        feature_type ([string]): Type of the feature set
        metric ([string]): The metric that needs to be returned: "TPR", "FPR", "Precision"

    Returns:
        [list]: 
    """
    metric_list = []
    for i in range(10):
        name = f"{ml_name}_{feature_type}_{i+1}_{metric}"
        result = data[name]
        if metric == "Precision":
            result[0] = 0.0
            result[-1] = 1.0
                
            metric_list.append(result)
                
        elif metric == "TPR" or metric == "FPR":
            result[0] = 1.0
            result[-1] = 0.0
            
            metric_list.append(result)

    return metric_list

def draw_roc_from_dic(data, ml_name, feature_type, linestyle, ax, color):
    """Plots the ROC curve from dictionary storing the ROC PR info

    Args:
        data ([dict]): Dictionary containing the ROC PR info
        ml_name ([string]): Name of the machine learning algorithm
        feature_type ([string]): "All_features", "POI_features_only", "Spatial_features_only"
        linestyle ([string]): The type of line style used in matplotlib
        ax ([object]): The ax object from matplotlib to draw the ROC curve
        color ([string]): The color of the curve
    """
    tprs = get_metric_ROC_PR(data, ml_name, feature_type, metric="TPR")
    fprs = get_metric_ROC_PR(data, ml_name, feature_type, metric="FPR")
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    
    label = ml_name.replace("\n"," ")
    title = feature_type.replace("_"," ")

    
    ax.plot(mean_fpr, mean_tpr, color=color,
        lw=2, alpha=.8, linestyle=linestyle, label=label)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)
    ax.set_title(title,fontsize=15,color="black")
    
def draw_PR_from_dic(data, ml_name, feature_type, linestyle, ax, color):
    """Plots the PR curve from dictionary storing the ROC PR info

    Args:
        data ([dict]): Dictionary containing the ROC PR info
        ml_name ([string]): Name of the machine learning algorithm
        feature_type ([string]): "All_features", "POI_features_only", "Spatial_features_only"
        linestyle ([string]): The type of line style used in matplotlib
        ax ([object]): The ax object from matplotlib to draw the PR curve
        color ([string]): The color of the curve
    """
    tprs = get_metric_ROC_PR(data, ml_name, feature_type, metric="TPR")
    precisions = get_metric_ROC_PR(data, ml_name, feature_type, metric="Precision")
  
    mean_tpr = np.mean(tprs, axis=0)
    mean_precisions = np.mean(precisions, axis=0)
    
    label = ml_name.replace("\n"," ")
    title = feature_type.replace("_"," ")

    
    ax.plot(mean_tpr, mean_precisions,  color=color,
        lw=2, alpha=.8, linestyle=linestyle, label=label)
    
    
    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precisions + std_precision, 1)
    precision_lower = np.maximum(mean_precisions - std_precision, 0)
    ax.fill_between(mean_tpr, precision_lower, precision_upper, color=color, alpha=.2)
    ax.set_title(title,fontsize=15,color="black")

def figure_3_4():
    """ROC and PR curves of different classifiers with different feature sets. 
    """
    plt.rcParams.update({'font.size': 15})

    _, data = load_or_create_obj1_1_dataset(mode="load")

    fig, axs = plt.subplots(1,3, figsize=(16.6,5.2), sharey="row", gridspec_kw={'hspace': 0.0, 'wspace': 0.1})
    for i, (ax, feature_type) in enumerate(zip(axs.flat, ["All_features", "POI_features_only", "Spatial_features_only"])):
        
        draw_roc_from_dic(data, ml_name="Decision\nTree", feature_type=feature_type,linestyle="-", ax=ax, color="tab:blue")
        draw_roc_from_dic(data, ml_name="Gaussian\nNB", feature_type=feature_type,linestyle="-", ax=ax, color="tab:orange")
        draw_roc_from_dic(data, ml_name="LDA", feature_type=feature_type, ax=ax,linestyle="-", color="tab:green")
        draw_roc_from_dic(data, ml_name="Logistic\nRegression", feature_type=feature_type,linestyle="-", ax=ax, color="tab:red")
        draw_roc_from_dic(data, ml_name="Dummy\nCLF", feature_type=feature_type,linestyle="--", ax=ax, color="black")
        
        ax.set_title(feature_type.replace("_", " "), fontsize=20)
        
        if i == 0:
            ax.set_ylabel("TPR (True positive rate)",fontsize=18,color="black")
        else:
            ax.yaxis.set_ticks_position('none') 
            
        if i == 2:
            ax.legend()

        ax.set_xlabel("FPR (False positive rate)", fontsize=18,color="black")

    plt.savefig("ROC_ML.pdf")

    plt.rcParams.update({'font.size': 15})

    fig, axs = plt.subplots(1,3, figsize=(16.6,5.2), sharey="row", gridspec_kw={'hspace': 0.0, 'wspace': 0.1})

    for i, (ax, feature_type) in enumerate(zip(axs.flat, ["All_features", "POI_features_only", "Spatial_features_only"])):
        
        draw_PR_from_dic(data, ml_name="Decision\nTree", feature_type=feature_type,linestyle="-", ax=ax, color="tab:blue")
        draw_PR_from_dic(data, ml_name="Gaussian\nNB", feature_type=feature_type,linestyle="-", ax=ax, color="tab:orange")
        draw_PR_from_dic(data, ml_name="LDA", feature_type=feature_type,linestyle="-", ax=ax, color="tab:green")
        draw_PR_from_dic(data, ml_name="Logistic\nRegression", feature_type=feature_type,linestyle="-", ax=ax, color="tab:red")
        # draw_PR_from_dic(data, ml_name="Dummy\nCLF", feature_type=feature_type,linestyle="--", ax=ax, color="black")
        ax.plot([1, 0], [0, 1], color='black', linestyle="--", label="Dummy\nCLF")

        ax.set_title(feature_type.replace("_", " "), fontsize=20)
        
        
        if i == 0:
            ax.set_ylabel("Precision",fontsize=18,color="black")
        else:
            ax.yaxis.set_ticks_position('none') 
            
        if i == 2:
            ax.legend()

        ax.set_xlabel("TPR (True positive rate)",fontsize=18,color="black")

    plt.savefig("PR_ML.pdf")



def figure_3_5(mode):
    """ Comparing the number of proteins identified at a FDR <1% for different feature sets and LDA and logisticregression

    Args:
        mode ([string]): The type of statistical test to perform: "ttest_or_wilcoxon" or "Wilcoxon"

    Returns:
        [void]: PDF of the graph in the output directory
    """
    plt.rcParams.update({'font.size': 15})
    performance_data, _ = load_or_create_obj1_1_dataset(mode="load")

    metric_lists = ["test TargetMatch"]
    feature_types_list = ["All features", "POI features only", "Spatial features only"]
    # plt.rcParams.update({'font.size': 12})

    fig, axs = plt.subplots(1,3, figsize=(16.6,5.2))
    fig.set_facecolor('white')

    #Return the max y-values of the texts inside the axes of the ANOVa and friedman figures
    def get_max_y_value_ax(ax):
        y_values = []
        for i in range(len(ax)):
            y_values.append(ax[i].get_unitless_position()[1])

        return max(y_values)

    # fig, ax = plt.subplots(1, figsize=(5.4,8))
    # fig.set_facecolor('white')

    counter = 0

    for metric in metric_lists:
        for feature_type in feature_types_list:
            ax = axs.flat[counter]
            
            data_for_plotting = performance_data[(performance_data.metric == metric) & (performance_data.feature_type == feature_type)]
            #ylim = get_ylim(data_for_plotting, "value")
            if mode == "ttest_or_wilcoxon":
                t_test_or_wilcoxon(data_for_plotting, dependent_var_name="value", within_name="clf_type", 
                                        ax=ax, xlabel="Classifiers", ylabel=metric, verbose=False)
            elif mode == "Wilcoxon":
                draw_pairwise_wilcoxon(data_for_plotting, dependent_var_name="value", subject_name="fold" ,within_name="clf_type", 
                                        ax=ax, xlabel="Classifiers", ylabel=metric, verbose=False)
            
            ax.set_xlabel("")
            ax.set_title(feature_type, fontsize=20)
            if counter == 0:
                ax.set_ylabel(metric, fontsize=20,color="black")
            else:
                ax.set_ylabel("", fontsize=18,color="black")
            
            counter += 1
        
        
    fig.savefig(f'Obj1_1_stat_target_match_ttest_{mode}.pdf')

def figure_3_6(mode):
    """Comparing different metrics and feature sets for different classifiers on distinguishing correct and incorrect protein matches

    Args:
        mode ([string]): The statistical test to use: "ARM_or_Friedman", "Wilcoxon"

    Returns:
        [void]: The PDF of the figure stored in the output directory
    """
    performance_data, _ = load_or_create_obj1_1_dataset(mode="load")
    performance_data.feature_type = performance_data.feature_type.apply(lambda x: x.replace(" ","\n"))

    metric_lists = ["test Recall", "test Specificity", "test Precision", "test F1","test ROCAUC", "test PRAUC"]
    clf_types_list = ['Decision\nTree', 'Gaussian\nNB', 'LDA', 'Logistic\nRegression']

    def get_ylim(df, column_name):
        min_value = min(df[column_name])
        max_value = max(df[column_name])


    #Return the max y-values of the texts inside the axes of the ANOVa and friedman figures
    def get_max_y_value_ax(ax):
        y_values = []
        for i in range(len(ax)):
            y_values.append(ax[i].get_unitless_position()[1])

        return max(y_values)

    fig, axs = plt.subplots(len(metric_lists),len(clf_types_list), figsize=(16.6,23.4), sharex="col", gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.set_facecolor('white')

    counter = 0

    for metric in metric_lists:
        for clf_type in clf_types_list:
            
            data_for_plotting = performance_data[(performance_data.metric == metric) & (performance_data.clf_type == clf_type)]
            data_for_plotting = data_for_plotting.sort_values(by ='feature_type' )
            data_for_plotting = data_for_plotting.reset_index(drop=True)
            ylim = get_ylim(data_for_plotting, "value")

            if mode == "ARM_or_Friedman":
                draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="feature_type", 
                                        ax=axs.flat[counter], xlabel="Classifiers", ylabel=metric, verbose=False)
            elif mode == "Wilcoxon":
                draw_pairwise_wilcoxon(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="feature_type", 
                                        ax=axs.flat[counter], xlabel="Classifiers", ylabel=metric, verbose=False)

            axs.flat[counter].set_ylabel("")
            if counter < 4:
                axs.flat[counter].set_title(clf_type, fontsize=20)
            if counter == 0 or counter%4==0:
                axs.flat[counter].set_ylabel(metric, fontsize=18)
                
            
            axs.flat[counter].set_xlabel("")
            
            if counter == 0:
                counter_size_adj = 0
                min_y = []
                max_y = []
                
            if counter_size_adj > 3:
                min_y = min(min_y)
                max_y = max(max_y)
                
                for i in [1,2,3,4]:
                    ax_num = counter - i
                    axs.flat[ax_num].set_ylim(min_y, max_y+0.1)
                    axs.flat[ax_num].axes.texts[len(axs.flat[ax_num].axes.texts)-1].set_y(max_y+0.07)
                    

                
                counter_size_adj = 0
                min_y = []
                max_y = []
            else:
                min_y.append(axs.flat[counter].get_ylim()[0])
                max_y.append(get_max_y_value_ax(axs.flat[counter].axes.texts))
                
            
            axs.flat[counter].label_outer()
            counter_size_adj += 1
            counter += 1
            
    plt.savefig(f'Obj1_1_stat2_corrected_{mode}.pdf')

def figure_3_7(mode):
    """Comparing the number of protein identifications identiefied at decoy-based FDR >1% for LDA and logistic regression with different feature sets.

    Args:
        mode ([string]): Different statistical approach that can be done: "ARM_or_Friedman", "Wilcoxon".
                         WE DECIDED ON FRIEDMAN

    Returns:
        [void]: The PDF of the figure stored in the output directory
    """

    performance_data, _ = load_or_create_obj1_1_dataset(mode="load")
    performance_data.feature_type = performance_data.feature_type.apply(lambda x: x.replace(" ","\n"))

    metric_lists = ["test TargetMatch"]
    clf_types_list = ['LDA', 'Logistic\nRegression']
    import numpy as np

    #Return the max y-values of the texts inside the axes of the ANOVa and friedman figures
    def get_max_y_value_ax(ax):
        y_values = []
        for i in range(len(ax)):
            y_values.append(ax[i].get_unitless_position()[1])

        return max(y_values)

    fig, axs = plt.subplots(1,2, figsize=(17,8))
    fig.set_facecolor('white')

    counter = 0

    for metric in metric_lists:
        for i, clf_type in enumerate(clf_types_list):
    #         fig, ax = plt.subplots(1, figsize=(5.4,8))
    #         fig.set_facecolor('white')
            
            data_for_plotting = performance_data[(performance_data.metric == metric) & (performance_data.clf_type == clf_type)]
            #ylim = get_ylim(data_for_plotting, "value")
            if mode=="ARM_or_Friedman":
                draw_ANOVA_or_Friedmann(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="feature_type", 
                                        ax=axs.flat[i], xlabel="Classifiers", ylabel=metric, verbose=False)
            elif mode=="Wilcoxon":
                draw_pairwise_wilcoxon(data_for_plotting, dependent_var_name="value", subject_name="fold", within_name="feature_type", 
                                        ax=axs.flat[i], xlabel="Classifiers", ylabel=metric, verbose=False)
            
            axs.flat[i].set_xlabel("")
            axs.flat[i].set_title(clf_type, fontsize=20)
            axs.flat[i].set_ylabel("")
            if i == 0:
                axs.flat[i].set_ylabel(metric, fontsize=18,color="black")
            
            #ax.label_outer()
            
    plt.tight_layout()     
    fig.savefig(f'Obj1_1_stat_target_match2_{mode}.pdf')

def get_proteins():
    """Perform the classification using SAPID and PP and then store the accession number of proteins and decoys matches at the specified FDR for later use
    """
    directory="C:/TPP/data/soroush/output_spatial_data/"
    FDR = 0.01
    ratio = "0.00"

    training_data_name = directory+"training_"+ratio+"_BIG_.csv"
    testing_data_name = directory+"testing_"+ratio+"_BIG_.csv"
    
    #Perform the classification
    FDR_classifier_info, protein_classifier_info, FDR_PP_dic, protein_PP_dic = get_protein_hits_at_FDR(training_filename=training_data_name, 
        testing_filename=testing_data_name, target_FDR=FDR, decoy_string=decoy_prefix, clf_type="Logistic\nRegression", 
        spatial="All_features", over_under_sample="None", grid_search_scoring="f1", get_PP=True)

    with open('FDR_classifier_info_Aug9.pkl', 'wb') as handle:
        pickle.dump(FDR_classifier_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('protein_classifier_info_Aug9.pkl', 'wb') as handle:
        pickle.dump(protein_classifier_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('FDR_PP_dic_Aug9.pkl', 'wb') as handle:
        pickle.dump(FDR_PP_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('protein_PP_dic_Aug9.pkl', 'wb') as handle:
        pickle.dump(protein_PP_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


def figure_3_8(mode):
    """Plot the sum of the total number of protein instances identified in the testing dataset with SAPID, Percolator, 
       and ProteinProphet at different decoy-based FDR thresholds.

    Args:
        mode ([string]): only "load" the dataset and make the graph, or "create" the dataset beforehand from fresh
    """
    
    if mode == "create":
        directory="C:/TPP/data/soroush/output_spatial_data/"
        training_data_name = "training_0.00_BIG_.csv"
        testing_data_name = "testing_0.00_BIG_.csv"
        plot_info_dic = dict()
        for feature_types in ["All_features", "POI_features_only", "Spatial_features_only"]:
            plot_info_dic[feature_types] = analyse_file_plotting(directory+training_data_name, directory+testing_data_name, num_threshold=300, decoy_string=decoy_prefix, 
                    clf_type="Logistic\nRegression", spatial=feature_types, over_under_sample="None", grid_search_scoring="f1")

        with open('result_of_FDR_banchmark_PP_SAPID_June7_2021.pkl', 'wb') as handle:
            pickle.dump(plot_info_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('result_of_FDR_banchmark_PP_SAPID_June7_2021.pkl', 'rb') as handle:
            plot_info_dic = pickle.load(handle)

    elif mode == "load":
        with open('result_of_FDR_banchmark_PP_SAPID_June7_2021.pkl', 'rb') as handle:
            plot_info_dic = pickle.load(handle)


    #Get Percolator's FDR info
    percolator_output_directory = "C:/TPP/data/percolator/"
    FDR_percolator, protein_matched_percola = percolator_calc_FDR_plot(percolator_output_dir=percolator_output_directory,
                                                                        downsampled_ratio="0.00", number_of_points=300)


    plot_info_dic["SAPID-All features"] = plot_info_dic.pop("All_features")
    plot_info_dic["SAPID-POI features only"] = plot_info_dic.pop("POI_features_only")
    plot_info_dic["SAPID-Spatial features only"] = plot_info_dic.pop("Spatial_features_only")

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})

    plt.figure(figsize=(24, 16))
    for feature_types in ['SAPID-All features', 'SAPID-POI features only', 'SAPID-Spatial features only']:
        FDR_classifier_info_dic = plot_info_dic[feature_types]["classifier_FDR_dic"]
        FDR_PP_info_dic = plot_info_dic[feature_types]["PP_FDR_dic"]
        
        classifier_FDR_list = []
        PP_FDR_list = []
        classifier_protein_list = []
        PP_protein_list = []

        for thr in FDR_classifier_info_dic:
            classifier_FDR_list.append(plot_info_dic[feature_types]["classifier_FDR_dic"][thr][0])
            classifier_protein_list.append(plot_info_dic[feature_types]["classifier_FDR_dic"][thr][1])
            
        for thr in FDR_PP_info_dic:
            PP_FDR_list.append(plot_info_dic[feature_types]["PP_FDR_dic"][thr][0])
            PP_protein_list.append(plot_info_dic[feature_types]["PP_FDR_dic"][thr][1])
            
        if feature_types == 'SAPID-All features':
            color=matplotlib.cm.get_cmap('tab20c')(0)
            marker="8"
        elif feature_types == 'SAPID-POI features only':
            color=matplotlib.cm.get_cmap('tab20c')(0.2)
            marker="s"
        else:
            color=matplotlib.cm.get_cmap('tab20c')(0.4)
            marker="p"
            
        plt.plot(classifier_FDR_list, classifier_protein_list, marker,markersize=10,color=color, label=feature_types)
    
    plt.plot(FDR_percolator, protein_matched_percola, "P",markersize=10,color=matplotlib.cm.get_cmap('tab20c')(0.6), label="Percolator")

    plt.plot(PP_FDR_list, PP_protein_list,"h",markersize=10,color=matplotlib.cm.get_cmap('tab20c')(0.8), label="ProteinProphet")
    plt.xlim(0,0.1)
    plt.ylim(0,60000)
    plt.ylabel("Sum of the number of \n proteins identified in all pixels")
    plt.xlabel("FDR")
    plt.vlines(x=0.01,ymin=0, ymax=60000, linestyles='--', color="red")
    plt.text(x=0.011, y=10000, s="FDR=1.0%", fontsize=30)
    plt.legend(loc="lower right", markerscale=2)
    plt.tight_layout()
    plt.savefig("FDR_vs_proteins_originalDataset2.pdf")

def original_0DS_PP_proteins(filename):
    """Get the correctly identified proteins at 0% downsampling according to PP

    Args:
        filename ([string]): Full path and name of a csv dataset

    Returns:
        [list]: List of accession for the correctly identified proteins at 0% downsampling according to PP
    """
    data = pd.read_csv(filename)
    data = data[data["target"]==1]
    data = data[~data["accession"].str.contains(decoy_prefix)]
    return list(data["accession"])

def original_0DS_PP_proteins_pixelwise(filename):
    """Get the correctly identified proteins at 0% downsampling according to PP, but adds pixel number to the pixel accession

    Args:
        filename ([string]): Full path and name of a csv dataset

    Returns:
        [list]: List of accession for the correctly identified proteins at 0% downsampling according to PP
    """
    data = pd.read_csv(filename)
    data = data[data["target"]==1]
    data = data[~data["accession"].str.contains(decoy_prefix)]
    return list(data.apply(lambda x: str(x.POI_name) +":"+ x.accession, axis=1))

def percent_pro_SAPID_not_PP(protein_matches_PP, protein_matches_SAPID, original_dataset_file):
    """Calculates the percent of proteins that are only identified by SAPID but not PP

    Args:
        protein_matches_PP ([list]): List of proteins identified by ProteinProphet
        protein_matches_SAPID ([list]): List of proteins identified by SAPID
        original_dataset_file ([string]): Full path and name to the original csv (0% ds) dataset

    Returns:
        [float]: The percent of proteins that are only identified by SAPID but not PP
    """

    #All the proteins
    omega = set(original_0DS_PP_proteins_pixelwise(original_dataset_file))
    
    #Proteins only in SAPID but not PP
    proteins_only_in_sapid = set([protein for protein in protein_matches_SAPID if protein not in protein_matches_PP])
    
    #Check to see if all the proteins_only_in_sapid are in the omega
    # for protein in proteins_only_in_sapid:
    #     if protein not in omega:
    #         print("WARNING", protein, "not in OMEGA")

    #Remove the proteins not in omega
    print("protein only in SAPID", len(proteins_only_in_sapid))
    proteins_only_in_sapid = [protein for protein in proteins_only_in_sapid if protein in omega]
    print("protein only in SAPID", len(proteins_only_in_sapid))

    
    return (len(proteins_only_in_sapid)/len(omega))*100


def percent_pro_SAPID_not_Percolator(protein_matches_Percolator, protein_matches_SAPID, original_dataset_file):
    """Calculates the percent of proteins that are only identified by SAPID but not Percolator

    Args:
        protein_matches_Percolator ([list]): List of proteins identified by Percolator
        protein_matches_SAPID ([list]): List of proteins identified by SAPID
        original_dataset_file ([string]): Full path and name to the original csv (0% ds) dataset

    Returns:
        [float]: The percent of proteins that are only identified by SAPID but not Percolator
    """
    #All the correctly identified proteins in the ProteinProphet
    omega = set(original_0DS_PP_proteins_pixelwise(original_dataset_file))
    
    #Proteins only in SAPID but not Percolator
    proteins_only_in_sapid = set([protein for protein in protein_matches_SAPID if protein not in protein_matches_Percolator])
    
    #Remove the proteins not in omega
    print("protein only in SAPID compare to Percolator", len(proteins_only_in_sapid))
    proteins_only_in_sapid = [protein for protein in proteins_only_in_sapid if protein in omega]
    print("protein only in SAPID compare to Percolator", len(proteins_only_in_sapid))

    
    return (len(proteins_only_in_sapid)/len(omega))*100


def percent_pro_Percolator_not_SAPID(protein_matches_Percolator, protein_matches_SAPID, original_dataset_file):
    """Calculates the percent of proteins that are only identified by Percolator but not SAPID

    Args:
        protein_matches_Percolator ([list]): List of proteins identified by Percolator
        protein_matches_SAPID ([list]): List of proteins identified by SAPID
        original_dataset_file ([string]): Full path and name to the original csv (0% ds) dataset

    Returns:
        [float]: The percent of proteins that are only identified by Percolator but not SAPID
    """
    #All the correctly identified proteins in the ProteinProphet
    omega = set(original_0DS_PP_proteins_pixelwise(original_dataset_file))
    
    #Proteins only in SAPID but not PP
    proteins_only_in_Percolator = set([protein for protein in protein_matches_Percolator if protein not in protein_matches_SAPID])

    #Remove the proteins not in omega
    print("protein only in Percolator compare to SAPID", len(proteins_only_in_Percolator))
    proteins_only_in_Percolator = [protein for protein in proteins_only_in_Percolator if protein in omega]
    print("protein only in Percolator compare to SAPID", len(proteins_only_in_Percolator))

    
    return (len(proteins_only_in_Percolator)/len(omega))*100

def percent_pro_PP_not_SAPID(protein_matches_PP, protein_matches_SAPID, original_dataset_file):
    """Calculates the percent of proteins that are only identified by PP but not SAPID

    Args:
        protein_matches_PP ([list]): List of proteins identified by ProteinProphet
        protein_matches_SAPID ([list]): List of proteins identified by SAPID
        original_dataset_file ([string]): Full path and name to the original csv (0% ds) dataset

    Returns:
        [float]: The percent of proteins that are only identified by PP but not SAPID
    """

    #All the proteins
    omega = set(original_0DS_PP_proteins_pixelwise(original_dataset_file))
    
    #Proteins only in PP but not SAPID
    proteins_only_in_PP = set([protein for protein in protein_matches_PP if protein not in protein_matches_SAPID])
    
    #Check to see if all the proteins_only_in_PP are in the omega
    # for protein in proteins_only_in_PP:
    #     if protein not in omega:
    #         print("WARNING", protein, "not in OMEGA")
    #Remove proteins not in omega
    print("protein only in PP", len(proteins_only_in_PP))
    proteins_only_in_PP = [protein for protein in proteins_only_in_PP if protein in omega]
    print("protein only in PP", len(proteins_only_in_PP))

    
    return (len(proteins_only_in_PP)/len(omega))*100


def load_or_create_obj_1_2(mode):
    """Calculates the inclusions percentage of the protein matches for PP, SAPID, and Percolator on the downsampled datasets

    Args:
        mode ([string]): Whether to "create" the dataset from fresh, or just "load" and return the previously saved dataframe

    Returns:
        [pandas.dataframe]: The dataframe with information regarding % protein inclusion
    """
    if mode == "create":
        ratios = ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]
        directory="C:/TPP/data/soroush/output_spatial_data/"
        original_0DS_testing_dataset = "C:/TPP/data/soroush/output_spatial_data/testing_0.00_BIG_.csv"
        percolator_output_dir = "C:/TPP/data/percolator/"
        FDR = 0.01

        values_dic = dict()
        for ratio in ratios:
            training_data_name = directory+"training_"+ratio+"_BIG_.csv"
            testing_data_name = directory+"testing_"+ratio+"_BIG_.csv"
            
            #Perform the classification
            FDR_classifier_info, protein_classifier_info, FDR_PP_dic, protein_PP_dic = get_protein_hits_at_FDR(training_filename=training_data_name, 
                testing_filename=testing_data_name, target_FDR=FDR, decoy_string=decoy_prefix, clf_type="Logistic\nRegression", 
                spatial="All_features", over_under_sample="None", grid_search_scoring="f1", get_PP=True)
            
            
            #Get the thresholds at which the PP and SAPID have FDR close to the given FDR
            #PP_thr, classifier_thr = get_x_percent_FDR_threshold(FDR_PP_dic, FDR_classifier_info, FDR)
            
            #Use thr to get hold of the [decoy_matches, protein_matches, protein_not_matches]
            protein_matches_PP= list(protein_PP_dic.values())[0][1]
            protein_matches_SAPID = list(protein_classifier_info.values())[0][1] 
            
            #Print the thr and FDR for both PP abd SAPID
            print(ratio)
            print("Protein Prophet","thr:",list(protein_PP_dic.keys())[0], "FDR", list((FDR_PP_dic.values()))[0][0])
            print("SAPID","thr:",list(protein_classifier_info.keys())[0], "FDR", list((FDR_classifier_info.values()))[0][0])

            #Get percent inclusion
            percent_pro_SAPID_only_compare_PP = percent_pro_SAPID_not_PP(protein_matches_PP, protein_matches_SAPID, original_0DS_testing_dataset)
            percent_pro_PP_only_compare_SAPID = percent_pro_PP_not_SAPID(protein_matches_PP, protein_matches_SAPID, original_0DS_testing_dataset)
            
            _, protein_dic_percolator = get_FDR_Percolator(percolator_output_dir, downsampled_ratio=ratio, target_FDR=FDR)
            protein_matches_Percolator = list(protein_dic_percolator.values())[0][1]
            
            percent_pro_SAPID_only_compare_Percolator = percent_pro_SAPID_not_Percolator(protein_matches_Percolator, protein_matches_SAPID, original_0DS_testing_dataset)
            percent_pro_Percolator_only_compare_SAPID = percent_pro_Percolator_not_SAPID(protein_matches_Percolator, protein_matches_SAPID, original_0DS_testing_dataset)
            
            
            
            values_dic[ratio] = [percent_pro_PP_only_compare_SAPID, percent_pro_SAPID_only_compare_PP, percent_pro_SAPID_only_compare_Percolator, 
                                percent_pro_Percolator_only_compare_SAPID,len(set(protein_matches_PP)), len(set(protein_matches_SAPID)), 
                                len(set(protein_matches_Percolator)) ]
            print("\n")

        #Store the data in a dataframe
        final_data = pd.DataFrame(data=None, columns=["Downsampling Ratio", "Value", "Type"])
        for ratio in ratios:
            values = values_dic[ratio]
            
            a_series = pd.Series([ratio,values[0],"% Protein only in ProProphet"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
            
            a_series = pd.Series([ratio,values[1],"% Protein only in SAPID"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
            
            a_series = pd.Series([ratio,values[2],"% Protein only in SAPID Percolator"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
            
            a_series = pd.Series([ratio,values[3],"% Protein only in Percolator SAPID"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
            
            a_series = pd.Series([ratio,values[4],"Total correct proteins ProProphet"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
            
            a_series = pd.Series([ratio,values[5],"Total correct proteins SAPID"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
            
            a_series = pd.Series([ratio,values[6],"Total correct proteins Percolator"], index = final_data.columns)
            final_data=final_data.append(a_series,ignore_index=True)
        
        #Store the dataframe as pickle object
        pd.to_pickle(final_data, "Obj1_1_FDR_dsratio.pkl")
        final_data = pd.read_pickle("Obj1_1_FDR_dsratio.pkl")

    elif mode == "load":
        final_data = pd.read_pickle("Obj1_1_FDR_dsratio.pkl")

    return final_data


def figure_3_10():
    """ Percentage of protein instances correctly identified only by ProteinProphet or SAPID
    """

    final_data = load_or_create_obj_1_2(mode="load")
    
    final_data.loc[(final_data["Type"] == "% Protein only in ProProphet"), "Type"] = "% Protein only in ProteinProphet"
    
    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(12, 8))
    #plt.title(f"Percent protein INSTANCES only detected with PP or SAPID at {float(FDR)*100} % FDR")

    ax = sns.barplot(x="Downsampling Ratio", y="Value", hue="Type", data=
                    final_data[(final_data["Type"] == "% Protein only in ProteinProphet") | (final_data["Type"] == "% Protein only in SAPID")])
    lagend= ax.get_legend()
    lagend.set_title("")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=3, fancybox=False, shadow=False)
    plt.ylabel("Identified protein instances (%)", size=18)

    plt.savefig("percent_protein_in_instaces.pdf")


def figure_3_9():
    """Total number of protein instances identified with ProteinProphet, SAPID, and Percolator at different downsampling thresholds at a 1% decoy-based FDR threshold
    """
    final_data = load_or_create_obj_1_2(mode="load")
    final_data.loc[(final_data["Type"] == "Total correct proteins ProProphet"), "Type"] = "Total correct proteins ProteinProphet"

    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(12, 8))
    #plt.title(f"Number of unique protein INSTANCES correctly identified at {float(FDR)*100} % FDR")
    ax = sns.barplot(x="Downsampling Ratio", y="Value", hue="Type", data=
                    final_data[(final_data["Type"] == "Total correct proteins ProteinProphet") | (final_data["Type"] == "Total correct proteins SAPID") | 
                                (final_data["Type"] == "Total correct proteins Percolator")])
    leg = ax.get_legend()
    leg.set_title("")
    plt.ylabel("Sum of the number of \n proteins identified in all pixels", size=18)
    plt.savefig("FDR_vs_proteinInstances.pdf")

    


def main():
    figure_3_1()
    figure_3_2()
    figure_3_3(mode="ARM_or_Friedman")
    figure_3_4()
    figure_3_5(mode="ttest_or_wilcoxon")
    figure_3_6(mode="ARM_or_Friedman")
    figure_3_7(mode="ARM_or_Friedman")
    figure_3_8(mode="load")
    figure_3_9()
    figure_3_10()
        



if __name__=="__main__":
    main()