import time
import csv
import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg') # #http://stackoverflow.com/questions/4706451/how-to-save-a-figure-remotely-with-pylab
import matplotlib.pyplot as plt
import numpy as np
import sweeps
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold

INDEX_COL = 0
DEFAULT_NUM_BINS = 8
TOP_CODE_PERCENTILE = .99
K_FOLDS = 5
OUTPUT_HEADER = ['recall','precision','testing_accuracy','f1','threshold','AUC','training_accuracy','train_time','test_time']
RESULT_COL = '2014-12-01'

# Read and Analyze
def read_file_into_panda(filename, index_column=-1):
    '''
    Checks for file type and loads into panda
    '''

    if filename[-4:] == '.csv':
        if index_column >= 0:
            dataframe = pd.read_csv(filename,index_col=index_column)
        else:
            dataframe = pd.read_csv(filename)
    else:
        sys.exit("Filetype either not specified or currently supported")

    return dataframe

def describe_data(dataframe, output_file):
    '''
    Writes data decriptions to file, unused
    '''
   
    summary = dataframe.describe()

    summary.T.to_csv(output_file)

    # Draw Charts
    names = dataframe.columns.values
    local_dataframe = dataframe.copy()
    
    for x in range(local_dataframe.shape[1]):
        column_data = local_dataframe.ix[:,x]
        column_title = names[x]
        #make_histogram(column_data,column_title)
        make_bar_chart(column_data, column_title)

def make_bar_chart(data,title):
    '''
    Create bar charts with cut-off at pre-determined percentile, unused
    '''

    top_code_bins = get_top_coded_bins(data,DEFAULT_NUM_BINS,TOP_CODE_PERCENTILE)
    categorized_data = pd.cut(data, bins=top_code_bins,include_lowest=True).value_counts(sort=False)
    categorized_data.plot(kind='bar')
    plt.xlabel(title + ' Buckets')
    plt.xticks(rotation=15,size='small')
    plt.ylabel('Count')
    plt.ylim(0,categorized_data.max()*1.1)
    plt.title('Top-coded Bar Chart of ' + title)
    plt.margins(.5)
    plt.tight_layout()
    plt.savefig('Top-coded_Bar_Chart_' + title + '.png', format='png')
    plt.close()
        
def make_histogram(data, title):
    '''
    Creates histogram of data with flexible number of bins, no more than 10
    '''

    data_list = data.dropna().tolist()
    top_code_val = data.quantile(TOP_CODE_PERCENTILE)
    count_distinct = set(data_list)
    distinct_vals = len(count_distinct)
    num_bins = min(distinct_vals, DEFAULT_NUM_BINS)    
    data_list.hist(bins=np.linspace(0,top_code_val,num_bins+1),normed=True)
    plt.xlabel(title)
    plt.title('Histogram of ' + title)
    plt.tight_layout()
    plt.savefig('Histogram_' + title + '.png', format='png')
    plt.close()

# Filling missing data

def fill_missing_with_conditional_median(dataframe,conditional_category):
    '''
    Replaces means with means based on output variable
    '''
    local_dataframe = dataframe.copy()
    for v in local_dataframe.ix[:,conditional_category].unique():
        p = local_dataframe.ix[:,conditional_category] == v
        local_dataframe.ix[p] = local_dataframe[p].fillna(local_dataframe[p].median())

    return local_dataframe

# Bins
def get_top_coded_bins(data_column,num_bins,top_code_percentile=1):
    '''
    Sets categories assuming high outliers, uses given top percintile to limit binning over irrelevant numbers, defaults to no cap
    '''
    top_code_val = data_column.quantile(top_code_percentile)
    bins = np.arange(0,top_code_val,top_code_val/num_bins)
    distinct_bins = bin_list_conversion(bins,data_column.max())
    return distinct_bins

def bin_list_conversion(bins,data_max):
    '''
    Cleans and returns list of bins
    '''        
    
    distinct_bins = set(bins)
    distinct_bins = list(distinct_bins)
    distinct_bins.sort()
    distinct_bins.append(data_max)
        
    return distinct_bins

def get_continuous_variables(dataframe):
    '''
    Returns all variables that have more than 2 categorizedes, ie are not binary
    '''
    column_names = dataframe.columns.values
    continuous = []
    for name in column_names:
        if len(dataframe[name].unique()) > 2:   
            continuous.append(name)
    return continuous

# Model selection
def get_best_models():
    '''
    List of models with hard coded parameters with best parameters selected
    '''

    neighbors = {'model':KNeighborsClassifier(n_neighbors=2,algorithm='auto'),'name': 'KNeighbors','type':'prob'}
    decision_tree = {'model':DecisionTreeClassifier(max_depth=7),'name': "Decision Tree",'type':"prob"}
    random_forest = {'model':RandomForestClassifier(n_estimators=10,max_depth=5,n_jobs=-1),'name': "RandomForest",'type':"prob"}
    bagging = {'model':BaggingClassifier(n_estimators=14), 'name': "Bagging",'type':"prob"}
    log_reg = {'model':LogisticRegression(), 'name': "LogReg",'type':'decision'}
    svm = {'model':LinearSVC(C=.5,multi_class='ovr'),'name': "SVM",'type':'decision'}
    boosting = {'model':AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=10,max_depth=5,n_jobs=-1),n_estimators=30,learning_rate=.3),'name': "Boosting",'type':'decision'}
    
    return [svm,neighbors,decision_tree,random_forest,boosting,bagging,log_reg]

def convert_column_to_categories(dataframe,column,bins,labels,keep_initial=False):
    '''
    Makes new dataframe with categorized columns
    '''
    
    dataframe.loc[:,column] = pd.cut(dataframe[column],bins=bins,labels=labels,include_lowest=True)
    copy = dataframe[column].copy()
    dummied_dataframe = pd.get_dummies(dataframe,columns=[column])
    dummied_columns = [column + '_' + x for x in labels]
    if keep_initial:
        dummied_dataframe[column] = copy

    return dummied_dataframe, dummied_columns

# Helper functions
def sigmoid_array(array):
    '''
    Used to convert decision rankings to a 0-1 probabilistic scale
    '''
    return 1 / (1 + np.exp(-array))

def impute_and_standardize(dataframe,test_column):
    '''
    Imputation with median and standard scaling
    '''
    train_columns = get_train_columns(dataframe,test_column)
    dataframe.fillna(dataframe.median(),inplace=True)
    scaler = StandardScaler()
    for x in train_columns:
        if len(dataframe[x].unique()) > 2:
            # Scale none-binary 
            dataframe.loc[:,x] = scaler.fit_transform(dataframe.loc[:,x].astype(float))

def get_train_columns(dataframe,test_column):

    return list(set(dataframe.columns.values) - {test_column})

# Model evaluations

def evaluate_multi_class_model(predictions,actuals):
 
    f1 = f1_score(actuals,predictions,average='weighted')
    accuracy = accuracy_score(actuals,predictions)
    
    return {'precision':0,'recall':0, 'testing_accuracy':accuracy, 'f1':f1,'threshold':0,'AUC':0,'predictions':predictions}

def evaluate_binary_model(predictions, actuals, model_dict,optimizer,include_chart=False):
    '''
    Run through thresholds and determine best one, output precision/recall curve to file and return line
    '''
    precision_list = []
    recall_list = []
    best_threshold = {'f1':0,'precision':0,'recall':0,'testing_accuracy':0,'predictions':[] }
    
    # As all are normalized 0-1, this range provides appropriate granularity
    for x in np.arange(0,1, .025):
        threshold_dict = calculate_binary_evaluation_metrics(predictions,actuals,x)
        precision_list.append(threshold_dict['precision'])
        recall_list.append(threshold_dict['recall'])
        
        # Update if threshold is best based on important metric
        if threshold_dict[optimizer] > best_threshold[optimizer]:
            best_threshold = threshold_dict


    # Add AUC to dictionary and create histogram for precision recall curve
    AUC = roc_auc_score(actuals,predictions)
    best_threshold['AUC'] = AUC
    
    if include_chart:
        plt.plot(recall_list,precision_list)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title('Precision/Recall of ' + model_dict['name']+ '_' + model_dict['filename'])
        plt.tight_layout()
        plt.savefig(model_dict['filename'] +'_Precision_Recall_of' + model_dict['name'] + '.png', format='png')
        plt.close()


    return best_threshold
  
def calculate_binary_evaluation_metrics(predictions, actuals,threshold):
    '''
    Compare actual and predict values with a threshold and return stats on evaluation metrics
    '''

    threshold_predictions =[]
    for prediction in predictions:
        if prediction < threshold:
            rounded_prediction = 0
        else:
            rounded_prediction = 1
        threshold_predictions.append(rounded_prediction)

    f1 = f1_score(actuals,threshold_predictions)
    precision = precision_score(actuals,threshold_predictions)
    recall = recall_score(actuals,threshold_predictions)
    accuracy = accuracy_score(actuals,threshold_predictions)

    return {'precision':precision,'recall':recall, 'testing_accuracy':accuracy, 'f1':f1,'threshold':threshold,'predictions':threshold_predictions}

def write_confusion_and_features(model_dict,write_file_name):
    '''
    Prints the important features and confusion matrix
    '''

    features = model_dict['importances']
    column_names = model_dict['column_names']
    prediction_tuples = model_dict['prediction_tuples']
    importances = model_dict['importances']
    index_list = []
    predictions = []
    actuals = []
    length = len(prediction_tuples)
    for prediction_tuple in prediction_tuples:
        index_list.append(prediction_tuple[0])
        predictions.append(prediction_tuple[1])
        actuals.append(prediction_tuple[2])
   
    prediction_dataframe = pd.DataFrame(index=index_list)
    prediction_dataframe['actuals'] = actuals
    prediction_dataframe['predictions'] = predictions
    np.savetxt(write_file_name +'_confusion.csv',confusion_matrix(actuals,predictions),delimiter=",",fmt = '%s')
    prediction_dataframe.to_csv(write_file_name+'_predictions.csv')
    f = open(write_file_name+'_features.csv','w')
    f.write("name,importance for binary models\n")

    for x in range(len(column_names)):
        line = [str(column_names[x]),str(importances[x])]
        f.write(",".join(line) + '\n')

# Folds and running model
def pool_k_folds(dataframe,optimizer,parameterized_model,model_dict,test_column,folds=K_FOLDS,verbose=False):
    '''
    Runs k folds through multiprocessing
    '''
    p = Pool(folds)
    kf = KFold(len(dataframe),n_folds=folds,shuffle=True)

    training_data_list = []
    testing_data_list = []
    for train_rows, test_rows in kf:
        testing_data = dataframe.ix[test_rows].copy()
        training_data = dataframe.ix[train_rows].copy()
        impute_and_standardize(testing_data,test_column)
        impute_and_standardize(training_data,test_column)
        training_data_list.append(training_data)
        testing_data_list.append(testing_data)
        
    model_list = [parameterized_model] * folds
    optimizer_list = [optimizer] * folds
    model_dict_list = [model_dict] * folds
    test_column_list = [test_column] * folds
    chart_bools = [False] * folds
    chart_bools[0] = True
    if verbose:
        verbose_list = [True]*folds
    else:
        verbose_list = [False]*folds

    args = zip(training_data_list,testing_data_list,model_list,optimizer_list,model_dict_list,test_column_list,chart_bools,verbose_list)
    
    dicts = p.map(unzip_and_run_model,args)
    
    return combine_model_dicts(dicts,folds)

def unzip_and_run_model(args):

    return run_model(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7])

def combine_model_dicts(list_of_dicts,folds):
    '''
    Combines the results for each fold to final summed results, provides predictions for all items in a tupled dictionary
    '''
    rv = {'prediction_tuples':[]}
    for dictionary in list_of_dicts:
        # Build full list of predictors
        for x in range(len(dictionary['test_indicies'])):
            index = dictionary['test_indicies'][x]
            prediction = dictionary['predictions'][x]
            actual = dictionary['actuals'][x]
            rv['prediction_tuples'].append((index,prediction,actual))

        for key in dictionary:
            if key in ('predictions','test_indicies','actuals'):
                continue
            rv[key] = rv.get(key,0) + np.array(dictionary[key])

    for key in rv:
            if key in ('prediction_tuples'):
                continue
            try:
                rv[key] = round(rv.get(key,0)/folds,2)
            except:
                #print rv[key]
                rv[key] = rv.get(key,0)/folds
    return rv        


def run_model(training_data,testing_data,model,optimizer,model_dict,test_column,make_charts=False,verbose=False):

    train_columns = get_train_columns(training_data,test_column) 
    if verbose:
        print "Fitting Model"
    start_time = time.time()
    model.fit(training_data[train_columns],training_data[test_column])
    train_time = time.time()
    if verbose:
        print "Model Fit"


    if len(training_data[test_column].unique()) > 2:
        # Multiple classes in a column
        testing_predictions = model.predict(testing_data[train_columns])
        end_time = time.time()
        training_predictions = model.predict(training_data[train_columns])
        training_metrics = evaluate_multi_class_model(training_predictions,training_data[test_column].tolist())
        model_metrics = evaluate_multi_class_model(testing_predictions,testing_data[test_column].tolist())

    else:
        # Binary
        if model_dict['type'] == 'prob':
            # Probability model
            probability_predictions = model.predict_proba(testing_data[train_columns])[:,1]
            end_time = time.time()  
            training_predictions = model.predict_proba(training_data[train_columns])[:,1]
            training_metrics = evaluate_binary_model(training_predictions,training_data[test_column].tolist(),model_dict,optimizer)
            model_metrics = evaluate_binary_model(probability_predictions,testing_data[test_column].tolist(),model_dict,optimizer,make_charts)    
        else:
            # Decision function
            decision_metrics = model.decision_function(testing_data[train_columns])
            end_time = time.time()
            decision_metrics_training = model.decision_function(training_data[train_columns])
            # Convert decision function to 0-1    
            probability_predictions = sigmoid_array(decision_metrics)
            training_predictions = sigmoid_array(decision_metrics_training)    
            training_metrics = evaluate_binary_model(training_predictions,training_data[test_column].tolist(),model_dict,optimizer)
            model_metrics = evaluate_binary_model(probability_predictions,testing_data[test_column].tolist(),model_dict,optimizer,make_charts)

    model_metrics['train_time'] = train_time - start_time
    model_metrics['test_time'] = end_time - train_time
    model_metrics['training_accuracy'] = training_metrics['testing_accuracy']
    model_metrics['test_indicies'] = testing_data.index.tolist()
    model_metrics['actuals'] = testing_data[test_column].tolist()

    try:
        model_metrics['importances'] = np.array(model.feature_importances_)
        
    except Exception, e:
        try:
            # Only works for binary classifiers
            model_metrics['importances'] = np.array(model.coef_)[0]
            
        except:
            model_metrics['importances'] = np.array(np.zeros(len(train_columns))) 
    
    return model_metrics

# Main function
def full_pipeline(filename,result_col,k_folds=K_FOLDS,optimizer_list=['f1'],verbose=False):
    '''
    Full machine learning pipeline for testing, relevant code for CS123 is pool_k_folds
    '''
    if verbose:
        start_time = time.time()
        print "start pipeline"
    
    dataframe = read_file_into_panda(filename,INDEX_COL)
    # Hardcoded for CS123
    models = {'model':AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=10,max_depth=5,n_jobs=-1),n_estimators=30,learning_rate=.3),'name': "Boosting",'type':'decision'}

    
    #Run each model
    for optimizer in optimizer_list:
        header = ['model'] + [item for item in OUTPUT_HEADER]
        output_array = np.array(header)
        print output_array

        for model_dict in models:
            model_dict['filename'] = filename[:-4]
            if verbose:
                print model_dict['name'], time.time() - start_time
            # Impute in K-Folds
            model_metrics = pool_k_folds(dataframe,optimizer,model_dict['model'],model_dict,result_col,k_folds,verbose=verbose)
            average_line = [model_dict['name']]
            for key in OUTPUT_HEADER:
                average_line.append(model_metrics[key])
   
            if verbose:
                print average_line
            output_array = np.vstack([output_array,np.array(average_line)])
            model_metrics['column_names'] = get_train_columns(dataframe,result_col)
            write_file_name = filename[:-4] + '_' + model_dict['name']
            write_confusion_and_features(model_metrics,write_file_name)
        np.savetxt(filename[:-4] + '_final_output_' + result_col + '_' + optimizer + '.csv',output_array,delimiter=",",fmt = '%s')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python pipeline.py Filename column v (for verbose)"
        exit(1)
      
    if len(sys.argv) > 3 and sys.argv[3] == 'v':
        verbose = True
    else:
        verbose = False
    
    result_col = sys.argv[2]
    filename = sys.argv[1]

    
    full_pipeline(filename,result_col,verbose=verbose)
    
