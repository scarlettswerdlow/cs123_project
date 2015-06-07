import pandas as pd
from sklearn.preprocessing import Binarizer
import time
import sys
import csv

MODEL_EVAL_NAME = 3
FILE_TO_INCLUDE = 'data/features_to_include.csv'
INDEX_COL = 'bus_business_id'
CONTINUOUS = ['bus_review_count','bus_stars']
BINS = [-5,-0.1,5]
LABELS = [1,0]
COLS_TO_REMOVE = ['bus_longitude',
 'awater',
 'geoid',
 'geo_id',
 'aland',
 'bc_business_id',
 'bus_location',
 'the_geom',
 'intptlat',
 'blkgrpce',
 'intptlon',
 'namelsad',
 'bus_name',
 'bus_latitude',
 'bus_location',
 'tractce',
 'bus_full_address',
 'statefp',
 'rev_business_id']

def import_test_column(filename,new_col,dataframe):
	'''
	Assumes file is a list of business_ids with a given column to test_list
	'''
	
	f = open(filename)
	data = csv.reader(f)

	biz_ids = np.array([line[0] for line in data])
	biz_ids_df = pd.DataFrame(data=np.ones(len(biz_ids)),index=biz_ids,colums=new_col)
	new_df = dataframe.join(biz_ids_df)
	new_df[new_col].fillna(0,inplace=True)

	return new_df

def get_columns_to_include(filename):
	'''
	Reads colums out of feature sweep output
	'''
	
	f = open(filename,'rU')
	data = csv.reader(f)
	data.next()
	return [line[MODEL_EVAL_NAME] for line in data]



def check_for_true_false(dataframe,column):
	'''
	Converts true/false to 0/1
	'''
	test_list = dataframe[column].unique()
	len_list = len(test_list)
	# Account for NaN
	hasNan = has_nan(test_list)
	
	if (len_list == 3 and hasNan == True) or len_list <= 2:
		# Check if there are two printed items or all true and all false
		if (('t' in test_list) or ('f' in test_list)) or ((1 in test_list) or (0 in test_list)):
			dataframe[column].replace(to_replace='t',value=1,inplace=True)
			dataframe[column].replace(to_replace='f',value=0,inplace=True)
			# False is appropriate here as it means the category doesn't have that as an option
			dataframe[column].fillna(value=0,inplace=True)
			return True

	return False

def check_for_single_name(dataframe,column):
	'''
	Converts fields with only one entry to true/false
	'''
	test_list = dataframe[column].unique()
	len_list = len(test_list)
	hasNan = has_nan(test_list)

	if (len_list == 2 and hasNan) or (len_list == 1 and hasNan == False):
		name = dataframe[column].value_counts(dropna=True).index[0]
		dataframe[column].replace(to_replace=name,value=1,inplace=True)
		dataframe[column].fillna(value=0,inplace=True)	
		return True

	return False	

def all_nan(dataframe,column):
	'''
	Replaces a column with all 0s if all are NaN
	'''
	test_list = dataframe[column].unique()
	len_list = len(test_list)
	hasNan = has_nan(test_list)

	if (len_list == 1 and hasNan):
		dataframe[column].fillna(value=0,inplace=True)
		return True
	else:
		return False

def has_nan(input_list):
	'''
	Checks if NaN is in a list
	'''
	hasNan = False
	for item in input_list:
		if pd.isnull(item):
			hasNan = True
			return hasNan
	return hasNan

def make_binary_and_clean(filename,index_col,result_col,bins=BINS,labels=LABELS):
	'''
	Main function
	'''
	df = pd.read_csv(filename,index_col=index_col)
	categorical = []
	
	for col in COLS_TO_REMOVE:
		try:
			# Error check in case some columns aren't included
			df.drop(col,axis=1,inplace=True)
		except Exception, e:
			pass
		
	else:
		new_filename = sys.argv[1][:-4] + '_filled.csv'

	
	# Update each columns with either T/F, single use or categorical
	names = df.columns.values
	for name in names:
		if check_for_true_false(df,name):
			continue
		elif check_for_single_name(df,name):
			continue
		elif all_nan(df,name):
			continue
		else:
			# Excludes census data from categorical as well as select columns
			if name[:4] == 'pct_' or name[:3] == 'med' or name[:3] == '201' or name[:3] == '200' or (name in CONTINUOUS) or (name in result_col) or (name in COLS_TO_REMOVE):
				continue
			else:
				categorical.append(name)

	new_df = pd.get_dummies(df,columns=categorical)
	
	# Remove rows with Null results
	new_df = new_df[pd.notnull(new_df[result_col])]

	if len(new_df[result_col].unique()) != 2:
		new_df.loc[:,result_col] = pd.cut(new_df[result_col],bins=bins,labels=labels,include_lowest=True)
    
	

	return new_df 

def print_binary_and_clean(filename,index_col,result_col):
	'''
	Saves file for future use
	'''
	new_filename = filename[:-4] + '_filled.csv'
	new_df = make_binary_and_clean(filename,index_col,result_col,bins=BINS,labels=LABELS)
	new_df.to_csv(new_filename)


if __name__ == '__main__':
	start = time.time()
	if len(sys.argv) < 2:
		print "Usage: python make_binary_and_clean.py FILENAME RESULT_COL"
	if len(sys.argv) > 2:
		result_col = sys.argv[2]
	print_binary_and_clean(sys.argv[1],INDEX_COL,result_col)
	print time.time() - start
	



	






