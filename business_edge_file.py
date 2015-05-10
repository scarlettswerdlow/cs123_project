import sys
import load_files

OUTPUT_FILE = 'business_edge_table.csv'

def build_business_dict(reviews_dict):
	'''
	Builds dictionary that maps business id to list of reviewer user ids
	Args:
		* reviews_dict (dictionary) - Dictionary of reviews built from load_files_v2.py
	Returns:
		* Dictionary that maps business id to list of reviewer user ids
	'''
	business_dict = {}

	# Iterate through each review key
	for key in reviews_dict:
		business_id = reviews_dict[key]['business_id']
		reviewer_id = reviews_dict[key]['user_id']

		# If business not already in dictionary, add it
		if business_id not in business_dict.keys():
			business_dict[business_id] = []

		# Append user id to existing list
		business_dict[business_id].append(reviewer_id)

	return business_dict

def biz_reviews_to_table(dictionary):
	'''
	Converts dictionary keyed with businesses with lists of reviewers
	'''

	filename = open(OUTPUT_FILE,'w')
	filename.write('business_id,user_1,user_2\n')
	
	business_list = dictionary.keys()

	for business in business_list:
		user_list = dictionary[business]
		user_set = set(user_list)
		write_list_to_file(filename, list(user_set),business)
	
	filename.close()

def write_list_to_file(filename, user_list, business_id):
	'''
	Converts a list of users to a file
	'''

	list_length = len(user_list)
	
	# Loop over unique combinations
	for x in range(list_length):
		# Avoid duplicates
		for y in range((x+1),list_length):
			filename.write(str(business_id) + ',' + str(user_list[x]) + ',' + str(user_list[y]) + '\n')

def edges_from_raw_dictionary(reviews_dict):
	business_dict = build_business_dict(reviews_dict)
	biz_reviews_to_table(business_dict)

if __name__ == '__main__':
	pass
	if len(sys.argv) != 2:
		print "Usage: business_dict.py REVIEWS_FILENAME"
	else:
		reviews_filename = sys.argv[1]
		review_dict = load_files.load_string_json_file(reviews_filename,'review_id')
		edges_from_raw_dictionary(review_dict)




