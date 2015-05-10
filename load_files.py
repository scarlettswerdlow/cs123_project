import json

# BUSINESSES = ('yelp_academic_dataset_business.json','business_id')
# CHECK_IN = ('yelp_academic_dataset_checkin.json','business_id')
# REVIEW = ('yelp_academic_dataset_review.json','review_id')
# TIP = ('yelp_academic_dataset_tip.json','user_id')
# USER = ('yelp_academic_dataset_user.json','user_id')

def load_string_json_file(filename,key_name):
	'''
	Loads json file with multiple lines and specified key
	'''

	f = open(filename,'rU')

	final_dict = {}
	duplicate_counter = 0
	for line in f:
		line_dict = json.loads(line)

		key = line_dict[key_name]
		if key in final_dict:
			final_dict[duplicate_counter] = line_dict
			duplicate_counter += 1
		else:
			final_dict[key] = line_dict

	return final_dict

def get_subkey_names(dictionary):
	'''
	Returns dictionary of all subkeys used in dataset
	'''
	try:
		initial_keys = dictionary.keys()
	except Exception, e:
		return 'Requires Dictionary'

	final_dict = {}

	for key in initial_keys:
		line_dict = get_all_subkeys_one_line(dictionary[key])
		write_subkey_dict_into_overall(line_dict,final_dict)

	return final_dict

def write_subkey_dict_into_overall(line_dict, overall_dict):
	'''
	Recursively traverses tree to get all keys used
	'''

	try:
		keys = line_dict.keys()
	except Exception, e:
		return

	# Adds keys and accesses subdictionaries
	for key in keys:
		if key not in overall_dict:
			overall_dict[key] = {}
		write_subkey_dict_into_overall(line_dict[key],overall_dict[key])
		
def get_all_subkeys_one_line(dictionary):
	'''
	Recursively returns all nested keys in a dictionary
	'''
	try:
		keys = dictionary.keys()
	except Exception, e:
		return 0

	key_dict = {}

	for key in keys:
		subkey_dict = get_all_subkeys_one_line(dictionary[key])
		key_dict[key] = subkey_dict

	return key_dict

def get_all_subkeys_one_list(dictionary):
	'''
	Recursively returns all nested keys in a dictionary
	'''
	
	keys = dictionary.keys()

	key_name = []

	for key in keys:
		subkey_list = get_all_subkeys_one_list(dictionary[key])
		#print subkey_list
		if subkey_list != []:
			for x in subkey_list:
				key_name.append(key + "_" + x)
				#print key_name
		else:
			key_name.append(key)
	return key_name

def count_distinct(dictionary, key_to_count):

	rd = {}
	for key in dictionary.keys():
		counted_id = dictionary[key][key_to_count]
		rd[counted_id] = rd.get(counted_id, 0) + 1

	return rd



def load_dictionary(filename, key):
	raw_dictionary = load_string_json_file(filename, key)
	item_dictionary = get_subkey_names(raw_dictionary)
	return raw_dictionary, item_dictionary

if __name__ == '__main__':
	pass





