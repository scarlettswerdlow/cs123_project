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
			business_dict[business_id] = {'users': []}

		# Append user id to existing list
		business_dict[business_id]['users'].append(reviewer_id)

	return business_dict
