# Creates reviews/, a directory where each file represents
# a restaurant and contains all reviews of that restaurant.

import json
import pandas as pd
import pickle

# business_id_to_category = {}

# with open('dataset/business.json') as business_file:
# 	for line in business_file:
# 		business = json.loads(line)
# 		if 'Restaurants' not in set(business['categories']):
# 			# print('Skipping ' + str(business['categories']))
# 			continue
# 		business_id = business['business_id']
# 		open('dataset/reviews/' + str(business_id) + '.json', 'w')
# 		business_id_to_category[business_id] = business['categories']

# json.dump(business_id_to_category, open('labels.json', 'w'))

with open('labels.json', 'r') as fp:
    business_id_to_category = json.load(fp)
# input(business_id_to_category)

with open('dataset/review.json') as review_file:
	for line in review_file:
		try:
			review = json.loads(line)
			business_id = review['business_id']
			review['categories'] = business_id_to_category[business_id]
			with open('dataset/reviews/' + str(business_id) + '.json', 'w') as business_file:
				json.dump(review, business_file)
				print('Success!')
		except:
			print(business_id)
