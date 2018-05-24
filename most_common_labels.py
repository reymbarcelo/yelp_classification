# Generates 'common_labels.txt', a list of labels 
# ordered by the number of reviews that contain them.

from collections import defaultdict
import json
from operator import itemgetter

labels_to_count = defaultdict(int)

with open('labels.json') as label_file:
	for line in label_file:
		labels = json.loads(line)
		for (business_id, categories) in labels.items():
			for category in categories:
				labels_to_count[category] += 1
for k, v in sorted(labels_to_count.items(), key=itemgetter(1), reverse=True):
	print(k, v)