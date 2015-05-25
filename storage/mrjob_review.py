from mrjob.job import MRJob
import json

OUTPUT_FILE = 'MRreview_count.csv'

class ReviewImport(MRJob):
  
  def mapper(self, _, line):
    line_dict = json.loads(line)
    user_id = line_dict['user_id']
    biz_id = line_dict['business_id']
    yield user_id, biz_id

  def combiner(self, user_id, biz_id):
    unique_biz = set(list(biz_id))
    for item in unique_biz:
      yield user_id, item
      
  def reducer_init(self):
  	self.review_count = 0

  def reducer(self, name, biz_id):
   	
    businesses = list(biz_id)
    biz_length = len(businesses)
    # Loop over unique combinations
    for x in range(biz_length):
		# Avoid duplicates
		for y in range(biz_length):
			if (x != y):
				yield (businesses[x] + '&&&' + businesses[y]), 1



if __name__ == '__main__':
  ReviewImport.run()