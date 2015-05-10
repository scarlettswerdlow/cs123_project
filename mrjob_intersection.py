from mrjob.job import MRJob
from mrjob.step import MRStep
import json

BUSINESS_LOC = 0
DIVIDER = '&&&'

class IntersectionCount(MRJob):
  
	def mapper_review(self, _, line):
		line_dict = json.loads(line)
		user_id = line_dict['user_id']
		biz_id = line_dict['business_id']
		yield user_id, biz_id

	def combiner_review(self, user_id, biz_id):
		unique_biz = set(list(biz_id))
		for item in unique_biz:
			yield user_id, item
      
	def reducer_review(self, name, biz_id):
   	
		businesses = list(biz_id)
		biz_length = len(businesses)
		# Loop over unique combinations
		for x in range(biz_length):
			# Avoid duplicates
			for y in range(biz_length):
				if (x != y):
					yield (businesses[x] + '&&&' + businesses[y]), 1
	def mapper_intersection(self,biz_pair,count):
		yield biz_pair, count

 	def combiner_intersection(self,biz_pair,count):
		yield biz_pair, sum(count)

	def reducer_intersection(self,biz_pair,count):
		yield biz_pair, sum(count)

	def steps(self):
		return [
			MRStep(mapper=self.mapper_review,combiner=self.combiner_review,reducer=self.reducer_review),
			MRStep(mapper=self.mapper_intersection,combiner=self.combiner_intersection,reducer=self.reducer_intersection)
			]
if __name__ == '__main__':
	IntersectionCount.run()




