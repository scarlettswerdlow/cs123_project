from mrjob.job import MRJob
from mrjob.step import MRStep
import json
import csv
import heapq

BUSINESS_LOC = 0
DIVIDER = ','
FREQUENCY_FILE = 'frequency/biz_freq.txt'
LIFT = 3
CUT_OFF = 0

class IntersectionCount(MRJob):
  	
	def configure_options(self):
		super(IntersectionCount, self).configure_options()
		self.add_file_option('--freq_table')
		self.add_passthrough_option('--user_count',type=float,help='Length of user file')
		self.add_passthrough_option('--review_count',type=float,help='Length of review file')


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
					yield (businesses[x] + DIVIDER + businesses[y]), 1
	
	def mapper_intersection(self,biz_pair,count):
		yield biz_pair, count

 	def combiner_intersection(self,biz_pair,count):
		yield biz_pair, sum(count)

	def reducer_intersection(self,biz_pair,count):
		yield biz_pair, sum(count)

	def final_mapper_init(self):
		
		f = open(self.options.freq_table,'rU')
		self.dictionary = {}
		for line in f:
			items = line.split()
			self.dictionary[items[0].strip('"')] = items[1]
		self.users = self.options.user_count
		self.reviews = self.options.review_count
		


	def final_mapper(self,biz_pair,intersection_count):

		businesses = biz_pair.split(DIVIDER)
		count_a = float(self.dictionary[businesses[0]])
		count_b = float(self.dictionary[businesses[1]])
		prob_a = count_a / self.users
		prob_b = count_b / self.reviews
		probability_a_b = intersection_count / self.users
		confidence = probability_a_b / prob_a
		lift = confidence / prob_b
		yield biz_pair, [prob_a,prob_b,confidence, lift]

	def final_reducer_init(self):
		self.heap = []
		heapq.heapify(self.heap)

	def final_reducer(self,biz_pair,line):
		line_list = list(line)
		lift = line_list[0][LIFT]
		output_tuple = (biz_pair,line_list[0])
		print output_tuple

		if lift > CUT_OFF:
			heapq.heappush(self.heap,(lift,output_tuple))

	def last_reducer_final(self):
		
		for item in self.heap:
			yield item[1][0], item[1][1]


	def steps(self):
		return [
			MRStep(mapper=self.mapper_review,combiner=self.combiner_review,reducer=self.reducer_review),
			MRStep(mapper=self.mapper_intersection,combiner=self.combiner_intersection,reducer=self.reducer_intersection),
			MRStep(mapper_init=self.final_mapper_init,mapper=self.final_mapper,reducer_init=self.final_reducer_init,reducer=self.final_reducer,reducer_final=self.last_reducer_final)
			]


if __name__ == '__main__':
	IntersectionCount.run()




