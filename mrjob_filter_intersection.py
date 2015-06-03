from mrjob.job import MRJob
from mrjob.step import MRStep
import json
import csv
import heapq
CUT_OFF = 20
REVIEW_CUT_OFF = 8
LIFT_CUT = 0

class FilterIntersections(MRJob):
	
	def configure_options(self):
		super(FilterIntersections, self).configure_options()
		self.add_file_option('--freq_table')
		self.add_passthrough_option('--user_count',type=float,help='Length of user file')
		self.add_passthrough_option('--review_count',type=float,help='Length of review file')


	def final_mapper_init(self):
		
		f = open(self.options.freq_table,'rU')
		self.dictionary = {}
		for line in f:
			items = line.split()
			self.dictionary[items[0].strip('"')] = items[1]
		self.users = self.options.user_count
		self.reviews = self.options.review_count
		self.mapper_heap = []
		heapq.heapify(mapper_heap)
		self.mapper_count = 0
	
	def final_mapper(self,biz_pair,intersection_count):

		businesses = biz_pair.split(DIVIDER)
		count_a = float(self.dictionary[businesses[0]])
		count_b = float(self.dictionary[businesses[1]])
		prob_a = count_a / self.reviews
		prob_b = count_b / self.reviews
		probability_a_b = intersection_count / self.users
		confidence = probability_a_b / prob_a #confidence * prob_a * user = intersection count
		lift = confidence / prob_b
		yield_list = [prob_a,prob_b,probability_a_b,confidence, lift]
		output_tuple = (biz_pair,yield_list)
		try:
			min_lift, min_output_tuple = self.mapper_heap[0]
		except:
			pass
		if (count_a > REVIEW_CUT_OFF) and (count_b > REVIEW_CUT_OFF) and lift > min_lift:
			if self.mapper_count < CUT_OFF:
				heapq.heappush(self.mapper_heap,(lift,output_tuple))
				self.mapper_count += 1
			elif lift > min_lift:
				heapq.heapreplace(self.mapper_heap,(lift,output_tuple))
	
	def mapper_final(self):
		
		self.heap.sort(reverse=True)
		for item in self.heap:
			yield item[1][0], item[1][1]

	def final_combiner(self,biz_pair,line):
		line_list = list(line)
		yield biz_pair, line_list[0]

	def final_reducer_init(self):
		self.heap = []
		heapq.heapify(self.heap)
		self.count = 0

	def final_reducer(self,biz_pair,line):
		line_list = list(line)
		lift = line_list[0][LIFT]
		output_tuple = (biz_pair,line_list[0])
		try:
			min_lift, min_output_tuple = self.heap[0]
		except:
			pass
		if self.count < CUT_OFF:
			heapq.heappush(self.heap,(lift,output_tuple))
			self.count += 1
		elif lift > min_lift:
			heapq.heapreplace(self.heap,(lift,output_tuple))

	def last_reducer_final(self):
		
		self.heap.sort(reverse=True)
		for item in self.heap:
			yield item[1][0], item[1][1]

	def steps(self):
		return [
			MRStep(mapper_init=self.final_mapper_init,mapper=self.final_mapper,combiner=self.final_combiner,reducer_init=self.final_reducer_init,reducer=self.final_reducer,reducer_final=self.last_reducer_final)
			]

