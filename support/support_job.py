from mrjob.job import MRJob
import json

#command line counts number of lines in file
#wc -l <filename>

#command line to output to directory
#python <run_file.py> <input file> -output_dir

class MRsupport(MRJob):

	def mapper(self, _, line):
		line_dict = json.loads(line)
		b_id = line_dict['business_id']
		yield b_id, 1.0

	def combiner(self, key, value):
		yield key, sum(value)

	def reducer_init(self):
		#self.tot_rev = 1569264.0 #file length for yelp_academic_dataset_review.json
		self.tot_rev = 1000.0
	def reducer(self, key, value):
		yield key, sum(value)/self.tot_rev


if __name__ == '__main__':
	MRsupport.run()