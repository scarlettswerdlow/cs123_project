from mrjob.job import MRJob
import json

#command line counts number of lines in file
#wc -l <filename>

#command line to output to directory
#python <run_file.py> <input file> -output_dir

class MRFreq(MRJob):

	def mapper(self, _, line):
		line_dict = json.loads(line)
		b_id = line_dict['business_id']
		yield b_id, 1.0

	def combiner(self, key, value):
		yield key, sum(value)

	def reducer(self, key, value):
		yield key, sum(value)


if __name__ == '__main__':
	MRsupport.run()