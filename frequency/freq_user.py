from mrjob.job import MRJob
import json

class MRUser(MRJob):

	def mapper(self, _, line):
		line_dict = json.loads(line)
		b_id = line_dict['business_id']
		u_id = line_dict['user_id']
		yield b_id, u_id

	def combiner(self, key, value):
		yield key, len(set(value))

	def reducer(self, key, val):
		yield key, sum(val)


if __name__ == '__main__':
	MRUser.run()