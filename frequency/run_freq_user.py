from freq_user import MRUser
import sys

if __name__ == '__main__':
	job = MRUser(args=sys.argv[1:])
	with job.make_runner() as runner:
		runner.run()

		f = open('f_user_run.txt', 'w')
		for line in runner.stream_output():
			key, value = job.parse_output_line(line)
			f.write(key + ',' + str(value) + '\n')
