from support_job import MRsupport
import sys

if __name__ == '__main__':
	job = MRsupport(args=sys.argv[1:])
	with job.make_runner() as runner:
		runner.run()

		for line in runner.stream_output():
			key, value = job.parse_output_line(line)
			print 'biz_id: ', key, '  P(biz_id): ', value
