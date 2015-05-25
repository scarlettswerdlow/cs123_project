from mrjob_review import ReviewImport
import sys

OUTPUT_FILE = 'MRreview_count.csv'

if __name__ == '__main__':
    # Creates an instance of our MRJob subclass
    job = ReviewImport(args=sys.argv[1:])
    with job.make_runner() as runner:
        # Run the job
        runner.run()
        filename = open(OUTPUT_FILE,'w')

        # Process the output
        for line in runner.stream_output():
            key, value = job.parse_output_line(line)
            filename.write(str(key) + ',' + str(value) + '\n')
            # filename.write(",".join(value))
            # filename.write('\n')
