from mrjob_intersection_combined import IntersectionCount
import sys

OUTPUT_FILE = 'MRreview_Intersection_combined_filtered.csv'

if __name__ == '__main__':
    # Creates an instance of our MRJob subclass
    
    job = IntersectionCount(args=sys.argv[1:])
    with job.make_runner() as runner:
        # Run the job
        runner.run()
        filename = open(OUTPUT_FILE,'w')
        filename.write('biz_1, biz_2, prob_a, prob_b,prob_a_b, confidence, lift\n')


        # Process the output
        for line in runner.stream_output():
            key, value = job.parse_output_line(line)
            filename.write(str(key) + ',' + ",".join([str(x) for x in value]) + '\n')
