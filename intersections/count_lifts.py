import csv
import sys
import heapq
import time

LIFT = 6

if __name__ == '__main__':
	filename = sys.argv[1]
	lift_floor = float(sys.argv[2])
	max_pairs = float(sys.argv[3])
	print "Min. Lift: ", lift_floor, "Limit :",max_pairs
	
	f = open(filename,'rU')
	data = csv.reader(f)
	data.next()
	
	# Create output file with appropriate header
	w = open(filename[:-4] + '_summary.csv','w')
	w.write('biz_1, biz_2, prob_a, prob_b,prob_a_b, confidence, lift\n')

	heap = []
	heapq.heapify(heap)
	count = 0
	line_count = 0
	start_time = time.time()

	for line in data:
		# Count length of file counted
		line_count += 1
		if line_count % 1000000 == 0:
			print "Counted: ",line_count, "Time: ",time.time() - start_time
		lift = float(line[LIFT])
		if lift < lift_floor:
			# Exclude items less than a floor level
			continue
		if count < max_pairs:
			heapq.heappush(heap,(lift,line))
			count += 1
			continue
		else:
			min_lift, min_output_tuple = heap[0]
			if lift > min_lift:
				heapq.heapreplace(heap,(lift,line))
				count +=1

	heap.sort(reverse=True)
	for item in heap:	
		w.write(",".join(item[1]))
		w.write('\n')


	f.close()
	w.close()



