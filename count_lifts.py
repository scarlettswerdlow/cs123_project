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
	f.next()
	data = csv.reader(f)
	
	w = open(filename[:-4] + '_summary.csv','w')
	w.write('biz_1, biz_2, prob_a, prob_b,prob_a_b, confidence, lift\n')

	heap = []
	heapq.heapify(heap)
	count = 0
	start_time = time.time()

	for line in data:
		if count % 1000000 == 0:
			print "Counted: ",count, "Time: ",time.time() - start_time
		lift = float(line[LIFT])
		if lift < lift_floor:
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



