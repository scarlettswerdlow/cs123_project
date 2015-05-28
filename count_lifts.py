import csv
import sys

LIFT = 5
BUCKET = 1000

if __name__ == '__main__':
	filename = sys.argv[1]

	f = open(filename,'rU')
	f.next()
	data = csv.reader(f)
	lift_dict = {}
	w = open('summary_' + filename,'w')
	w.write('Bucket, Number')


	for line in data:
		key = round(float(line[LIFT])/BUCKET)*BUCKET
		lift_dict[key] = lift_dict.get(key,0) + 1
	for key in lift_dict:
		line = str(key) + "," + str(lift_dict[key]+'\n')
		w.write(line)

	f.close()
	w.close()