import csv
import cv2
lines = []
with open('./data/data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    line.append(line)

print(len(lines))
print('loaed file')

images = []
measurements = []

for line in lines:
  source_path = line[0]
  filename = sourcepath.split('/') [-1]
  current_path = './dara/data/driving_log.csv' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurement.append(measurement)
print(len(images))
print('success')
