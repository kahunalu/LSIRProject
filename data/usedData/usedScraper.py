#Used to pull down ~150,000 images and associate with labels
import io
import sys
import json
import urllib
import pickle
import numpy as np
from PIL import Image

dataset_file = "dataset.dat"

with open('./dataset/all_data0.txt') as data_file:    
    data = json.load(data_file)

#bikes
bike_count = 0
bike_categories = [
	'road-bikes',
	'mountain-bikes',
	'cruising-bikes',
	'childrens-bikes',
]

#cats
cat_count = 0
cat_categories = [
	'cats'
]

#strollers
stroller_count = 0
stroller_categories = [
	'kids-strollers'
]

#guitars
guitar_count = 0
guitar_categories = [
	'strings-guitar'
]

#keyboards
keyboard_count = 0
keyboard_categories = [
	'pianos-organs-keyboards'
]

#cars
car_count = 0
car_categories = [
	'passenger-cars',
	'sports-cars',
]

#shoes
shoe_count = 0
shoe_categories = [
	'womens-shoes',
	'mens-shoes',
]

#golf
golf_count = 0
golf_categories = [
	'sports-golf',
]

#dogs
dog_count = 0 
dog_categories = [
	'dogs',
]

#microwaves
microwave_count = 0
microwave_categories = [
	'microwaves',
]

file_count = 0
total_count = 0
images = []
labels = []

for i in range(0, 6):
	with open('./dataset/all_data'+str(i)+'.txt') as data_file:    
		data = json.load(data_file)
	
	if total_count > 15000:
		break

	for listing in data:
		category = listing.get('category')


		if not listing.get('photos'):
			continue

		if category in bike_categories:
			if bike_count > 1500:
				continue
			bike_count = bike_count + len(listing.get('photos'))
			category = 0
		elif category in cat_categories:
			if cat_count > 1500:
				continue
			cat_count = cat_count + len(listing.get('photos'))
			category = 1
		elif category in car_categories:
			if car_count > 1500:
				continue
			car_count = car_count + len(listing.get('photos'))
			category = 2
		elif category in shoe_categories:
			if shoe_count > 1500:
				continue
			shoe_count = shoe_count + len(listing.get('photos'))
			category = 3
		elif category in golf_categories:
			if golf_count > 1500:
				continue
			golf_count = golf_count + len(listing.get('photos'))
			category = 4
		elif category in stroller_categories:
			if stroller_count > 1500:
				continue
			stroller_count = stroller_count + len(listing.get('photos'))
			category = 5
		elif category in microwave_categories:
			if microwave_count > 1500:
				continue
			microwave_count = microwave_count + len(listing.get('photos'))
			category = 6
		elif category in dog_categories:
			if dog_count > 1500:
				continue
			dog_count = dog_count + len(listing.get('photos'))
			category = 7
		elif category in keyboard_categories:
			if keyboard_count > 1500:
				continue
			keyboard_count = keyboard_count + len(listing.get('photos'))
			category = 8
		elif category in guitar_categories:
			if guitar_count > 1500:
				continue
			guitar_count = guitar_count + len(listing.get('photos'))
			category = 9
		else:
			continue


		for photo_id in listing.get('photos'):
			try:
				f = urllib.urlopen("https://s3-us-west-2.amazonaws.com/usedphotosna/"+str(photo_id)+"_614.jpg").read()
				img = Image.open(io.BytesIO(f))
				arr = np.array(img)[:460].reshape(847320)

				total_count = total_count + 1

				labels.append(category)
				images.append(arr)

				if not total_count%2000:

					with open('./data/images'+str(file_count)+'.dat', 'wb') as f:
						np.save(f,np.asarray(images))

					with open('./data/labels'+str(file_count)+'.dat', 'wb') as f:
						np.save(f,np.asarray(labels))

					file_count = file_count + 1

					labels = []
					images = []

				print total_count
			except:
				print "https://s3-us-west-2.amazonaws.com/usedphotosna/"+str(photo_id)+"_614.jpg"
				continue