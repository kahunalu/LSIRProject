import os
import random
import numpy as np
from PIL import Image, ImageChops

base = './data/'


img_path = []
size = (614, 460)
verifyset=set()

for imageset in os.listdir(base):
	label = 0

	if imageset == "bikeimages":
		label = 0
		verifyset.add(label)
	elif imageset == "catimages":
		label = 1
		verifyset.add(label)
	elif imageset == "carimages":
		label = 2
		verifyset.add(label)
	elif imageset == "shoeimages":
		label = 3
		verifyset.add(label)
	elif imageset == "golfimages":
		label = 4
		verifyset.add(label)
	elif imageset == "strollerimages":
		label = 5
		verifyset.add(label)
	elif imageset == "microwaveimages":
		label = 6
		verifyset.add(label)
	elif imageset == "dogimages":
		label = 7
		verifyset.add(label)
	elif imageset == "keyboardimages":
		label = 8
		verifyset.add(label)
	elif imageset == "guitarimages":
		label = 9
		verifyset.add(label)
	else:
		continue

	for filename in os.listdir(base+imageset):
		img_path.append((label,base+imageset+'/'+filename))

random.shuffle(img_path)

total_count = 0
file_count = 0
error_count = 0

print len(img_path)

images = []
labels = []

for (label,image) in img_path:
	try:
		image = Image.open(image)
		image.thumbnail(size, Image.ANTIALIAS)
		image_size = image.size

		normalized = image.crop( (0, 0, size[0], size[1]) )

		offset_x = max( (size[0] - image_size[0]) / 2, 0 )
		offset_y = max( (size[1] - image_size[1]) / 2, 0 )
		
		normalized = ImageChops.offset(normalized, offset_x, offset_y)

		arr = np.array(normalized)[:460].reshape(847320)

		images.append(arr)
		labels.append(label)

		total_count = total_count+1

		print float(total_count)/len(img_path)
	except:
		print "ERROR #"+str(error_count)
		error_count = error_count + 1
		continue

	if not total_count%2000:

		with open('./dataset/images'+str(file_count)+'.dat', 'wb') as f:
			np.save(f,np.asarray(images))

		with open('./dataset/labels'+str(file_count)+'.dat', 'wb') as f:
			np.save(f,np.asarray(labels))

		file_count = file_count + 1

		images = []
		labels = []

if len(images):
	with open('./dataset/images'+str(file_count)+'.dat', 'wb') as f:
			np.save(f,np.asarray(images))

	with open('./dataset/labels'+str(file_count)+'.dat', 'wb') as f:
		np.save(f,np.asarray(labels))

	file_count = file_count + 1

	images = []
	labels = []