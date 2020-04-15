

import subprocess

from gtts import gTTS

from matplotlib import pyplot as plt
from utils import *

max_val = 12
max_pt = -1
max_kp = 0
import os

orb = cv2.ORB_create()
# orb is an alternative to SIFT


#test_img = read_img('files/test_20_1.jpg')
#test_img = read_img('files/test_20_2.jpg')
#test_img = read_img('files/test_50_1.jpg')
#test_img = read_img('files/test_50_2.jpg')
test_img = read_img('files/test_100_1.jpg')
#test_img = read_img('files/test_100_2.jpg')
#test_img = read_img('files/test_100_4.jpg')
#test_img = read_img('files/test_100backside.jpg')
#test_img = read_img('files/test_100_6.jpg')
#test_img = read_img('files/test_100_25.jpg')
#test_img = read_img('files/test_100_27.jpg') #folded


# resizing must be dynamic
original = resize_img(test_img, 0.4)
display('original', original)

# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/100backside.jpg']

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 12:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected: Rs. ', note)


	
	mytext = ''+note+ 'Rupees'
	language = 'en'
	myobj = gTTS(text=mytext, lang=language)
	myobj.save("rupees.mp3")
	os.system("start rupees.mp3")

	#audio_file = 'audio/' + note + '.mp3'

	#audio_file = "value.mp3"
	#tts = gTTS(text=speech_out, lang="en")
	#tts.save("audio_file")
	#return_code = subprocess.call(["afplay", audio_file])

	(plt.imshow(img3), plt.show())
else:
	print('No Matches')
	mytext = 'Did not match'
	language = 'en'
	myobj = gTTS(text=mytext, lang=language)
	myobj.save("notmatched.mp3")
	os.system("start notmatched.mp3")
