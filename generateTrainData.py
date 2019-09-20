import cv2
import sys
import os
import numpy as np
import random

# unit box is 120x120
BoxUnitWidth = 120
AnchorPointSpacing = 60
ImgX = 1920
ImgY = 1080
Scales=[1, 1, 2, 2]
AspectRatios=[1, 2, 1, 2]
SkipFrames = 10
MinThreshold = 0.35

# read all of the args. args should be filenames of the ground truth
# data generated by ROIdemo.py.  These filenames must be numpy arrays
# and named with the format vidfilename_objname.npy
VideoGroundTruth = {}
for arg in sys.argv[1:]:
	gtarr = np.load(arg, allow_pickle=True)
	gttup = (int(gtarr[0]),int(gtarr[1]),int(gtarr[2]),int(gtarr[3]))
	fileName = os.path.split(arg[:arg.find('_')])[1]
	if fileName in VideoGroundTruth:
		VideoGroundTruth[fileName].append(gttup)
	else:
		VideoGroundTruth[fileName] = [gttup]
	img = cv2.imread("Output/FirstFrame/"+fileName+".jpg",cv2.IMREAD_GRAYSCALE)
	#cv2.rectangle(img,
	#cv2.imshow(fileName,img)
	#k = cv2.waitKey(100)


# Each video frame is 1920x1080 so I'm trying to break up each frame
# So I can have as many non-overlapping boxes that are 128x128 as
# possible.
# 1920/128 = 15
# 1080/128 ~= 8 so there will be some gaps. I'm ok with that.

# Generate all anchor points
def GenerateAnchorPoints():
	# calculate anchor point spacing along x
	nx = int(ImgX/AnchorPointSpacing)
	# calculate anchor point spacing along y
	ny = int(ImgY/AnchorPointSpacing)
	print("nx="+str(nx))
	print("ny="+str(ny))
	# calculate all of the X and Y anchor points
	apxs = np.array(range(0,nx))*(ImgX/nx)+AnchorPointSpacing/2
	apys = np.array(range(0,ny))*(ImgY/ny)+AnchorPointSpacing/2
	# combine all of the x/y combinations into a list
	aps=[]
	for apx in apxs:
		for apy in apys:
			tup = (int(apx),int(apy))
			aps.append(tup)
	return aps

# Generate boxes given the anchor points and
# a given scale and aspect ratio
def GenerateBoxes(anchors, scale, aspectRatio):
	boxes = []
	dx = (scale * BoxUnitWidth)
	dy = (dx * aspectRatio)
	for ap in anchors:
		# "center" of the box
		# the center is calculated such that it is shifted
		# away from the edges so boxes don't go off the edge.
		cx = min(max(dx,ap[0]),ImgX-dx)
		cy = min(max(dy,ap[1]),ImgY-dy)
		# upper left of thebox
		ux = cx-dx/2
		uy = cy-dy/2
		# box tuple contains(upper_left_x, upper_left_y, width, height)
		tup=(int(ux),int(uy),int(dx),int(dy))
		boxes.append(tup)
	return boxes

def CalcOverlap(face, box):
	# extract face coords
	fx1 = face[0]
	fx2 = face[0]+face[2]
	fy1 = face[1]
	fy2 = face[1]+face[3]
	# calc area of the face
	fa = face[2]*face[3]

	# extract box coords
	bx1 = box[0]
	bx2 = box[0]+box[2]
	by1 = box[1]
	by2 = box[1]+box[3]
	# calc area of the box
	ba = box[2]*box[3]

	# calc area of the intersection
	ia = max(0,min(fx2,bx2)-max(fx1,bx1))*max(0,min(fy2,by2)-max(fy1,by1))
	# calc IoU
	ratio = float(ia)/float(ba+fa-ia)
	return ratio

def BuildTrainBoxes(faces, boxes):
	boxes = list(boxes) # copy the boxes list
	faceBoxes = []
	# for each face...
	for face in faces:
		best_box = 0
		best_th = 0
		for box in boxes:
			# calculate the IoU
			th = CalcOverlap(face, box)
			# keep best box
			if th > best_th:
				best_box = box
				best_th = th
		# keep the best box which overlaps the face
		# and exceeds threshold
		if best_th > MinThreshold:
			print("found facebox threshold: " + str(best_th))
			faceBoxes.append(best_box)

	# remove the face boxes from boxes list
	for faceBox in faceBoxes:
		boxes.remove(faceBox)

	return faceBoxes, boxes

def ExtractBox(img, box):
	sub = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
	return sub

print("Calculating total frames...")
totalFrames = 0
for videoName in VideoGroundTruth:
	cap = cv2.VideoCapture("videos/BackCameraClips/" + videoName + ".mp4")
	totalFrames = totalFrames +  int((cap.get(cv2.CAP_PROP_FRAME_COUNT)+SkipFrames-1)/SkipFrames)

print("Using " + str(totalFrames) + " frames.")

aps = GenerateAnchorPoints()
for scale, aspectRatio in zip(Scales, AspectRatios):
	boxWidth = BoxUnitWidth*scale;
	boxHeight = boxWidth*aspectRatio
	datawidth = boxWidth*boxHeight
	#
	# Estimate the number of data items as 5000
	estimate = 5000
	y = np.zeros((estimate,))
	groups = np.zeros((estimate,))
	X = np.zeros((estimate,datawidth))
	dataNo = 0
	imgNo = 0
	boxes = GenerateBoxes(aps, scale, aspectRatio)
	for videoName in VideoGroundTruth:
		vf = 0
		cap = cv2.VideoCapture("videos/BackCameraClips/" + videoName + ".mp4")
		faces = VideoGroundTruth[videoName]
		faceBoxes, otherBoxes = BuildTrainBoxes(faces,boxes)
		while True:
			ret, frame = cap.read()
			if ret == False:
				break
			imgNo = imgNo + 1
			# Convert to grayscale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			if len(y) < dataNo + 2*len(faceBoxes):
				print("resizing arrays to " + str(len(y)*2))
				y = np.resize(y,(len(y)*2,))
				groups = np.resize(groups,(len(y)*2,))
				X = np.resize(X,(len(y)*2,datawidth))

			# Extract faceboxes data
			for face in faceBoxes:
				data = ExtractBox(gray, face)
				y[dataNo] = 1
				groups[dataNo] = imgNo
				X[dataNo,:] = data.flatten()
				dataNo = dataNo + 1

			# shuffle the other boxes so we always grab random boxes
			random.shuffle(otherBoxes)

			# Extract other boxes data
			# get the same number as faces
			for i in range(0,len(faceBoxes)):
				data = ExtractBox(gray, otherBoxes[i])
				y[dataNo] = 0
				groups[dataNo] = imgNo
				X[dataNo,:] = data.flatten()
				dataNo = dataNo + 1

			if vf == 0:
				# draw the original GT face boxes
				for face in faces:
					cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(128,0,0),3)
				# draw the best matching face boxes
				for face in faceBoxes:
					cv2.rectangle(frame,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(0,0,0),3)
				# draw the randomly selected boxes
				for i in range(0,len(faceBoxes)):
					ob = otherBoxes[i]
					cv2.rectangle(frame,(ob[0],ob[1]),(ob[0]+ob[2],ob[1]+ob[3]),(255,0,0),3)
				#cv2.imshow("frame 0", frame)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()

			# skip some frames
			for i in range(1,SkipFrames):
				ret, frame = cap.read()
			vf = vf + 1

	print("resizing from "+str(estimate)+" to "+str(dataNo))
	y = np.resize(y,(dataNo,))
	groups = np.resize(groups,(dataNo,))
	X = np.resize(X,(dataNo,datawidth))

	print("writing data files...")
	np.save("y_{}_{}.npy".format(scale,aspectRatio), y)
	np.save("groups_{}_{}.npy".format(scale,aspectRatio), groups)
	np.save("X_{}_{}.npy".format(scale,aspectRatio), X)

print("done")
