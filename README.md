# ECE 516 Homework 1

### Source code

The source code for this project mainly consists of 3 files.  The file `ROIDemo.py` was a provided file
but I modified the code so the user could specify command line options for the video to load, the name
of the object which is being "boxed" (i.e. face_1). Also the name of the output file was changed so it
contains the video name and object name so that information could be easily used in subsequent steps
and tracked. This file is run with a command like:

	python ROIdemo.py face_1 videos/BackCameraClips/Eating1Round1.mp4

To speed up the process of creating the ground truth data, I create the `Makefile` which automated the commands
so all I had to run was `make` and then selected the faces in order. The ground truth data files are then saved
to the directory `Output/Data/`.

After the ground truth data is generated, the python code in `generateTrainData.py` is used to read the
ground truth numpy files, read in the videos, generate boxes, calculate ground truth face overlap with
each box, extract image date which correlates to the selected boxes, and then generates 3 (X, y, group) numpy
files for each aspect ratio and scale combination. containing the raw X, y, and group data files. The program
is typicaly run with a command like:

	python generateTrainData.py Output/Data/*

After a few minutes, some very large .numpy files are created.

The next step was to train the models based on the generated data sets and the code for this step is in
`classify.py`. This file contains a class definition for the models as well as code for doing the cross validation
for selecting the best model parameters for each aspect ratio/scale combination.  The output for this step is in
the file `train.out`. I trained the final models on ALL of the data in the X/y numpy files so the results will
come from files that the models or I never saw.

### Training

The final cross validation scores and model parameters are show here:

	Finding best model for scale/aspect 1/1...
	Cross-validation scores: [0.93921569 0.93320236 0.94499018 0.95088409 0.93320236]
	Best parameters: {'C': 100, 'gamma': 0.001}
	Training final classifier....
	Saving classifier...
	Finding best model for scale/aspect 1/2...
	Cross-validation scores: [0.95646259 0.95918367 0.96185286 0.95776567 0.95640327]
	Best parameters: {'C': 100, 'gamma': 0.001}
	Training final classifier....
	Saving classifier...
	Cross-validation scores: [0.96536145 0.98042169 0.98192771 0.98340875 0.97888386]
	Best parameters: {'C': 10000, 'gamma': 0.001}
	Training final classifier....
	Saving classifier...
	Cross-validation scores: [0.96391753 0.97938144 0.98969072 0.96907216 0.98453608]
	Best parameters: {'C': 10000, 'gamma': 5e-05}

I don't really believe the scores...

### Final results

I was getting really poor results, nothing was classified as a face, until I added the `whiten=True`
flag to the PCA object. After that I started to get some okay results. I used the method `predict_proba`
to get probability scores for each box and then only chose boxes with probability greater than `0.95`.

The final results are shown in the following images. All boxes at each scale/aspect ratio are drawn in
grey (so it's a bit messy) but boxes which exceed the threshold are drawn in green. The results for 4 frames
are shown below. All frames can be viewed at https://github.com/crowleydi/ece516-hw1/tree/master/results.

Frame 1
<img src="https://github.com/crowleydi/ece516-hw1/raw/master/results/frame0001.jpeg" width="960" height="540">
Frame 61
<img src="https://github.com/crowleydi/ece516-hw1/raw/master/results/frame0061.jpeg" width="960" height="540">
Frame 121
<img src="https://github.com/crowleydi/ece516-hw1/raw/master/results/frame0121.jpeg" width="960" height="540">
Frame 181
<img src="https://github.com/crowleydi/ece516-hw1/raw/master/results/frame0181.jpeg" width="960" height="540">
