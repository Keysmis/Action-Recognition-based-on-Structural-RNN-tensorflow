The files are organized as
joint_positions/[class_name]/[video_name]/joint_positions.mat
Each mat file contains four variables
(1) viewpoint: a string, possible values are E ENE ESE N NE NNE NNW NW S SE SSE SSW SW W WNW WSW
(2) scale: an array whose length equals to the number of frames in the video. the i-th entry is the scale of the person in the i-th frame
(3) pos_img: a 3D matrix with size 2 x 15 x [the number of frames].
In the first dimension, the first and second value correspond to the x and y coordinate, respectively.
In the second dimension, the values are
0: neck			0-1	<- indices in vector (x1,y1,x2,..)
1: belly			2-3
2: face			4-5
3: right shoulder	6-7
4: left  shoulder	8-9
5: right hip		10-11
6: left  hip		12-13
7: right elbow		14-15
8: left elbow		...
9: right knee
10: left knee
11: right wrist
12: left wrist
13: right ankle
14: left ankle

1. See a sample annotation of the 15 positions at http://jhmdb.is.tue.mpg.de/puppet_tool
2. Due to the nature of the puppet annotation tool, all 15 joint positions are available even if they are not annotated when they are occluded or outiside the frame.
In this case, the joints are in the neutral puppet positions.
3. The right and left correspond to the right and left side of the annotated person. For example, a person facing the camera has his right side on the left side of the image, and a person back-facing the camera has his right side on the right side of the image.

(4) pos_world is the normalization of pos_img with respect to the frame size and puppet scale,�
the formula is as below

pos_world(1,:,:) = (pos_img(1,:,:)/W-0.5)*W/H./scale;
pos_world(2,:,:) = (pos_img(2,:,:)/H-0.5)./scale;

W and H are the width and height of the frame, respectively.
