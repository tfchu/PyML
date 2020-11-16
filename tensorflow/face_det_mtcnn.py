'''
Tensorflow + Matplotlib
MTCNN (Multi-Task Cascaded Convolutional Neural Network)
https://github.com/davidsandberg/facenet/
https://arxiv.org/abs/1604.02878
$ pip install mtcnn
'''
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
 
# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()							# Get the current Axes
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.show()
 
filename = '../images/test2.jpg'
# load image from file
pixels = pyplot.imread(filename)			# shape: (1280, 960, 3)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image, e.g. 
# [{'box': [389, 378, 223, 288], 'confidence': 0.9999948740005493, 'keypoints': {'left_eye': (469, 493), 'right_eye': (569, 497), 'nose': (521, 550), 'mouth_left': (467, 587), 'mouth_right': (555, 593)}}]
faces = detector.detect_faces(pixels)

pyplot.imshow(pixels)						# Display an image, i.e. data on a 2D regular raster.
ax = pyplot.gca()
# draw box and keypoints for faces
for face in faces:
	x, y, width, height = face['box']
	rect = Rectangle((x, y), width, height, fill=False, color='red')
	ax.add_patch(rect)
	for key, value in face['keypoints'].items():
		dot = Circle(value, radius=2, color='red')
		ax.add_patch(dot)
pyplot.show()

# show detected face
i = 0
for face in faces:
	x, y, width, height = face['box']
	x2, y2 = x + width, y + height
	pyplot.subplot(1, len(faces), i+1)		# (i+1)-th plot in 1 x len(faces) grid
	pyplot.axis('off')						# after subplot to remove axis
	pyplot.imshow(pixels[y:y2, x:x2])		# display in pyplot
	i += 1
pyplot.show()

# display faces on the original image
# draw_image_with_boxes(filename, faces)