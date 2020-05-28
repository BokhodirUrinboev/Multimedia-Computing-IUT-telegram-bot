import numpy as np
import cv2
import imutils
import pytesseract
from PIL import Image
import shutil
import dlib
from imutils import face_utils
from imutils import paths
import sys
import os

######################################
config ={
	"face_detector_prototxt": "assets/deploy.prototxt",
	"face_detector_weights": "assets/res10_300x300_ssd_iter_140000.caffemodel",
	"landmark_predictor": "assets/shape_predictor_68_face_landmarks.dat",
	"sunglasses": "assets/sunglasses.png",
	"sunglasses_mask": "assets/sunglasses_mask.png",
	"deal_with_it": "assets/deal_with_it.png",
	"deal_with_it_mask": "assets/deal_with_it_mask.png",
	"min_confidence": 0.5,
	"steps": 20,
	"delay": 5,
	"final_delay": 250,
	"loop": 0,
	"temp_dir": "temp"
}


######################################

def overlay_image(bg, fg, fgMask, coords):
	# grab the foreground spatial dimensions (width and height),
	# then unpack the coordinates tuple (i.e., where in the image
	# the foreground will be placed)
	(sH, sW) = fg.shape[:2]
	(x, y) = coords

	# the overlay should be the same width and height as the input
	# image and be totally blank *except* for the foreground which
	# we add to the overlay via array slicing
	overlay = np.zeros(bg.shape, dtype="uint8")
	overlay[y:y + sH, x:x + sW] = fg

	# the alpha channel, which controls *where* and *how much*
	# transparency a given region has, should also be the same
	# width and height as our input image, but will contain only
	# our foreground mask
	alpha = np.zeros(bg.shape[:2], dtype="uint8")
	alpha[y:y + sH, x:x + sW] = fgMask
	alpha = np.dstack([alpha] * 3)

	# perform alpha blending to merge the foreground, background,
	# and alpha channel together
	output = alpha_blend(overlay, bg, alpha)

	# return the output image
	return output

def alpha_blend(fg, bg, alpha):
	# convert the foreground, background, and alpha layers from
	# unsigned 8-bit integers to floats, making sure to scale the
	# alpha layer to the range [0, 1]
	fg = fg.astype("float")
	bg = bg.astype("float")
	alpha = alpha.astype("float") / 255

	# perform alpha blending
	fg = cv2.multiply(alpha, fg)
	bg = cv2.multiply(1 - alpha, bg)

	# add the foreground and background to obtain the final output
	# image
	output = cv2.add(fg, bg)

	# return the output image
	return output.astype("uint8")

def create_gif(inputPath, outputPath, delay, finalDelay, loop):
	# grab all image paths in the input directory
	imagePaths = sorted(list(paths.list_images(inputPath)))

	# remove the last image path in the list
	lastPath = imagePaths[-1]
	imagePaths = imagePaths[:-1]

	# construct the image magick 'convert' command that will be used
	# generate our output GIF, giving a larger delay to the final
	# frame (if so desired)
	cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
		delay, " ".join(imagePaths), finalDelay, lastPath, loop,
		outputPath)
	os.system(cmd)






def deal_with_it(img, output_dir):
		# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-c", "--config", required=True,
	# 	help="path to configuration file")
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# ap.add_argument("-o", "--output", required=True,
	# 	help="path to output GIF")
	# args = vars(ap.parse_args())

	# load the JSON configuration file and the "Deal With It" sunglasses
	# and associated mask
	# config = json.loads(open(config).read())   #######################
	global config
	sg = cv2.imread(config["sunglasses"])
	sgMask = cv2.imread(config["sunglasses_mask"])

	# delete any existing temporary directory (if it exists) and then
	# create a new, empty directory where we'll store each individual
	# frame in the GIF
	shutil.rmtree(config["temp_dir"], ignore_errors=True)
	os.makedirs(config["temp_dir"])

	# load our OpenCV face detector and dlib facial landmark predictor
	print("[INFO] loading models...")
	detector = cv2.dnn.readNetFromCaffe(config["face_detector_prototxt"],
		config["face_detector_weights"])
	predictor = dlib.shape_predictor(config["landmark_predictor"])

	# load the input image and construct an input blob from the image
	image = cv2.imread(img)  ########################################################
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections
	print("[INFO] computing object detections...")
	detector.setInput(blob)
	detections = detector.forward()

	# we'll assume there is only one face we'll be applying the "Deal
	# With It" sunglasses to so let's find the detection with the largest
	# probability
	i = np.argmax(detections[0, 0, :, 2])
	confidence = detections[0, 0, i, 2]

	# filter out weak detections
	if confidence < config["min_confidence"]:
		print("[INFO] no reliable faces found")
		return False

	# compute the (x, y)-coordinates of the bounding box for the face
	box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
	(startX, startY, endX, endY) = box.astype("int")

	# construct a dlib rectangle object from our bounding box coordinates
	# and then determine the facial landmarks for the face region
	rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
	shape = predictor(image, rect)
	shape = face_utils.shape_to_np(shape)

	# grab the indexes of the facial landmarks for the left and right
	# eye, respectively, then extract (x, y)-coordinates for each eye
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	leftEyePts = shape[lStart:lEnd]
	rightEyePts = shape[rStart:rEnd]

	# compute the center of mass for each eye
	leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
	rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

	# compute the angle between the eye centroids
	dY = rightEyeCenter[1] - leftEyeCenter[1]
	dX = rightEyeCenter[0] - leftEyeCenter[0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180

	# rotate the sunglasses image by our computed angle, ensuring the
	# sunglasses will align with how the head is tilted
	sg = imutils.rotate_bound(sg, angle)

	# the sunglasses shouldn't be the *entire* width of the face and
	# ideally should just cover the eyes -- here we'll do a quick
	# approximation and use 90% of the face width for the sunglasses
	# width
	sgW = int((endX - startX) * 0.9)
	sg = imutils.resize(sg, width=sgW)

	# our sunglasses contain transparency (the bottom parts, underneath
	# the lenses and nose) so in order to achieve that transparency in
	# the output image we need a mask which we'll use in conjunction with
	# alpha blending to obtain the desired result -- here we're binarizing
	# our mask and performing the same image processing operations as
	# above
	sgMask = cv2.cvtColor(sgMask, cv2.COLOR_BGR2GRAY)
	sgMask = cv2.threshold(sgMask, 0, 255, cv2.THRESH_BINARY)[1]
	sgMask = imutils.rotate_bound(sgMask, angle)
	sgMask = imutils.resize(sgMask, width=sgW, inter=cv2.INTER_NEAREST)

	# our sunglasses will drop down from the top of the frame so let's
	# define N equally spaced steps between the top of the frame and the
	# desired end location
	steps = np.linspace(0, rightEyeCenter[1], config["steps"],
		dtype="int")

	# start looping over the steps
	for (i, y) in enumerate(steps):
		# compute our translation values to move the sunglasses both
		# slighty to the left and slightly up -- the reason why we are
		# doing this is so the sunglasses don't *start* directly at
		# the center of our eye, translation helps us shift the
		# sunglasses to adequately cover our entire eyes (otherwise
		# what good are sunglasses!)
		shiftX = int(sg.shape[1] * 0.25)
		shiftY = int(sg.shape[0] * 0.35)
		y = max(0, y - shiftY)

		# add the sunglasses to the image
		output = overlay_image(image, sg, sgMask,
			(rightEyeCenter[0] - shiftX, y))

		# if this is the final step then we need to add the "DEAL WITH
		# IT" text to the bottom of the frame
		if i == len(steps) - 1:
			# load both the "DEAL WITH IT" image and mask from disk,
			# ensuring we threshold the mask as we did for the sunglasses
			dwi = cv2.imread(config["deal_with_it"])
			dwiMask = cv2.imread(config["deal_with_it_mask"])
			dwiMask = cv2.cvtColor(dwiMask, cv2.COLOR_BGR2GRAY)
			dwiMask = cv2.threshold(dwiMask, 0, 255,
				cv2.THRESH_BINARY)[1]

			# resize both the text image and mask to be 80% the width of
			# the output image
			oW = int(W * 0.8)
			dwi = imutils.resize(dwi, width=oW)
			dwiMask = imutils.resize(dwiMask, width=oW,
				inter=cv2.INTER_NEAREST)

			# compute the coordinates of where the text will go on the
			# output image and then add the text to the image
			oX = int(W * 0.1)
			oY = int(H * 0.8)
			output = overlay_image(output, dwi, dwiMask, (oX, oY))

		# write the output image to our temporary directory
		p = os.path.sep.join([config["temp_dir"], "{}.jpg".format(
			str(i).zfill(8))])
		cv2.imwrite(p, output)

	# now that all of our frames have been written to disk we can finally
	# create our output GIF image
	print("[INFO] creating GIF...")
	create_gif(config["temp_dir"], output_dir, config["delay"], ######################################
		config["final_delay"], config["loop"])

	# cleanup by deleting our temporary directory
	print("[INFO] cleaning up...")
	shutil.rmtree(config["temp_dir"], ignore_errors=True)

	return True












def vignette_filter_photo(img):
	rows, cols = img.shape[:2]
	# generating vignette mask using Gaussian kernels
	kernel_x = cv2.getGaussianKernel(cols, 200)
	kernel_y = cv2.getGaussianKernel(rows, 200)
	kernel = kernel_y * kernel_x.T
	mask = 255 * kernel / np.linalg.norm(kernel)
	output = np.copy(img)
	# applying the mask to each channel in the input image
	for i in range(3):
		output[:, :, i] = output[:, :, i] * mask
	return output

def cartoonize_photos(img, ksize=5):
	num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
	# Convert image to grayscale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Apply median filter to the grayscale image
	img_gray = cv2.medianBlur(img_gray, 7)
	# Detect edges in the image and threshold it
	edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
	ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
	# Resize the image to a smaller size for faster computation
	img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
	# Apply bilateral filter the image multiple times
	for i in range(num_repetitions):
		img_small = cv2.bilateralFilter(img_small, ksize, sigma_color,sigma_space)
	img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
	dst = np.zeros(img_gray.shape)
	# Add the thick boundary lines to the image using 'AND' operator
	dst = cv2.bitwise_and(img_output, img_output, mask=mask)
	return dst


def convert_to_pencil_sketch(rgb_image):
	gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
	blurred_image = cv2.GaussianBlur(gray_image, (21, 21), 0, 0)
	gray_sketch = cv2.divide(gray_image, blurred_image, scale=256)
	return cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2RGB)

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped

def scaner(image_dir):
	image = cv2.imread(image_dir)
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	warped = cv2.filter2D(warped, -1, kernel_sharpen_1)
	cv2.imwrite(image_dir, warped)
	return True

def image_to_text(image_dir,dir_name):
	# shutil.rmtree(config["temp_dir_img"], ignore_errors=True)
	# os.makedirs(config["temp_dir_img"])
	file_path= image_dir
	im = Image.open(file_path)
	im.save(dir_name)
	image = cv2.imread(dir_name)
	image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	retval, threshold = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

	text = pytesseract.image_to_string(threshold)
	return text