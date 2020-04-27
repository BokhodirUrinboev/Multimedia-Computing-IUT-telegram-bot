import numpy as np
import cv2
import imutils

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

def scaner(image):

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
	return warped