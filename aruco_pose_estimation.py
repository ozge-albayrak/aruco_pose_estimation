import pyrealsense2 as rs
import numpy as np 
import cv2
import cv2.aruco as aruco 
import glob 
import sys, time, math


cv_file = cv2.FileStorage("rs_capture/checkerboard_test.yaml", cv2.FILE_STORAGE_READ)

mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("distortion_coeff").mat()

cv_file.release()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

while(True):
	# Wait for a coherent pair of frames: color only
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
	gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) 

	# Set seelcted aruco marker dictionary size  
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

	parameters = aruco.DetectorParameters_create()
	parameters.adaptiveThreshConstant = 10

	# lists of ids and the corners belonging to each id
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	# font for displaying text (below)
	font = cv2.FONT_HERSHEY_SIMPLEX

	# check if the ids list is not empty
	# if no check is added the code will crash
	if np.all(ids != None):

		# estimate pose of each marker and return the values
		# rvet and tvec-different from camera coefficients

		for i in range(0, len(ids)):
	
			ret = aruco.estimatePoseSingleMarkers(corners[i],0.01,mtx,dist)
			[rvec, tvec] = [ret[0][0, 0, :], ret[1][0, 0, :]]
	
			(rvec-tvec).any() # get rid of that nasty numpy value array error
			rotation_mat, _ = cv2.Rodrigues(rvec)
	
			pose_mat = cv2.hconcat((rotation_mat, tvec))
			hom_mat = np.append(pose_mat,np.array([0,0,0,1])).reshape(4,4)
 			
		    # draw axis for the aruco markers
			print('-------------------------')
			print('ID : {}'.format(ids[i][0]))
			print(hom_mat)
			print('-------------------------')
			aruco.drawAxis(color_image, mtx, dist, rvec, tvec, 0.1)

		# draw a square around the markers
		aruco.drawDetectedMarkers(color_image, corners)

		# code to show ids of the marker found
		strg = ''
		for i in range(0, ids.size):
		    strg += str(ids[i][0])+', '

		cv2.putText(color_image, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

	else:
		# code to show 'No Ids' when no markers are found
		cv2.putText(color_image, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

	# display the resulting frame
	cv2.imshow('frame',color_image)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cv2.destroyAllWindows()














