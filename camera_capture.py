## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

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

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

       # depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        c = cv2.waitKey(1)

	# Manual collection (space bar)
	# Hit the space to save the image (ASCII code of the space bar is 32)

	if c == 32:

		# Calculate the number of jpg files in the folder to prevent overwriting the existing pictures
		count = 0 
		
		for filename in os.listdir('../aruco/rs_capture/checkerboard_imgs/'):
			if filename.endswith('.jpg'):
				count += 1
	  
		cv2.imwrite('../aruco/rs_capture/checkerboard_imgs/{}.jpg'.format(count + 1), color_image)

	if c == 27:

		cv2.destroyAllWindows()
		break

finally:

    # Stop streaming
    pipeline.stop() 








