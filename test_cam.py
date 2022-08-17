# import the opencv library
import cv2
import os




# define a video capture object
rtsp = "rtsp://admin:XEJVQU@10.37.239.113:554"
vid = cv2.VideoCapture(rtsp)

while(True):
		
	# Capture the video frame
	# by frame
	ret, frame = vid.read()

	# Display the resulting frame
	print(type(frame))
	cv2.putText(frame, '10', (7, 70), 0, 1, (100, 255, 0), 2)
	cv2.imshow('frame', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
