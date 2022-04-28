import warnings

import cv2
import numpy as np

from featureDet import FeatureDetector, Processing
from laneDet import LaneDetection
from selectVideoFile import SelectVideo


def drawImg(title, frame):
    cv2.imshow(title, frame)
    cv2.waitKey(1)  # 0xFF == ord('q')

detect = LaneDetection()

F = 910
height = 880
width = 1200

K = np.array(([F, 0, width//2], [0, F, height//2], [0, 0, 1]))
fe = FeatureDetector(K)
check = Processing(fe)


cap = SelectVideo().select()


# fps= int(cap.get(cv2.CAP_PROP_FPS))
# print("This is the fps ", fps)


isFirstFrame = True
while cap.isOpened():
    warnings.simplefilter('ignore', np.RankWarning)
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ret, currentFrame = cap.read()
    if ret == True:
        # Lane Detection Code
        # tempFrame = currentFrame
        currentFrame = currentFrame[0:640, :]

        canny_img = detect.canny_det(currentFrame)
        cropped_img = detect.roi(canny_img)
        lines = cv2.HoughLinesP(cropped_img, rho=5, theta=np.pi/160,
                                threshold=30, lines=np.array([]),
                                minLineLength=35, maxLineGap=8)
        avg_lines = detect.avg_line_img(currentFrame, lines)
        line_image = detect.displayLine(currentFrame, avg_lines)

        # Feature Detection
        # img2, kp2, des2 = check.process_frame(currentFrame)
        img2 = check.process_frame(currentFrame)

        # Draw on image
        combo = cv2.addWeighted(img2, 0.8, line_image, 1, 1)
        drawImg("Lane Detection", combo)
    else:
        break
