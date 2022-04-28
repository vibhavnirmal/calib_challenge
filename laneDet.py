import warnings

import cv2
from selectVideoFile import SelectVideo
import numpy as np


class LaneDetection:
    def __init__(self) -> None:
        self.prevX1, self.prevY1, self.prevX2, self.prevY2 = 0, 0, 0, 0
        self.prev_lines = []

    def make_coordinates(self, img, line_para):
        if str(type(line_para)) == "<class 'numpy.float64'>":
            slope, intercept = 0.01,0
        else:
            slope, intercept = line_para

        try:
            y1 = img.shape[0]
            y2 = int(y1*(3.65/5))
            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)

            self.prevX1, self.prevX2, self.prevY1, self.prevY2 = x1, x2, y1, y2
            return np.array([x1, y1, x2, y2])

        except OverflowError:
            print("OverflowError")
            return np.array([self.prevX1, self.prevY1, self.prevX2, self.prevY2])

    def roi(self, image, default=True):
        height = image.shape[0]
        width = image.shape[1]
        if default:
            poly = np.array([
                [(150, height-50), (width-150, height-50), (550, 250)]
            ])
        else:
            poly = np.array([
                [(100, height-270), (width-100, height-270), (830, 300)]
            ])

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, poly, 255)
        masked_img = cv2.bitwise_and(image, mask)
        return masked_img

    def avg_line_img(self, image, lines):
        left_ln, right_ln, left_fit, right_fit, bigData = [], [], [], [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                para = np.polyfit((x1, x2), (y1, y2), 1)

                slope = para[0]
                intercept = para[1]

                if slope < 0:
                    if (round(slope) > -2 and slope < -0.6) and (round(intercept) > 750 and round(intercept) < 950):
                        left_fit.append((slope, intercept))
                else:
                    if (round(slope) < 2 and slope > 0.6) and (round(intercept) > -34 and round(intercept) < 60):
                        right_fit.append((slope, intercept))

            left_fit_avg = np.average(left_fit, axis=0)
            right_fit_avg = np.average(right_fit, axis=0)


            left_ln.append(left_fit_avg)
            right_ln.append(right_fit_avg)
            
            left_line = self.make_coordinates(image, left_fit_avg)
            right_line = self.make_coordinates(image, right_fit_avg)

            return np.array([left_line, right_line])

    def canny_det(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        theCanny = cv2.Canny(blur, 20, 80)
        return theCanny

    def displayLine(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                if x1 < 0 or x2 < 0:
                    x1, x2 = self.prevX1, self.prevX2
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return line_image


if __name__ == "__main__":

    def drawImg(title, frame):
        cv2.imshow(title, frame)
        cv2.waitKey(1)

    detect = LaneDetection()

    
    cap = SelectVideo().select()

    croppedFrame = True

    while cap.isOpened():
        warnings.simplefilter('ignore', np.RankWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _, frame = cap.read()
        if _:
            if croppedFrame:
                frame = frame[0:640, :]
                canny_img = detect.canny_det(frame)
                cropped_img = detect.roi(canny_img)
            else:
                canny_img = detect.canny_det(frame)
                cropped_img = detect.roi(canny_img, False)

            lines = cv2.HoughLinesP(cropped_img, rho=8, theta=np.pi/160,
                                    threshold=70, lines=np.array([]),
                                    minLineLength=40, maxLineGap=5)
            avg_lines = detect.avg_line_img(frame, lines)
            line_image = detect.displayLine(frame, avg_lines)
            combo = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            # drawImg("Lane Detection", cropped_img)
            drawImg("Lane Detection", combo)
