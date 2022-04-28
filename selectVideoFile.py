import cv2

class SelectVideo(object):
    def __init__(self) -> None:
        pass

    def select(self):
        # cap = cv2.VideoCapture('labeled/0.hevc')  #NIGHT HIGHWAY
        # cap = cv2.VideoCapture('labeled/1.hevc')  #NIGHT HIGHWAY
        # cap = cv2.VideoCapture('labeled/2.hevc')  #Day + Blur
        # cap = cv2.VideoCapture('labeled/3.hevc')  #NIGHT CITY
        cap = cv2.VideoCapture('labeled/4.hevc')  #Day
        # cap = cv2.VideoCapture('unlabeled/5.hevc')#Day HIGHWAY  
        # cap = cv2.VideoCapture('unlabeled/6.hevc')#DAY RAIN
        # cap = cv2.VideoCapture('unlabeled/7.hevc')#NIGHT Suburb
        # cap = cv2.VideoCapture('unlabeled/8.hevc')#DAY HIGHWAY
        # cap = cv2.VideoCapture('unlabeled/9.hevc')#SNOW

        return cap