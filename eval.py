import warnings

import numpy as np

from main import FeatureDetector, Processing, SelectVideo

F = 910
height = 880
width = 1200

K = np.array(([F, 0, width//2], [0, F, height//2], [0, 0, 1]))
fe = FeatureDetector(K)
check = Processing(fe)

cap = SelectVideo().select()

myData = []
isFirstFrame = True
while cap.isOpened():
    warnings.simplefilter('ignore', np.RankWarning)
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ret, currentFrame = cap.read()
    if ret == True:
        currentFrame = currentFrame[0:640, :]
        img2, yaw, pitch = check.process_frame(currentFrame)
        myData.append([np.float64(yaw.tolist()[0]),
                      np.float64(pitch.tolist()[0])])
    else:
        break

GT_DIR = 'labeled/'
gt = np.loadtxt(GT_DIR + str(4) + '.txt')

myData = np.array(myData)

def get_mse(gt, test):
    test = np.nan_to_num(test)
    return np.mean(np.nanmean((gt - test)**2, axis=0))


zero_mses = []
mses = []
zero_mses.append(get_mse(gt, np.zeros_like(gt)))
mses.append(get_mse(gt, myData))

percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)

print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
