import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    # gray 변환
    gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)

    """
    # 직접 정규화 Normalization 
    norm_img = gray.astype(np.float32)
    norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
    """

    # open cv를 이용한 정규화
    norm_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # opencv 히스토그램 이퀄라이저
    hist_eq = cv2.equalizeHist(norm_img)

    # 칼라 이미지를 평탄화 Equalization 해보자
    # 1. 영상을 RGB 채널로 나눈다
    b = src[:, :, 0] # blue
    g = src[:, :, 1] # green
    r = src[:, :, 2] # red

    # 2. 각 채널에 대해 평탄화를 진행한다
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)

    # 3. 평탄화된 채널을 합친다
    color_eq = cv2.merge((b, g, r))

    # for showing
    cv2.imshow("gray image", gray)
    cv2.imshow("normalized image", norm_img)
    cv2.imshow("histogram equalization ", hist_eq)
    cv2.imshow("Color image equalization ", color_eq)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
