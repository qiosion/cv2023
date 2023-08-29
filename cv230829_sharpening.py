import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    # 기존의 이미지와 똑같다
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    # 샤프닝
    sharpen_filter = np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]]) / 9

    # 샤픈 이미지
    s_img = cv2.filter2D(src, -1, sharpen_filter)

    # for showing
    cv2.imshow("src", src)
    cv2.imshow("sharpen filter", s_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()