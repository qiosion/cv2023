import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    blur_img = cv2.blur(src, (5, 5))
    gauss_img = cv2.GaussianBlur(src, (5, 5), 3)

    # for showing
    cv2.imshow("src", src)
    cv2.imshow("mean", blur_img)
    cv2.imshow("gaussian", gauss_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
