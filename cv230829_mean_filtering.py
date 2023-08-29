import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    # 1로 가득찬 3x3 배열이 나옴. 그걸 9로 나누어서 평균을 냄
    kernel = np.ones((3, 3), np.float32) / 9

    # 필터의 크기를 5x5 로 키우면 더 흐릿해진다
    # kernel = np.ones((5, 5), np.float32) / 25
    # print(kernel)

    mean_img = cv2.filter2D(src, -1, kernel)

    """
    filter2D 의 두번째 인자 argument 는 ddepth.
    desired depth of the destination image
    when ddepth=-1, the output image will have the same depth as the source
    """

    # 이러한 배열의 경우 기존의 이미지와 똑같다
    kernel2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    dst = cv2.filter2D(src, -1, kernel2)

    # 이미지 색상 값이 왼쪽 픽셀 값에 5배 곱해진다 ???
    kernel3 = np.array([[0, 0, 0], [5, 0, 0], [0, 0, 0]])
    times = cv2.filter2D(src, -1, kernel3)

    # for showing
    cv2.imshow("src", src)
    cv2.imshow("mean filter", mean_img)
    cv2.imshow("destination", dst)
    cv2.imshow("moved left", times)

    cv2.waitKey(0)
    cv2.destroyAllWindows()