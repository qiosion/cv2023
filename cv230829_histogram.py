import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    # gray 변환
    gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    gray20 = gray + 20 # 밝기 20 추가됨

    """
    # 히스토그램 스스로 구하기
    hist = [0] * 256
    # 2차원 배열 for문 돌릴때는 row(y축) 기준으로 하자
    # 이부분 이해안됨
    
    # x : 영상의 명도 값
    # y : 픽셀의 갯수
    for y in range(gray.shape[0]):
        for x in range(gray.shape[1]):
            hist[gray[y, x]] += 1

    # OpenCV 라이브러리의 calcHist 함수 사용
    hist = cv2.calcHist(images=[gray],
                        channels=[0],
                        mask=None,
                        histSize=[256],
                        ranges=[0, 256])

    # numpy 라이브러리 사용 1
    hist, bins = np.histogram(gray.ravel(), 256, [0, 256])
    
    # 2차원 이미지를 flatten 한다
    hist = np.bincount(gray.flatten(), minlength=256)
    """

    # numpy 라이브러리 사용 2 : 더 빠름
    hist = np.bincount(gray.ravel(), minlength=256)

    # 정규화 Normalization
    # hist 타입이 int라서 float 으로 바꿔줘야함
    hist = hist.astype(np.float64)
    hist /= (512. * 512)

    plt.plot(hist)
    plt.show()

    """
    # for showing
    cv2.imshow("src", src)
    cv2.imshow("gray img", gray)
    cv2.imshow("gray 20", gray20)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """