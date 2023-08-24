import cv2

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
    print("gray : ", gray[0, 0]) # 162

    # Gray 영상을 Color 로 바꾸면?
    g2c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print("color : ", g2c[0, 0]) # [162 162 162]
    # 이미 정보가 다 날라가고 없다. 모르는 값을 전부 평균값인 162로 채운다

    # g2c는 흑백 이미지이지만, 이 위에 다시 칼라 요소를 입힐 수 있다
    cv2.rectangle(g2c, (100, 100), (200, 200), (0, 255, 0), 2)

    # for showing
    cv2.imshow("src", src)
    cv2.imshow("gray", gray)
    cv2.imshow("g2c", g2c)

    cv2.waitKey(0)
    cv2.destroyAllWindows()