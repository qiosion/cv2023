import cv2

if __name__ == '__main__':
    src = cv2.imread("./image/vegetable.jpg")

    """
    # split 은 자원 소모가 심한 함수이므로 사용하지 말자. 시간도 오래걸린다
    b, g, r = cv2.split(src)
    """
    b = src[:, :, 0]
    g = src[:, :, 1]
    r = src[:, :, 2]

    # # 빨간색 값이 일정수준(200) 이상인 것만 보여줘 -> 빨간색은 흰색으로 나옴
    # _, r = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)

    # 채널 합치기
    img = cv2.merge((b, g, r))

    # for showing
    cv2.imshow("src", src)
    cv2.imshow("blue", b)
    cv2.imshow("green", g)
    cv2.imshow("red", r)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()