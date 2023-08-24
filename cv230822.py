import cv2

if __name__ == '__main__':
    src = cv2.imread("./image/Lenna.png")

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    """
    print(src.shape) # row, column, channel(B,G,R)

    # 초록 점 찍기
    print(src[100][100]) # 해당 좌표 부분의 RGB를 출력
    src[100][100] = [0, 255, 0] # 해당 좌표 부분을 초록색으로 변경
    
    # 고전적인 방식으로 초록 줄 찍기
    green = [0, 255, 0]
    for i in range(100, 200):
        src[100][i] = green # row는 100으로 고정하고 i를 100~200으로 줌 -> 그 부분이 쭉 초록줄이 됨

    # 파이썬 스러운 방법으로 줄 찍기
    green = [0, 255, 0]
    red = [0, 0, 255]
    src[100, 100:200] = green # row를 고정하고 coloumn을 변경한 것
    src[150:250, 100] = red
    # 이렇게 한뒤 저장하면 원본이 바뀌므로, 보통은 원본 소스를 따로빼놓는다

    blue = [255, 0, 0]
    green = [0, 255, 0]
    red = [0, 0, 255]

    # 사진에서 얼굴 좌표만 가져오기
    # shallow copy 하면 face만 수정했어도 src 파일에서도 변경된 채로 나온다 -> 원본도 함께 바뀌고 있다는 뜻
    # deep copy 해보자
    face = src[180:390, 180:370].copy() # row, col 순서

    # face 전체를 흰색으로 바꾼다
    # face[:, :] = [255, 255, 255]

    # 채널로 접근
    # 0 : blue, 1 : green, 2 : red
    face[:, :, 1] = 255

    # face 파일을 저장해보자
    cv2.imwrite("./image/Lenna_green.png", face)
    """


    # for showing
    cv2.imshow("src", src)
    # cv2.imshow("face", face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

