import numpy as np
import cv2

def cal_l2_dist(mat1, mat2):
    """
    - 2개의 matrix mat1, mat2의 l2 거리를 계산하는 함수
    - 거리 distance 가 작을수록 유사하다
    """

    # 예외처리 : 크기가 같지 않으면 안됨
    if mat1.shape != mat2.shape:
        print('Template size not match')
        exit(1) # Process finished with exit code 1 가 뜨면서 꺼짐

    dist = 0.0

    for row in range(mat1.shape[0]): # 행 row
        for col in range(mat1.shape[1]): # 열 col
            l1 = float(mat1[row, col]) - float(mat2[row, col])
            # dist += l1
            l2 = np.sqrt(l1 ** 2)
            # l2 = np.sqrt(np.sum(l1 ** 2))
            dist += l2
        return dist
        # return l2

def mse_template_match(template, img):
    """
    :param template: 우리가 [어떤 물체] 이라고 규정한 것
    :param img: 비교할 이미지
    :return: result: 결과값을 저장할 배열
        결과물 크기 = 비교 이미지 크기 - (템플릿 크기 -1)
        예로들어서 img 가 4*7이고 template 이 3*2 라면
        for문으로 배열 이동할 경우 row는 4 - (3-1) = 2, col은 7 - (2-1) = 6이 된다
    """
    # 결과물
    row = img.shape[0] - (template.shape[0] - 1)
    col = img.shape[1] - (template.shape[1] - 1)

    result = np.zeros((row, col), dtype=float)

    # 좌측 상단 기준으로 for문 돌리기
    for y in range(row):
        for x in range(col):
            roi = img[y:y+template.shape[0], x:x+template.shape[1]]
            # roi : 관심영역 Region of Interest
            # template.shape[0 또는 1] : 템플릿 크기

            result[y, x] = cal_l2_dist(template, roi)

            """
            # img에 rectangle 직접 그리면 이미지에 변형이 생겨서 안됨
            # for문을 돌면서 보이는 roi를 확인해보자
            cv2.imshow("roi", roi)

            print(result[y, x])
            if result[y, x] < 0.1:
                cv2.waitKey(0)
            else:
                cv2.waitKey(10)
            """
    return result

if __name__ == '__main__':
    img = cv2.imread("./image/Lenna.png")

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    template = img_grey[245:370, 240:350].copy()

    mse_template_match(template, img_grey)

    # for showing
    cv2.imshow("img", img_grey)
    cv2.imshow("template", template)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
