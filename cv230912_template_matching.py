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
            l1 = mat1[row, col] - mat2[row, col]
            # dist += l1
            l2 = np.sqrt(l1 ** 2)
            dist += l2
        return dist

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
            # template.shape[0] : 템플릿 크기
            result[y, x] = cal_l2_dist(template, roi)

    return result

if __name__ == '__main__':
    # 템플릿 크기 3*2
    my_template = np.array([[1, 3],
                           [3, 5],
                           [1, 3]])

    # 이미지 크기 4*7
    img = np.array([[1, 2, 3, 4, 5, 6, 7],
                    [1, 3, 5, 7, 9, 11, 13],
                    [3, 5, 6, 7, 8, 9, 10],
                    [1, 3, 0, 1, 2, 0, 1]])

    test = np.array([[1, 5],
                     [3, 1],
                     [0, 2]])

    """
    sum = 0
    sum_l1 = 0
    sum_l2 = 0
    for row in range(my_template.shape[0]):
        for col in range(my_template.shape[1]):
            l1 = my_template[row, col] - img[row, col]
            l2 = l1**2
            sum_l1 += l1
            sum_l2 += l2
            sum_l2 = np.sqrt(sum_l2)

    print('sum_l1', sum_l1)
    print('sum_l2', sum_l2)

    print(cal_l2_dist(my_template, test)) # 2.0

    # 이미지 따는거 좌표 예시. 3*2 템플릿이므로 각 행과 열에 +3, +2 해준것임
    test_img = img[1:1+3, 0:0+2]
    """

    match_result = mse_template_match(my_template, img)
    print(match_result)
