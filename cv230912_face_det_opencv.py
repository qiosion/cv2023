import cv2

# 얼굴을 검출하고 싶은 이미지
src = cv2.imread("image/newjeans.jpg")

# 얼굴 검출 모델 가져오기
face_det = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)

# 얼굴이 여러개이므로 detectMultiScale 를 이용하여 list 형식으로 만듦
rects = face_det.detectMultiScale(gray, # 입력 이미지
                                  scaleFactor=1.2, # 이미지 피라미드 스케일을 1.2배로 설정
                                  minNeighbors=2, # 픽셀
                                  minSize=(25, 25)
                                  )

for bbox in rects:
    x, y, w, h = bbox
    cv2.rectangle(src, (x, y), (x+w, y+h),
                  (0, 255, 0), thickness=2)



cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()