import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 학습 데이터 보기
def show_all_image(path):
    category = os.listdir(path) # ['NG', 'OK']

    # 경로 만들기
    # folder = path + category[0]
    for class_name in category:
        # 폴더
        folder = path + class_name + "/"

        # 폴더 내의 파일들
        files = os.listdir(folder)
        for file in files:
            # 전체 경로
            full_path = folder + file

            img = cv2.imread(full_path)

            # TODO: 만약 이미지를 저장할거면 반드시 지울 것. 원본이 손상 됨
            # NG / OK 표시
            cv2.putText(img, class_name, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

            data = cv2.resize(img, (224, 224))  # 사이즈 조절
            cv2.imshow("original_image", data)

            # 이미지가 너무 많으니까 중간에 끌수있게 함
            # esc를 누르면 이미지를 덜 봐도 종료
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                exit(0)

# 학습용 data set 로 만들기
def make_npz(path):
    X = [] # 이미지
    Y = [] # 클래스 이름

    category = os.listdir(path)  # ['NG', 'OK']

    # 경로 만들기
    # folder = path + category[0]
    for class_name in category:
        # 폴더
        folder = path + class_name + "/"

        # 폴더 내의 파일들
        files = os.listdir(folder)
        for file in files:
            # 전체 경로
            full_path = folder + file
            img = cv2.imread(full_path)
            data = cv2.resize(img, (224, 224)) # 사이즈 조절

            # 빈 리스트 X, Y에 append
            X.append(data/255.)
            Y.append(class_name) # class_name이 String인걸 잊지말기

    # np array 형태로 바꾼다
    X = np.array(X)
    Y = np.array(Y)

    # 파일로 저장
    np.savez("./train_data.npz", X, Y)

# 모델 만들기
def make_cnn_model():
    model = Sequential()

    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     activation="relu",
                     input_shape=(224, 224, 3), # 크기 주의
                     padding = "same"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dense(2, activation="softmax")) # 결과는 2개로 분리

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    return model

# 학습 : 클래스 개수가 너무 많아서 인코더 사용
def train(model):
    # 학습용 data set 불러오기
    test = np.load("./train_data.npz")
    X = test["arr_0"]
    Y_str_label = test["arr_1"]

    # 스트링으로 되어있는 label을 int형으로 변경
    # 키워드 이름들을 0과 1로 바꿔줌
    encoder = LabelEncoder()
    encoder.fit(Y_str_label)
    Y_int = encoder.transform(Y_str_label)
    # print(Y_str_label) # ['NG' 'NG' ... 'OK' 'OK']
    # print(Y_int) # [0 0 ... 1 1]

    # one-hot encoding
    Y = tf.keras.utils.to_categorical(Y_int, 2)
    # print(Y) # [[1. 0.] [1. 0.] [0. 1.] [0. 1.]]

    # 학습 후 저장
    model.fit(X, Y, batch_size=2, epochs=10)
    model.save("e10_test.h5")

# 테스트 data set 만들기
def make_test_data():
    path = './ML_kuffers/TestSet/'

    X = []  # 이미지
    Y = []  # 클래스 이름

    category = os.listdir(path)  # ['NG', 'OK']

    # 경로 만들기
    for class_name in category:
        # 폴더
        folder = path + class_name + "/"

        # 폴더 내의 파일들
        files = os.listdir(folder)
        for file in files:
            # 전체 경로
            full_path = folder + file
            img = cv2.imread(full_path)
            data = cv2.resize(img, (224, 224))  # 사이즈 조절

            # 빈 리스트 X, Y에 append
            X.append(data / 255.)
            Y.append(class_name)  # class_name이 String인걸 잊지말기

    # np array 형태로 바꾼다
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# 테스트 함수
def test():
    model = tf.keras.models.load_model("./e10_test.h5")
    X, Y = make_test_data()

    # result = model.predict(X)
    # print(result)

    # 결과를 하나하나씩 따로 보고싶다면
    for i, img in enumerate(X):
        # 차원을 하나 늘려줌
        input_x = np.expand_dims(img, axis=0)
        result = model.predict(input_x)

        # 어떤 이미지가 들어가서 어떤 결과를 보는지 알고싶다
        cv2.imshow("input img", img)
        print("정답은 ", Y[i])
        ans = result[0].argmax()

        if Y[i] == "NG" and ans == 0:
            print("맞췄음")
        elif Y[i] == "OK" and ans == 1:
            print("맞췄음")
        else:
            print("틀렸음")

        key = cv2.waitKey()
        if key == 27: # esc 키를 누르면
            cv2.destroyAllWindows()
            exit(0)


if __name__ == '__main__':
    # ML_kuffers 는 첨부자료의 "2_1_딥러닝 양불 판별" 폴더

    # 파일 내 이미지 모두 보기
    # show_all_image('./ML_kuffers/TrainSet/')

    # 학습용 data set 만들기 -> 한번 사용했다면 파일이 생겼으니 주석 처리
    # make_npz('./ML_kuffers/TrainSet/') # 이 결과로 cv2023 폴더에 파일이 생김

    # 학습용 data set 불러오기
    # call_train_data_set = np.load("./train_data.npz")
    # print(test) # NpzFile './train_data.npz' with keys: arr_0, arr_1
    # X = call_train_data_set["arr_0"]
    # Y = call_train_data_set["arr_1"]

    # print(X.shape) # (80, 224, 224, 3)
    # print(Y.shape) # (80,)

    # 모델
    # model = make_cnn_model()

    # 학습
    # train(model)

    # # 테스트 data set 만들기
    # make_test_data()

    test()