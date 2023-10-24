import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# 학습용 data set
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

# 모델
def make_cnn_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", input_shape=(224, 224, 3),
                     padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

# 학습
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

    # one-hot encoding
    Y = tf.keras.utils.to_categorical(Y_int, 2)

    # 학습 후 저장
    # model.fit(X, Y, batch_size=2, epochs=10)
    model.fit(X, Y, batch_size=2, epochs=100)
    model.save("e10_test2.h5")

# 테스트 data set
def test_one_sample(path):
    print('path : ', path)

    img = cv2.imread(path)
    data = cv2.resize(img, (224, 224))  # 사이즈 조절
    data = data / 255.

    model = tf.keras.models.load_model("./e10_test2.h5")

    input_x = np.expand_dims(data, axis=0)
    result = model.predict(input_x)

    # 파일명에서 fail 또는 pass 추출
    if "fail" in path:
        text = 'fail'
        actual_class = "NG"
    elif "pass" in path:
        text = 'pass'
        actual_class = "OK"
    else:
        actual_class = "Unknown"

    ans = result[0].argmax()

    if actual_class == "NG" and ans == 0:
        print('맞음')
        print("actual_class : ", actual_class)
        print("ans : NG")
    elif actual_class == "OK" and ans == 1:
        print('맞음')
        print("actual_class ", actual_class)
        print("ans : OK")
    else:
        print('틀림')

if __name__ == '__main__':
    # 학습용 data set
    # make_npz('./ML_kuffers/TrainSet/')

    # 학습용 data set
    # call_train_data_set = np.load("./train_data.npz")

    # 모델
    # model = make_cnn_model()

    # 학습
    # train(model)

    # 테스트 data set 만들기
    img_path = 'D:/KES/Python/cv2023/ML_kuffers/TestAll/20171120_102900.699_src_pass_960x960.png'
    test_one_sample(img_path)


