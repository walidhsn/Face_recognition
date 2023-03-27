from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from deepface.basemodels import VGGFace
import pandas as pd

model=VGGFace.loadModel()

def chosing():
    names = ["walid", "kurt", "leo"]
    name = input('>Enter The Name of the Person for The Search in the DataBase : ')
    if name in names:
        return name
    else:
        return "f"
def display_img(n):
    path = f"C:/Users/WALID/PycharmProjects/pythonProject/image/{n}.jpg"
    img = cv2.imread(path)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    return path
def searching_data(path):
    df = DeepFace.find(img_path=path, db_path='C:/Users/WALID/PycharmProjects/pythonProject/DataBase', model_name= 'VGG-Face', model=model,distance_metric= 'cosine')
    print(df.head())
    print(f">>>The Closest Picture with Path : {df.iloc[0].values}")
    return df.iloc[0].identity
def analyze_pic(path):
    alz = DeepFace.analyze(img_path=path,actions=['age', 'gender', 'race', 'emotion'])
    print("\n Picture Analyze : ")
    print("> Age : ", alz["age"])
    print("> Gender : ", alz["gender"])
    print("> Race : ", alz["race"])
    print("> Emotion : ", alz["emotion"])
name = chosing()
if name != "f":
    p = display_img(name)
    p_closePic = searching_data(p)
    img = cv2.imread(p_closePic)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    analyze_pic(p)

else:
    print("ERROR : CAN'T Find THE PERSON")
    exit()
