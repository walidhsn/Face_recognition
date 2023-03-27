import cv2.cv2
from cv2 import cv2
from config import *
from deepface import DeepFace
from deepface.commons import functions
from simple_facerec import SimpleFacerec
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def display_img(n):

    path_p = f"../pythonProject/images/target{n}.jpg"
    picture = functions.preprocess_face(path_p, target_size=(224, 224), enforce_detection=False)
    print(picture.shape)
    plt.imshow(picture[0][:, :, ::-1])
    plt.axis('off')
    plt.show()
    return picture

# VGG expects 224x224.px


model = DeepFace.build_model("VGG-Face")


def get_pic(img, x):
    f = f"../pythonProject/images/target{x}.jpg"
    cv2.imwrite(f, img)

srf = SimpleFacerec()
srf.load_encoding_images("C:/Users/WALID/PycharmProjects/pythonProject/image/")

# Camera

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0
count = -1
while True:
    ret, frame = cap.read()
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
    face_location, face_names = srf.detect_known_faces(frame)
    for face_loc, name in zip(face_location, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 1), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (2, 1, 200), 4)
    cv2.imshow("Taking Picture", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32:
        count = count + 1
        get_pic(frame, count)


cap.release()
cv2.destroyWindow("Taking Picture")

facial_img_paths = []
for root, directory, files in os.walk("../pythonProject/DataCenter"):
    for file in files:
        if '.jpg' in file:
            facial_img_paths.append(root + "/" + file)
instances = []
for i in tqdm(range(0, len(facial_img_paths))):
    facial_img_path = facial_img_paths[i]
    # detect and align
    facial_img = functions.preprocess_face(facial_img_path, target_size=(224, 224))
    # represent
    embedding = model.predict(facial_img)[0]

    # store
    instance = []
    instance.append(facial_img_path)
    instance.append(embedding)
    instances.append(instance)
df = pd.DataFrame(instances, columns=["img_name", "embedding"])
print(df.head(11))
# Connecting to Db
con = connexion("")
conn = con.getcon()
cursor = conn.cursor()
# Creating Tabels
cursor.execute("drop table if exists face_meta")
cursor.execute("drop table if exists face_embeddings")
cursor.execute("create table face_meta (ID INT primary key, IMG_NAME VARCHAR(70), EMBEDDING BLOB)")
cursor.execute("create table face_embeddings (FACE_ID INT, DIMENSION INT, VALUE DECIMAL(30, 30))")
print("\n <<DATA TABLES CREATED>>")
# Storing Data
for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
    img_name = instance["img_name"]
    embeddings = instance["embedding"]
    insert_statement = '''INSERT INTO face_meta (ID, IMG_NAME, EMBEDDING) VALUES (%s, %s, %s)'''
    insert_args = (index, img_name, embeddings.tobytes())
    cursor.execute(insert_statement, insert_args)
    for i, embedding in enumerate(embeddings):
        insert_statement = '''INSERT INTO face_embeddings (FACE_ID, DIMENSION, VALUE) VALUES (%s, %s, %s)'''
        insert_args = (index, i, str(embedding))
        cursor.execute(insert_statement, insert_args)
        conn.commit()
print("\n << DATA Stored in TABLES >>")

# Choising Target

target_img = display_img(count)

target_embedding = model.predict(target_img)[0].tolist()
# Process
tic = time.time()
select_statement = "select img_name, embedding from face_meta"
cursor.execute(select_statement)
results = cursor.fetchall()
instances = []
for result in results:
    img_name = result[0]
    embedding_bytes = result[1]
    embedding = np.frombuffer(embedding_bytes, dtype='float32')
    instance = []
    instance.append(img_name)
    instance.append(embedding)
    instances.append(instance)
toc = time.time()
print(toc-tic, "Seconds")
result_df = pd.DataFrame(instances, columns=["img_name", "embedding"])
target_duplicated = np.array([target_embedding, ]*result_df.shape[0])
result_df['target'] = target_duplicated.tolist()
print(result_df.head(11))


def findEcludienDistance(row):
    source = np.array(row['embedding'])
    target = np.array(row['target'])
    distance = (source - target)
    return np.sqrt(np.sum(np.multiply(distance, distance)))

# Display Table of Matches

tic = time.time()
result_df['distance'] = result_df.apply(findEcludienDistance, axis=1)
result_df = result_df[result_df['distance'] <= 10]
result_df = result_df.sort_values(by=['distance']).reset_index(drop=True)
result_df = result_df.drop(columns=["embedding", "target"])
toc = time.time()
print(toc-tic, "seconds")
print("Table of Matches :")
print(result_df.head(10))
# Display pictures of Matches

print("Pictures of Matches :")
j=0
for index, instance in result_df.iterrows():
    img_path = instance["img_name"]
    distance = instance["distance"]

    pic = functions.preprocess_face(img_path, target_size=(224, 224))
    plt.imshow(pic[0][:, :, ::-1])

    #plt.axis('off')

    plt.show()
    print(distance)
    print("--------------------------------")
    j = j + 1
    if j == 4:
        break
