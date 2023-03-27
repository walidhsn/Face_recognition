from config import *
from deepface import DeepFace
from deepface.commons import functions

import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle

# Functions


def chosing():
    names = ["walid", "kurt", "leo"]
    name = input('===={INSERT}======> Enter The Name of the Target (walid, leo, kurt) : ')
    if name in names:
        return name
    else:
        return "f"


def display_img(n):
    path = f"../pythonProject/images/{n}.jpg"
    img = functions.preprocess_face(path, target_size=(224, 224))
    print(img.shape)
    plt.imshow(img[0][:, :, ::-1])
    plt.axis('off')
    plt.show()
    return img

# VGG expects 224x224.px


model = DeepFace.build_model("VGG-Face")


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
T_name = chosing()
if T_name != "f":# Processing Target
    target_img = display_img(T_name)
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
    target_duplicated = np.array([target_embedding,]*result_df.shape[0])
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
    result_df = result_df.sort_values(by= ['distance']).reset_index(drop= True)
    result_df = result_df.drop(columns= ["embedding", "target"])
    toc = time.time()
    print(toc-tic, "seconds")
    print("Table of Matches :")
    print(result_df.head(10))
    # Display pictures of Matches
    print("Pictures of Matches :")
    i=0
    for index, instance in result_df.iterrows():
        img_path = instance["img_name"]
        distance = instance["distance"]

        pic = functions.preprocess_face(img_path, target_size=(224, 224))
        plt.imshow(pic[0][:, :, ::-1])
        #plt.axis('off')
        plt.show()
        print(distance)
        print("--------------------------------")
        i = i + 1
        if i == 4:
            break
else:
    print("\n<< Error THE Person u searching for doesn't exist >>")


