import os
import shutil

path1 = r"predict_test/predict"
path2 = [os.path.join(path1, i) for i in os.listdir(path1)]

for path in path2:
    path3 = [os.path.join(path, i) for i in os.listdir(path) if len(i) == 4]
    for j in path3:
        skin = os.path.join(j, "predict_skin.npy")
        j = j.replace('mixamo/', "mixamo/1")
        j = j.replace('vroid/', "vroid/1")
        print(skin, os.path.join(j, "predict_skin.npy"))
        os.makedirs(j, exist_ok=True)
        shutil.copy(skin, os.path.join(j, "predict_skin.npy"))