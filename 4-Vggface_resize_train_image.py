import os
import numpy as np
import random as rd
import shutil
import glob



##將resize後的影像隨機挑一定數量照片當作訓練資料

resize_image_path='C:/Tibame/專題/人臉資料集/VGGface_MTCNN_process'

os.chdir(resize_image_path)

os.chdir('./resize_images/with_mask')

n_folders = os.listdir()
length_folder=[]

# files = glob.glob('/YOUR/PATH/*')
# for f in files:
#     os.remove(f)
for nfolder in n_folders:
    print(nfolder)

    if os.path.exists(resize_image_path+'/resize_train_images/{}'.format(nfolder)):
        shutil.rmtree(resize_image_path+'/resize_train_images/{}'.format(nfolder))
        print('test')

    if not os.path.exists(resize_image_path+'/resize_train_images/{}'.format(nfolder)):
        print('test')
        os.mkdir(resize_image_path+'/resize_train_images/{}'.format(nfolder))
        print('test')



    os.chdir(resize_image_path+'/resize_images/with_mask/{}'.format(nfolder))


    n_images = os.listdir()
    length_folder.append(len(n_images))

min_len= min(np.array(length_folder))

print(min_len)

for nfolder in n_folders:
    # print(nfolder)

    os.chdir(resize_image_path+'/resize_images/with_mask/{}'.format(nfolder))
    n_images = os.listdir()

    if len(n_images) >= 20 and len(n_images) < 30:
        samples = rd.sample(n_images, len(n_images))
    else:
        samples = rd.sample(n_images, 30)

    for images in samples:

        shutil.copy(images,resize_image_path+'/resize_train_images/{}'.format(nfolder))