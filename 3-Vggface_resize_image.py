import os
import cv2

resize_image_path='C:/Tibame/專題/人臉資料集/VGGface_MTCNN_process'

os.chdir(resize_image_path)

if not os.path.exists('./resize_images'):
    os.mkdir('./resize_images')


os.chdir('./train_images')

n_folders = os.listdir()
# print(n_folders)
# print(os.getcwd())
# os.chdir('n000002')

for nfolder in n_folders:
    print(nfolder)

    os.chdir(resize_image_path+'/train_images/{}'.format(nfolder))

    if os.path.exists(resize_image_path+'/resize_images/with_mask/{}'.format(nfolder)):
        continue
    elif not os.path.exists(resize_image_path+'/resize_images/with_mask/{}'.format(nfolder)):
        os.mkdir(resize_image_path+'/resize_images/with_mask/{}'.format(nfolder))

        n_images = os.listdir()
        for images in n_images:
            if '.jpg' not in images:
                continue
            else:
            # print(images)
                image = cv2.imread(images)
                # print(image.shape)
                new_images = cv2.resize(image,(160,160))
                cv2.imwrite(resize_image_path+'/resize_images/with_mask/{}/{}'.format(nfolder,images), new_images)
        os.chdir(resize_image_path+'/train_images')
