# coding: utf-8

import numpy as np
import os
from tqdm import tqdm
import math
import tensorflow as tf
from scipy import misc
import sys

from skimage.transform import rotate
import random
import numpy as np
import time


def IOU(box, boxes):
    '''裁剪的box和圖片所有人臉box的iou值
    參數：
      box：裁剪的box,當box維度為4時表示box左上右下座標，維度為5時最後一維為box的可信度
      boxes：圖片所有人臉 box,[n,4]
    返回值：
      iou值，[n,]
    '''
    # box面積
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes面積,[n,]
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 重疊部分左上右下座標
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 重疊部分長寬
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # 重疊部分面積
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)


def read_annotation(base_dir, label_path):
    '''讀取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 圖像位置
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人臉數量
        nums = labelfile.readline().strip('\n')

        one_image_bboxes = []
        for i in range(int(nums)):
            bb_info = labelfile.readline().strip('\n').split(' ')
            # 人臉框
            face_box = [float(bb_info[i]) for i in range(4)]

            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]

            one_image_bboxes.append([xmin, ymin, xmax, ymax])

        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes
    return data


def convert_to_square(box):
    '''將box轉換成更大正方形
    參數：
      box：預測的,[n,5]
    返回值：
      調整後的正方形box，[n,5]
    '''
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    # 找正方形最大邊長
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


class ImageClass():
    '''獲取圖片類別和路徑'''

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(paths):  # 輸入的paths為放所有待訓練影像的資料夾
    dataset = []
    classes = [path for path in os.listdir(paths) if
               os.path.isdir(os.path.join(paths, path))]  # 如果paths/path是資料夾的話, classes = path
    classes.sort()
    nrof_classes = len(classes)  # 有幾個資料夾
    for i in tqdm(range(nrof_classes)):
        class_name = classes[i]
        facedir = os.path.join(paths, class_name)  # 得到每個資料夾路徑
        image_paths = get_image_paths(facedir)  # 得到每張影像的路徑
        dataset.append(ImageClass(class_name, image_paths))  # 將標籤名稱與影像路徑透過ImageClass 合併
    return dataset


def get_image_paths(facedir):  # 輸入的facedir為放個別待影像的資料夾
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]  # 導出每張影像的路徑
    return image_paths


def split_dataset(dataset, split_ratio, min_nrof_images_per_class):
    '''拆分訓練集與驗證集
    參數：
      dataset:由get_dataset生成的數據集
      split_ratio:驗證集的比率
      min_nrof_images_per_class：一個類別中最少含有的圖片數量，過少捨棄
    返回值：
      train_set,test_set:含有圖片的類別和路徑的訓練集與驗證集
    '''
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths  # 讀入class的dataset部分
        np.random.shuffle(paths)  # 隨機重新排列
        # 種類的圖片總數輛量
        nrof_images_in_class = len(paths)
        # 留取訓練的比例
        split = int(math.floor(nrof_images_in_class * (1 - split_ratio)))  # 計算訓練集影像的數量
        if split == nrof_images_in_class:  # 如果split_ratio設定成0 則沒有拆分
            split = nrof_images_in_class - 1
        if split >= min_nrof_images_per_class and nrof_images_in_class - split >= 1:
            # 如果拆分後的訓練集影像數量大於該類別影像的最小張數值，而且類別影像總張數大於等於拆分後的訓練集影像數量
            train_set.append(ImageClass(cls.name, paths[:split]))  # 利用ImageClass合併類別名稱與訓練集
            test_set.append(ImageClass(cls.name, paths[split:]))  # 利用ImageClass合併類別名稱與驗證集
    return train_set, test_set


def get_image_paths_and_labels(dataset):
    '''獲取所有圖像位置和相應類別'''
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths  # dataset 的影像位置
        labels_flat += [i] * len(dataset[i].image_paths)  # dataset 的類別標籤
    return image_paths_flat, labels_flat


def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, bath_size_placeholder):
    '''由輸入隊列返回圖片和label的batch組合
    參數：
      input_queue:書入隊列
      image_size:影像尺寸
      nrof_preprocess_threads:線程數，相當於設定用了幾個核心CPU同時來運作
      batch_size_placeholder:batch_size的placeholder
    返回值：
      image_batch,label_batch:影像和標籤的batch組合
    '''
    image_and_labels_list = []
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):  # tf.unstack為將列內的filename依序拆開
            file_contents = tf.io.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)  # 讀取影像
            # 隨機旋轉圖像
            image = tf.cond(tf.constant(np.random.uniform() > 0.8),
                            lambda: tf.compat.v1.py_func(random_rotate_image, [image], tf.uint8),
                            lambda: tf.identity(image))
            # 隨機裁剪圖像
            image = tf.cond(tf.constant(np.random.uniform() > 0.5),
                            lambda: tf.image.random_crop(image, image_size + (3,)),
                            lambda: tf.compat.v1.image.resize_image_with_crop_or_pad(image, image_size[0],
                                                                                     image_size[1]))
            # 隨機左右翻轉圖像
            image = tf.cond(tf.constant(np.random.uniform() > 0.7),
                            lambda: tf.image.random_flip_left_right(image),
                            lambda: tf.identity(image))
            # 圖像歸一到[-1,1]內
            image = (tf.cast(image, tf.float32) - 127.5) / 128.0
            image.set_shape(image_size + (3,))
            images.append(image)
        image_and_labels_list.append([images, label])
    image_batch, label_batch = tf.compat.v1.train.batch_join(image_and_labels_list,
                                                             batch_size=bath_size_placeholder,
                                                             shapes=[image_size + (3,), ()],
                                                             # 每個樣本的形狀，默認為tensor_list_list[i]的形狀
                                                             enqueue_many=True,  # 在tensor_list_list是多個樣本
                                                             capacity=4 * nrof_preprocess_threads * 100,
                                                             allow_smaller_final_batch=True)  # 如果最後一個batch資料量不夠時，允許繼續進行
    return image_batch, label_batch


def random_rotate_image(image):
    '''随机翻转图片'''
    angle = np.random.uniform(low=-10.0, high=10.0)
    return rotate(image, angle)


import tensorflow as tf
import tf_slim as slim


# Inception-resnet-A模塊 要重複執行5次
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """建立一個 35x35 resnet block."""
    with tf.compat.v1.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            # 35 x 35 x 32
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            # 35 x 35 x 32
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            # 35 x 35 x 32
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
            # 35 x 35 x 32
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            # 35 x 35 x 32
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            # 35 x 35 x 32
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        # 35 x 35 x 96
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        # 35 x 35 x 256
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        # 使用殘差網路 scale = 0.17
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# 5 x Inception-resnet-A
# net = slim.repeat(net, 5, block35, scale = 0.17)
# end_points['Mixed_5a'] = net

# Reduction-A 模塊
def reduction_a(net, k, l, m, n):
    # 192, 192, 256, 384
    with tf.compat.v1.variable_scope('Branch_0'):
        # 17 x 17 x 384
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.compat.v1.variable_scope('Branch_1'):
        # 35 x 35 x 192
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        # 35 x 35 x 192
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        # 17 x 17 x 256
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.compat.v1.variable_scope('Branch_2'):
        # 17 x 17 x 256
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    # 17 x 17 x 896
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net


# Reduction-A
# with tf.variable_scope(('Mixed_6a'):
#     net = reduction_a(net, 192, 192, 256, 384)
#     end_points['Mixed_6a'] = net


# Inception-Resnet-B 模塊，要重複10次，輸入為 17×17×896，輸出為17×17×896
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.compat.v1.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.compat.v1.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


def reduction_b(net):
    with tf.compat.v1.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.compat.v1.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.compat.v1.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.compat.v1.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                     tower_conv2_2, tower_pool], 3)
    return net


def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected], \
                        weights_initializer=slim.initializers.xavier_initializer(), \
                        weights_regularizer=slim.l2_regularizer(weight_decay), \
                        normalizer_fn=slim.batch_norm, \
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size,
                                   reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None,
                        scope='InceptionResnetV1'):
    """創建 Inception Resnet V1 model.
    參數:
      inputs: 輸入圖像 [batch_size, height, width, 3].
      num_classes: 预测类别数量.
      is_training: 是否训练
      dropout_keep_prob: dropout的機率
      reuse: 參數是否共享
      scope: 變數命名
    返回值:
      logits: 模型輸出.
      end_points: 模型節點輸出集合
    """
    end_points = {}

    with tf.compat.v1.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net

                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                # Reduction-A
                with tf.compat.v1.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net

                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net

                # Reduction-B
                with tf.compat.v1.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net

                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net

                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.compat.v1.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    # pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)

                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')

                    end_points['PreLogitsFlatten'] = net

                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                                           scope='Bottleneck', reuse=False)

    return net, end_points


def main():
    # 輸入影像大小
    image_size = 160
    # graph儲存位置
    graph_dir = 'C:/Tibame/專題/人臉資料集/VGGface_MTCNN_process/graph/'
    # 模型儲存位置
    model_dir = 'C:/Tibame/專題/人臉資料集/VGGface_MTCNN_process/model/'
    # 驗證集占比
    validation_set_split_ratio = 0.2
    # 一個類別至少含有圖片數量
    min_nrof_val_images_per_class = 20.0
    # 數據存放位置
    data_dir = 'C:/Tibame/專題/人臉資料集/VGGface_MTCNN_process/resize_train_images'

    batch_size = 20  # 90
    # dropout的保留率
    keep_probability = 0.8
    # 網路輸出層維度
    embedding_size = 512
    # L2權值正則
    weight_decay = 5e-4
    # center_loss的center更新參數
    center_loss_alfa = 0.6
    # center_loss占比
    center_loss_factor = 1e-2
    # 初始學習率
    learning_rate = 0.01
    # 學習率衰减epoch數
    # 學習率減少的迭代次數
    LR_EPOCH = [1,2,3,4,5, 6,7,8, 9, 10, 20, 40]
    # 學習率衰減率
    learning_rate_decay_factor = 0.85
    # 指數衰減參數
    moving_average_decay = 0.999
    # 訓練最大epoch
    max_nrof_epochs = 10


    image_size = (image_size, image_size)  # 輸入影像大小
    # 創建graph和model存放目錄
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # 獲取圖片地址和類別
    dataset = get_dataset(data_dir)  # 參考align.utils
    # 訓練集與驗證集拆分
    if validation_set_split_ratio > 0.0:
        train_set, val_set = split_dataset(dataset, validation_set_split_ratio,
                                           min_nrof_val_images_per_class)  # 參考align.utils
    else:
        train_set, val_set = dataset, []  # 如果split_ratio設為0，則沒有驗證集
    # 訓練集的種類數量
    nrof_classes = len(train_set)
    with tf.Graph().as_default():  # 另開一個新設定tensorflow運作區塊
        global_step = tf.Variable(0, trainable=False)
        # 獲取所有圖像位置和相應類別
        image_list, label_list = get_image_paths_and_labels(train_set)  # 參考align.utils
        assert len(image_list) > 0, '訓練集不能為空'
        val_image_list, val_label_list = get_image_paths_and_labels(val_set)  # 參考align.utils

        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)  # 將類別標籤轉為張量
        # 樣本數量
        range_size = labels.get_shape().as_list()[0]
        # 每一個epoch的batch數量
        epoch_size = range_size // batch_size
        # 創建隊列
        index_queue = tf.compat.v1.train.range_input_producer(range_size, num_epochs=None,
                                                              shuffle=True, seed=None, capacity=32000)

        index_dequeue_op = index_queue.dequeue_many(batch_size * epoch_size, 'index_dequeue')  # 從隊列中取出訓練的資料

        batch_size_placeholder = tf.compat.v1.placeholder(tf.int32, name='batch_size')  # 定義batch_size的型態
        phase_train_placeholder = tf.compat.v1.placeholder(tf.bool, name='phase_train')  # 定義phase_train的型態
        image_paths_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None, 1),
                                                           name='image_paths')  # 定義image_paths的型態
        labels_placeholder = tf.compat.v1.placeholder(tf.int32, shape=(None, 1), name='label')  # 定義labels的型態
        keep_probability_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                                name='keep_probability')  # 定義keep_probability的型態

        nrof_preprocess_threads = 4  # ?
        # 輸入隊列
        input_queue = tf.queue.FIFOQueue(capacity=2000000,  # 創建輸入隊列採先進先出，總共可含2000000數量
                                         dtypes=[tf.string, tf.int32],  # 隊列可儲存的資料型態
                                         shapes=[(1,), (1,)],  # 隊量內各元素的尺寸
                                         shared_name=None,
                                         name=None)  # shared_name=None :隊列的名稱不能在不同session間共享  ， Name:隊列操作的命名
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder],
                                              # 將image_paths和labels分別放入隊列相對位置並按照隊列設定方法依據取出
                                              name='enqueue_op')
        # 獲取圖像和label的batch形式
        image_batch, label_batch = create_input_pipeline(input_queue,
                                                         image_size,
                                                         nrof_preprocess_threads,
                                                         batch_size_placeholder)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        # 網路輸出
        prelogits, _ = inference(image_batch,
                                 keep_probability_placeholder,
                                 phase_train=phase_train_placeholder,
                                 bottleneck_layer_size=embedding_size,
                                 weight_decay=weight_decay)
        # 用於計算loss
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(weight_decay),
                                      scope='Logits', reuse=False)
        # 正則化的embeddings主要用於測試，對比兩張圖片差異
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # 計算centerloss
        prelogits_center_loss, _ = center_loss(prelogits, label_batch, center_loss_alfa, nrof_classes)
        tf.identity(prelogits_center_loss, name='center_loss')
        tf.summary.scalar('center_loss', prelogits_center_loss)
        # 學習率
        boundaries = [int(epoch * range_size / batch_size) for epoch in LR_EPOCH]
        lr_values = [learning_rate * (learning_rate_decay_factor ** x) for x in range(0, len(LR_EPOCH) + 1)] #到設定的epoch時學習率減少 0.5 倍
        learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries, lr_values)
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        # 交叉熵損失
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                       logits=logits,
                                                                       name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.identity(cross_entropy_mean, name='cross_entropy_mean')
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        # L2正則loss
        L2_loss = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        # 全部loss
        total_loss = cross_entropy_mean + center_loss_factor * prelogits_center_loss + L2_loss
        tf.identity(total_loss, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

        # 準確率
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.identity(accuracy, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

        train_op = optimize(total_loss, global_step,
                            learning_rate,
                            moving_average_decay,
                            tf.compat.v1.global_variables())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=3)
        summary_op = tf.compat.v1.summary.merge_all()
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        # 訓練和驗證的graph保存位置
        train_writer = tf.compat.v1.summary.FileWriter(graph_dir + 'train/', sess.graph)
        val_writer = tf.compat.v1.summary.FileWriter(graph_dir + 'val/', sess.graph)
        coord = tf.compat.v1.train.Coordinator()
        tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if os.path.exists(model_dir):
                model_file = tf.train.latest_checkpoint(model_dir)
                if model_file:
                    saver.restore(sess, model_file)
                    print('重載模型訓練')

            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            start_training_time=time.time()
            for epoch in range(1, max_nrof_epochs + 1):
                start_time=time.time()
                step = sess.run(global_step, feed_dict=None)
                # 訓練
                batch_number = 0
                # 獲取image和label
                index_epoch = sess.run(index_dequeue_op)
                label_epoch = np.array(label_list)[index_epoch]
                image_epoch = np.array(image_list)[index_epoch]

                labels_array = np.expand_dims(np.array(label_epoch), 1)
                image_paths_array = np.expand_dims(np.array(image_epoch), 1)
                # 運行輸入隊列

                print(len(image_paths_array), len(labels_array))

                sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
                while batch_number < epoch_size:

                    feed_dict = {phase_train_placeholder: True, batch_size_placeholder: batch_size,
                                 keep_probability_placeholder: keep_probability}
                    tensor_list = [total_loss, train_op, global_step, learning_rate, prelogits,
                                   cross_entropy_mean, accuracy, prelogits_center_loss]
                    # 每經過5个batch更新一次graph
                    if batch_number % 5 == 0:
                        loss_, _, step_, lr_, prelogits_, cross_entropy_mean_, accuracy_, center_loss_, summary_str = sess.run(
                            tensor_list + [summary_op], feed_dict=feed_dict)
                        train_writer.add_summary(summary_str, global_step=step_)
                        saver.save(sess=sess, save_path=model_dir + 'model.ckpt', global_step=(step_))
                        print('epoch:%d/%d' % (epoch, max_nrof_epochs))
                        print(
                            "Step: %d/%d, accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f ,lr:%f " % (
                            step_, epoch_size * max_nrof_epochs, accuracy_, center_loss_, cross_entropy_mean_, loss_,
                            lr_))
                    else:
                        loss_, _, step_, lr_, prelogits_, cross_entropy_mean_, accuracy_, center_loss_, = sess.run(
                            tensor_list, feed_dict=feed_dict)
                    batch_number += 1
                train_writer.add_summary(summary_str, global_step=step_)
                # 驗證模型
                nrof_val_batches = len(val_label_list) // batch_size
                nrof_val_images = nrof_val_batches * batch_size

                labels_val_array = np.expand_dims(np.array(val_label_list[:nrof_val_images]), 1)
                image_paths_val_array = np.expand_dims(np.array(val_image_list[:nrof_val_images]), 1)
                # 運行驗證集輸入隊列
                sess.run(enqueue_op,
                         {image_paths_placeholder: image_paths_val_array, labels_placeholder: labels_val_array})
                loss_val_mean = 0
                center_loss_val_mean = 0
                cross_entropy_mean_val_mean = 0
                accuracy_val_mean = 0
                for i in range(nrof_val_batches):
                    feed_dict = {phase_train_placeholder: False, batch_size_placeholder: batch_size,
                                 keep_probability_placeholder: 1.0}
                    loss_val, center_loss_val, cross_entropy_mean_val, accuracy_val, summary_val = sess.run(
                        [total_loss, prelogits_center_loss, cross_entropy_mean, accuracy, summary_op],
                        feed_dict=feed_dict)
                    loss_val_mean += loss_val
                    center_loss_val_mean += center_loss_val
                    cross_entropy_mean_val_mean += cross_entropy_mean_val
                    accuracy_val_mean += accuracy_val
                    if i % 10 == 9:
                        print('.', end='')
                        sys.stdout.flush()
                val_writer.add_summary(summary_val, global_step=epoch)
                loss_val_mean /= nrof_val_batches
                center_loss_val_mean /= nrof_val_batches
                cross_entropy_mean_val_mean /= nrof_val_batches
                accuracy_val_mean /= nrof_val_batches
                end_time=time.time()
                process_time= end_time - start_time
                print('\n=================================================================================================')
                print('epoch 運行時間:', process_time, "s")
                print("val: accuracy: %3f, center loss: %4f, cross loss: %4f, Total Loss: %4f " % (
                accuracy_val_mean, center_loss_val_mean, cross_entropy_mean_val_mean, loss_val_mean))
                print('=================================================================================================')
            training_time = time.time()
            total_trianing_time = training_time - start_training_time
            print('訓練總時間:', total_trianing_time, "s")



def center_loss(features, label, alfa, nrof_classes):
    '''計算centerloss
    參數：
      features:網路最終輸出[batch,512]
      label:對應類別標籤[batch,1]
      alfa:center更新比例
      nrof_classes:類別總數
    返回值：
      loss:center_loss損失值
      centers:中心點embeddings
    '''
    # embedding的维度
    nrof_features = features.get_shape()[1]
    centers = tf.compat.v1.get_variable('centers', [nrof_classes, nrof_features],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(0),
                                        trainable=False)
    label = tf.reshape(label, [-1])
    # 挑選出每個batch對應的centers [batch,nrof_features]
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    # 相同類別會累計相減
    centers = tf.compat.v1.scatter_sub(centers, label, diff)
    # 先更新完centers再計算loss
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def optimize(total_loss, global_step, learning_rate, moving_average_decay, update_gradient_vars):
    '''優化參數
    參數：
      total_loss:總損失函數
      global_step：全局step數
      learning_rate: 學習率
      moving_average_decay：指數平均參數
      update_gradient_vars：需更新的參數
    返回值：
      train_op
    '''

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    # 梯度計算
    grads = opt.compute_gradients(total_loss, update_gradient_vars)
    # 應用更新的梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # 參數和梯度分布圖
    for var in tf.compat.v1.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    # 指數平均
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.compat.v1.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op


if __name__ == '__main__':
    main()
