import numpy as np
import torch
from numpy.random import *
import scipy.io as sio
import os
import random
from sklearn import preprocessing
from skimage import exposure
from sklearn.decomposition import PCA
import cv2

def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:',Sample_data.shape)
    nBand = Sample_data.shape[2]

    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    val = {}
    val_indices = []

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices[:num_per_class]
        val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    #val
    print('the number of val data:', len(val_indices))
    nVAL = len(val_indices)
    val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    val_label = np.zeros([nVAL], dtype=np.int64)
    RandPerm = val_indices
    RandPerm = np.array(RandPerm)

    for i in range(nVAL):
        val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    val_label = val_label - 1

    #train
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label#, val_data, val_label

def get_all_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(All_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    return index, processed_data, processed_label, label, RandPerm, Row, Column


def get_all_test_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    All_label += 1
    label = np.pad(All_label, HalfWidth, mode='constant')
    # label += 1
    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   (label[Row[j], Column[j]] == i + 1 and not np.all(data[Row[j]][Column[j]] == 0))]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    All_label -= 1
    label_test = np.pad(All_label, HalfWidth, mode='constant')
    return index, processed_data, processed_label, label_test, RandPerm, Row, Column


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
def seed_everything(seed,use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if use_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True

def cubeData(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1'] #up
    data2 = total['DataCube2'] #pc
    gt1 = total['gt1']
    gt2 = total['gt2']

    # Data_Band_Scaler_s = data1
    # Data_Band_Scaler_t = data2
    # print('max and min ')

    # 归一化 [-0.5,0.5]
    # data1 = data1.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler_s = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))# - 0.5
    #
    # data2 = data2.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler_t = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) #- 0.5

    # # # 标准化
    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])

    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))  # (111104,204)
    data_scaler_t = preprocessing.scale(data_t)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1],data2.shape[2])
    print(np.max(Data_Band_Scaler_s),np.min(Data_Band_Scaler_s))
    print(np.max(Data_Band_Scaler_t),np.min(Data_Band_Scaler_t))
    return Data_Band_Scaler_s,Data_Band_Scaler_t, gt1,gt2  # image:(512,217,3),label:(512,217)

def load_data_hyrank(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    # Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_pavia(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)


    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    return data_all, GroundTruth
    #
    # [nRow, nColumn, nBand] = data_all.shape
    # print(data_key, nRow, nColumn, nBand)
    #
    #
    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])
    #
    # # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # # Data_Band_Scaler = (data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all))
    #
    # # Data_Band_Scaler = data_all
    #
    # print(np.max(Data_Band_Scaler),np.min(Data_Band_Scaler))
    #
    # return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def load_data_houston13(image_file, train_label_file,test_label_file):
    image_data = sio.loadmat(image_file)
    train_label_data = sio.loadmat(train_label_file)
    test_label_data = sio.loadmat(test_label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['data']

    GroundTruth_train = train_label_data['mask_train']

    GroundTruth_test = test_label_data['mask_test']

    # Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth_train,GroundTruth_test # image:(512,217,3),label:(512,217)

def load_data_houston(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_YRD(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['HSI']

    GroundTruth = label_data['GT']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_sh(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['DataCube1']

    GroundTruth = label_data['gt1']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_hz(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['DataCube2']

    GroundTruth = label_data['gt2']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def textread(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines

def adjust_learning_rate(optimizer, epoch,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * 0.99#min(1, 2 - epoch/float(20))#0.95 best
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=0.04):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=10, size=data.shape)
    x = alpha * data + beta * noise
    return x

def pca(data,n):
    "进行pca过程的代码"
    pca = PCA(n_components=n)   # 定义pca的方法
    height, width, channels = data.shape
    data = data.reshape(-1,channels)
    data_PCA = pca.fit_transform(data).reshape(height,width,n)
    Score = pca.explained_variance_ratio_
    # 检验三个主成分是否可以代表
    # print("前三个主成分的贡献程度:")
    # print(Score)
    # 在每个通道上进行归一化操作    有没有可能这里的归一化会把东西搞坏
    min_value = np.min(data_PCA, axis=(0, 1))  # 在前两个维度上找最小值
    max_value = np.max(data_PCA, axis=(0, 1))  # 在前两个维度上找最大值
    data_PCA = (data_PCA - min_value) / (max_value - min_value)
    return data_PCA,Score

def gammaCorrect(SourcePic,TargetPic):
    """
    进行gamma校准的代码
    SourcePic:一个传入源域的三维的图像
    TargetPic:一个传入目标的三维的图像
    """
    # 分别计算源域和目标域的亮度
    # Sourcebrightness1 = 0.299 * SourcePic[:,:,0] + 0.587 * SourcePic[:,:,1] + 0.114 * SourcePic[:,:,2]
    # Targetbrightness1 = 0.299 * TargetPic[:,:,0] + 0.587 * TargetPic[:,:,1] + 0.114 * TargetPic[:,:,2]
    Sourcebrightness1 = np.mean(SourcePic)
    Targetbrightness1 = np.mean(TargetPic)
    # 计算需要应用的Gamma值，以使两个图像的亮度一致
    average_brightness_Source = np.mean(Sourcebrightness1)
    average_brightness_Target = np.mean(Targetbrightness1)
    gamma_value = abs(average_brightness_Source / average_brightness_Target )
    # 对源域和目标域的图像分别应用gamma校准
    Source_corrected = np.power(SourcePic,gamma_value)
    Target_corrected = TargetPic
    Target_corrected = np.clip(Target_corrected, 0, 1)
    # Sourcebrightness2 = (0.299 * Source_corrected[:,:,0]
    #                      + 0.587 * Source_corrected[:,:,1] + 0.114 * Source_corrected[:,:,2])
    # Targetbrightness2 = (0.299 * Target_corrected[:,:,0]
    #                      + 0.587 * Target_corrected[:,:,1] + 0.114 * Target_corrected[:,:,2])
    Sourcebrightness2 = np.mean(Source_corrected)
    Targetbrightness2 = np.mean(Target_corrected)
    print("gamma:"+str(gamma_value))
    print(Sourcebrightness1.mean())
    print(Targetbrightness1.mean())
    print(Sourcebrightness2.mean())
    print(Targetbrightness2.mean())
    Brightness = [Sourcebrightness1.sum(),Targetbrightness1.sum(),
                  Sourcebrightness2.sum(),Targetbrightness2.sum()]
    return Source_corrected,Target_corrected,Brightness

def colorAdaption(Source_image,Target_image):
    # 进行颜色直方图匹配的代码
    Source_image_Matched = exposure.match_histograms(Source_image, Target_image)
    return Source_image_Matched,Target_image

def GuideFilter(Pic_MChannels, Pic_3Channels,r):
    # 进行导向滤波的代码
    Pic_3Channels = Pic_3Channels.astype(np.float32)
    Pic_MChannels = Pic_MChannels.astype(np.float32)
    # 执行导向滤波
    print("1, " + str(r))
    radius = 1  # 滤波半径
    epsilon =  r  # 正则化参数
    filtered_image = cv2.ximgproc.guidedFilter(Pic_3Channels,Pic_MChannels, radius, epsilon)
    return filtered_image


def TotalAdaption(Source_Pic, Target_Pic, pca_n,r):
    """
    进行图像级域迁移的代码
    """
    Source_data, Source_pca_score = pca(Source_Pic, pca_n)
    Target_data, Target_pca_score = pca(Target_Pic, pca_n)
    # Pca之前的操作都是没有问题的
    Source_data_gamma, Target_data_gamma, Brightness = gammaCorrect(Source_data, Target_data)  # 经过gamma校准
    print(Source_data_gamma.shape,Target_data_gamma.shape)
    Source_data_Color, Target_data_Color = colorAdaption(Source_data_gamma, Target_data_gamma)  # 经过颜色直方图匹配
    print(Source_data_Color.shape,Target_data_Color.shape)
    # print("经过gamma校准和颜色直方图匹配之后的效果展示：")
    # fig, axes = plt.subplots(3, 2, figsize=(90, 10))
    # axes[0, 0].imshow(Source_data)
    # axes[0, 0].set_title('源域经过PCA之后的三通道,亮度为' + str(Brightness[0]))  # 设置子图标题
    # axes[0, 1].imshow(Target_data)
    # axes[0, 1].set_title('目标域经过PCA之后的三通道,亮度为' + str(Brightness[1]))  # 设置子图标题
    # axes[1, 0].imshow(Source_data_gamma)
    # axes[1, 0].set_title('源域经过PCA，gamma校准之后的三通道,亮度为' + str(Brightness[2]))  # 设置子图标题
    # axes[1, 1].imshow(Target_data_gamma)
    # axes[1, 1].set_title('目标域经过PCA,gamma校准之后的三通道,亮度为' + str(Brightness[3]))  # 设置子图标题
    # axes[2, 0].imshow(Source_data_Color)
    # axes[2, 0].set_title('源域经过颜色直方图匹配之后的三通道:')  # 设置子图标题
    # axes[2, 1].imshow(Target_data_Color)
    # axes[2, 1].set_title('目标域经过颜色直方图之后的三通道:')  # 设置子图标题
    # plt.savefig('PIC/Gamma.png', dpi=300)
    Final_Source = GuideFilter(Source_Pic, Source_data_Color,r)
    Final_Target = GuideFilter(Target_Pic, Target_data_Color,r)

    return Final_Source, Final_Target

def ILDA(data_s,data_t,pca_n,r):
    # if param is None:
    #     # Pavia数据集
    #     # data_path_s = './datasets/Pavia/paviaU.mat'
    #     # label_path_s = './datasets/Pavia/paviaU_gt_7.mat'
    #     # data_path_t = './datasets/Pavia/pavia.mat'
    #     # label_path_t = './datasets/Pavia/pavia_gt_7.mat'
    #     # Houston数据集
    #     data_path_s = './datasets/Houston/Houston13.mat'
    #     label_path_s = './datasets/Houston/Houston13_7gt.mat'
    #     data_path_t = './datasets/Houston/Houston18.mat'
    #     label_path_t = './datasets/Houston/Houston18_7gt.mat'
    #     data_s,label_s = load_data_houston(data_path_s,label_path_s)
    #     data_t,label_t = load_data_houston(data_path_t,label_path_t)
    # 图像级迁移的执行部分
    ILDA_S,ILDA_T= TotalAdaption(data_s,data_t,pca_n,r)
    return ILDA_S,ILDA_T

