'''
@Descripttion: 
@Path: ~/anaconda3/envs/tensorflow/bin/python
@Coding: _*_UTF-8 _*_
@Version: python3.6.9  tensorflow-gpu1.14
@Author: yubin
@Date: 2019-12-03 19:47:09
'''
print("5 fingers classification")
import __future__ as division
import scipy.io as sio
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

data_path = '/home/yubin/桌面/novel_movement_recognition/novel movements classification/preprocessing/mat files/'
user_name = ['1202_chenyiheng','1202_zhangxuan','1212_wangkun','1214_lixinhui','1217_chenxi','1222_zhuge','1223_lingyu']
motion_name = ['A1','A12','A123','A1234','A12345','A2345','A345','A45','A5','A2','A3','A4','A15','A23','A34']
channels = 4
len_class = len(motion_name)
rows = 6
columns = 8
fingers=5
repitition_num = 4
all_user_acc = []
all_user_movement_acc = []
for test_repitition_id in range(repitition_num):
    print("test repitition {}".format(test_repitition_id))
    for user_id in range(len(user_name)):
        print('**********user_{}**********'.format(user_id))
        f = sio.loadmat(data_path+user_name[user_id]+'//featureData_nor2_TD4')
        feature_data = f['featureData']
        train_data = np.empty(shape=[0,rows,columns,channels])
        train_label = np.empty(shape=[0,1])
        test_data = np.empty(shape=[0,rows,columns,channels])
        test_label = np.empty(shape=[0,1])
        train_movement_num = np.zeros(shape=[len_class,]).astype(int)
        test_movement_num = np.zeros(shape=[len_class,]).astype(int)
        train_temp_num = 0
        test_temp_num = 0
        for movement_id in range(len_class):
            for repitition_id in range(repitition_num):
                temp_data = feature_data[0,0][motion_name[movement_id]][0,repitition_id]
                temp_label = movement_id * np.ones(shape=[temp_data.shape[0],1])
                if repitition_id != test_repitition_id:
                    train_data = np.concatenate((train_data,temp_data),axis=0)
                    train_label = np.concatenate((train_label,temp_label),axis=0)
                else:
                    test_data = np.concatenate((test_data,temp_data),axis=0)
                    test_label = np.concatenate((test_label,temp_label),axis=0)
            train_movement_num[movement_id] = np.int(train_data.shape[0]-train_temp_num)
            train_temp_num = train_data.shape[0]
            test_movement_num[movement_id] = np.int(test_data.shape[0]-test_temp_num)
            test_temp_num = test_data.shape[0]

        label_mat={0:[1,0,0,0,0],1:[1,1,0,0,0],2:[1,1,1,0,0],3:[1,1,1,1,0],4:[1,1,1,1,1],5:[0,1,1,1,1],6:[0,0,1,1,1],
          7:[0,0,0,1,1],8:[0,0,0,0,1],9:[0,1,0,0,0],10:[0,0,1,0,0],11:[0,0,0,1,0],12:[1,0,0,0,1],13:[0,1,1,0,0],14:[0,0,1,1,0]}
        train_label_detail = np.empty(shape=[0,fingers])
        for label_id in train_label:
            train_label_detail=np.concatenate((train_label_detail,np.array([label_mat[label_id[0]]])),axis=0)
        test_label_detail = np.empty(shape=[0,fingers])
        for label_id in test_label:
            test_label_detail=np.concatenate((test_label_detail,np.array([label_mat[label_id[0]]])),axis=0)

        train_pred_detail = np.zeros(shape=[train_data.shape[0],fingers])
        test_pred_detail = np.zeros(shape=[test_data.shape[0],fingers])
        for finger_id in range(fingers):
            print("*****finger_{}*****".format(finger_id))
            clf = LinearDiscriminantAnalysis()
            clf.fit(train_data.reshape(train_data.shape[0],-1),np.squeeze(train_label_detail[:,finger_id]).astype(np.int32))
            train_pred = clf.predict(train_data.reshape(train_data.shape[0],-1))
            train_pred_detail[:,finger_id] = train_pred
            test_pred = clf.predict(test_data.reshape(test_data.shape[0],-1))
            test_pred_detail[:,finger_id] = test_pred

        print("training accuracy  ")
        print(np.sum([(x==y).all() for x,y in zip(train_pred_detail,train_label_detail)])/train_data.shape[0])
        print("testing accuracy  ")
        print(np.sum([(x==y).all() for x,y in zip(test_pred_detail,test_label_detail)])/test_data.shape[0])
        all_user_acc.append(np.sum([(x==y).all() for x,y in zip(test_pred_detail,test_label_detail)])/test_data.shape[0])
print('mean accuracy:{}+/-{}'.format(np.mean(all_user_acc),np.std(all_user_acc)))