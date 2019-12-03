def lsqnonneg(C, d, x0=None, tol=None, itmax_factor=3):
    '''Linear least squares with nonnegativity constraints.
 
    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''
 
    eps = 2.22e-16    # from matlab
    def norm1(x):
        return abs(x).sum().max()
 
    def msize(x, dim):
        s = x.shape
        if dim >= len(s):
            return 1
        else:
            return s[dim]
 
    if tol is None: tol = 10*eps*norm1(C)*(max(C.shape)+1)
 
    C = np.asarray(C)
 
    (m,n) = C.shape
    P = np.zeros(n)
    Z = np.arange(1, n+1)
 
    if x0 is None: x=P
    else:
        if any(x0 < 0): x=P
        else: x=x0
 
    ZZ=Z
 
    resid = d - np.dot(C, x)
    w = np.dot(C.T, resid)
 
    outeriter=0; it=0
    itmax=itmax_factor*n
    exitflag=1
    # outer loop to put variables into set to hold positive coefficients
    while np.any(Z) and np.any(w[ZZ-1] > tol):
        outeriter += 1
 
        t = w[ZZ-1].argmax()
        t = ZZ[t]
 
        P[t-1]=t
        Z[t-1]=0
 
        PP = np.where(P != 0)[0]+1
        ZZ = np.where(Z != 0)[0]+1
 
        CP = np.zeros(C.shape)
 
        CP[:, PP-1] = C[:, PP-1]
        CP[:, ZZ-1] = np.zeros((m, msize(ZZ, 1)))
 
        z=np.dot(np.linalg.pinv(CP), d)
 
        z[ZZ-1] = np.zeros((msize(ZZ,1), msize(ZZ,0)))
        # inner loop to remove elements from tmodel_path='G:\\jupyter notebook\\20190320_novel_tasks_recognition\\model_LDA'he positve set which no longer belong
        while np.any(z[PP-1] <= tol):
            it += 1
            if it > itmax:
                max_error = z[PP-1].max()
                raise Exception('Exiting: Iteration count (=%d) exceeded\n Try raising the \
                                 tolerance tol. (max_error=%d)' % (it, max_error))
            QQ = np.where((z <= tol) & (P != 0))[0]
            alpha = min(x[QQ]/(x[QQ] - z[QQ]))
            x = x + alpha*(z-x)
            ij = np.where((abs(x) < tol) & (P != 0))[0]+1
            Z[ij-1] = ij
            P[ij-1] = np.zeros(max(ij.shape))
            PP = np.where(P != 0)[0]+1
            ZZ = np.where(Z != 0)[0]+1
            CP[:, PP-1] = C[:, PP-1]
            CP[:, ZZ-1] = np.zeros((m, msize(ZZ, 1)))
            z=np.dot(np.linalg.pinv(CP), d)
            z[ZZ-1] = np.zeros((msize(ZZ,1), msize(ZZ,0)))
        x = z
        resid = d - np.dot(C, x)
        w = np.dot(C.T, resid)
    return (x, sum(resid * resid), resid)
print("5 fingers classification task specific NMF")
import __future__ as division
import scipy.io as sio
import numpy as np
import math
from scipy.optimize import leastsq
from sklearn.decomposition import NMF
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
    for user_id in range(len(user_name)):
        print('**********user_{}**********'.format(user_id))
        f = sio.loadmat(data_path+user_name[user_id]+'//featureData_TD4')
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

            W_mat = []
            temp_sam_id = 0
            for finger_state in range(2):
                model = NMF(n_components=25)
                temp_data = train_data[np.where(train_label_detail[:,finger_id]==finger_state)\
                                       ,:,:,:].reshape(-1,rows*columns*channels).T
                W = model.fit_transform(temp_data)
                W_mat.append(W)

            train_pred = []
            temp_sam_id = 0
            for movement_id in range(len_class):
                temp_data = train_data[temp_sam_id:temp_sam_id+train_movement_num[movement_id],\
                                       :,:,:].reshape(-1,rows*columns*channels).T
                for sam_id in range(temp_data.shape[1]):
                    data = temp_data[:,sam_id].reshape([rows*columns*channels,])
                    cls_like = []
                    for class_id in range(2):
                        [h_sim,_,_] = lsqnonneg(W_mat[class_id],data)
                        v_sim = np.dot(W_mat[class_id],h_sim)
                        dist = np.linalg.norm(data-v_sim)
                        cls_like.append(dist)
                    train_pred.append(np.argmin(cls_like))
                temp_sam_id += train_movement_num[movement_id]
            train_pred = np.array(train_pred)
            train_pred_detail[:,finger_id] = train_pred

            test_pred = []
            temp_sam_id = 0
            for movement_id in range(len_class):
                temp_data = test_data[temp_sam_id:temp_sam_id+test_movement_num[movement_id],:,:,:].reshape(-1,rows*columns*channels).T
                for sam_id in range(temp_data.shape[1]):
                    data = temp_data[:,sam_id].reshape([rows*columns*channels,])
                    cls_like = []
                    for class_id in range(2):
                        [h_sim,_,_] = lsqnonneg(W_mat[class_id],data)
                        v_sim = np.dot(W_mat[class_id],h_sim)
                        dist = np.linalg.norm(data-v_sim)
                        cls_like.append(dist)
                    test_pred.append(np.argmin(cls_like))
                temp_sam_id += test_movement_num[movement_id]
            test_pred = np.array(test_pred)
            test_pred_detail[:,finger_id] = test_pred


        print("training accuracy  ")
        print(np.sum([(x==y).all() for x,y in zip(train_pred_detail,train_label_detail)])/train_data.shape[0])
        print("testing accuracy  ")
        print(np.sum([(x==y).all() for x,y in zip(test_pred_detail,test_label_detail)])/test_data.shape[0])
        all_user_acc.append(np.sum([(x==y).all() for x,y in zip(test_pred_detail,test_label_detail)])/test_data.shape[0])
print('mean accuracy:{}+/-{}'.format(np.mean(all_user_acc),np.std(all_user_acc)))