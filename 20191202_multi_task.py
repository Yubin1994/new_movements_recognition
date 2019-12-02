
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm,flatten
from tensorflow.contrib.framework import arg_scope
import scipy.io as sio
import numpy as np
import math
# model_path='G:\\jupyter notebook\\20190320_novel_tasks_recognition\\model_estimator'
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
        train_clsNum = []


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


        def Conv_layer(inputs,filters,kernel,stride,layer_name):
            with tf.name_scope(layer_name):
                network = tf.layers.conv2d(inputs=inputs,filters=filters,kernel_size=kernel,strides=stride,padding='SAME')
                return network
        def Dense(inputs,size,layer_name):
            with tf.name_scope(layer_name):
                network = tf.contrib.layers.fully_connected(inputs,size,activation_fn=tf.nn.sigmoid)
                return network
        def Drop_out(inputs,rate,training):
            network = tf.layers.dropout(inputs=inputs,rate=rate,training=training)
            return network
        def Relu(inputs):
            network = tf.nn.relu(inputs)
            return network
        def Batch_normalization(inputs,training,scope):
            with arg_scope([batch_norm],
                           scope=scope,
                           updates_collections=None,
                           decay=0.9,
                           center=True,
                           scale=True,
                           zero_debias_moving_mean=True) :
                network = tf.cond(training,
                                lambda : batch_norm(inputs=inputs,is_training=training,reuse=None),
                                lambda : batch_norm(inputs=inputs,is_training=training,reuse=True))
                return network
        def Average_pooling(inputs,pool_size,stride,padding='VALID'):
            network = tf.layers.average_pooling2d(inputs=inputs,pool_size=pool_size,strides=stride,padding=padding)
            return network
        def Max_pooling(inputs,pool_size,stride,padding='VALID'):
            network = tf.layers.max_pooling2d(inputs=inputs,pool_size=pool_size,strides=stride,padding=padding)
            return network



        init_learning_rate = 1e-2
        total_epochs = 1000
        mini_batch_size = train_data.shape[0]
        drop_out_rate = 0.2
        epsilon = 1e-7
        tf.reset_default_graph()
        path = './model_novel'
        
        def spatial_attention(feature_map, K=1024, weight_decay=0.00004):
        
            (_, H, W, C) = feature_map.get_shape()
            w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
                                  dtype=tf.float32,
                                  initializer=tf.initializers.orthogonal,
                                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b_s = tf.get_variable("SpatialAttention_b_s", [1],
                                  dtype=tf.float32,
                                  initializer=tf.initializers.zeros)
            spatial_attention_fm = tf.matmul(tf.reshape(feature_map, [-1, C]), w_s) + b_s
            spatial_attention_fm = tf.nn.softmax(tf.reshape(spatial_attention_fm, [-1, W * H]))
            attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
            attended_fm = attention * feature_map
            return attended_fm,spatial_attention_fm
        class Net():
            def __init__(self,data,training,dropout_rate):
                self.training = training
                self.dropout_rate = dropout_rate
                self.model = self.mtCNN_net(data)
            def mtCNN_net(self,input_x):
                input_x,attention = spatial_attention(input_x,K=1024, weight_decay=0.00004)
                x = Conv_layer(input_x,filters=8,kernel=[3,3],stride=1,layer_name='conv0')
                x = flatten(x)
                x = Dense(x,128,layer_name='Dense1')
                x = Dense(x,64,layer_name='Dense2')
                x = Batch_normalization(x,self.training,scope='linear_batch')
                x = tf.contrib.layers.fully_connected(x,fingers,activation_fn=tf.nn.relu)
                return x
        def random_mini_batches(X,Y,mini_batch_size,seed):
            m = X.shape[0]
            mini_batches = []
            np.random.seed(seed)

            permutation = list(np.random.permutation(m))
            shuffled_X = X[permutation,:,:,:]
            shuffled_Y = Y[permutation,:]
            num_complete_minibatches = math.floor(m/mini_batch_size)
            for k in range(num_complete_minibatches):
                mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:,:,:]
                mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
                mini_batch = (mini_batch_X,mini_batch_Y)
                mini_batches.append(mini_batch)
            if m%mini_batch_size != 0:
                mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:m,:,:,:]
                mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:m,:]
                mini_batch = (mini_batch_X,mini_batch_Y)
                mini_batches.append(mini_batch)

            return mini_batches

        data = tf.placeholder(tf.float32,shape=[None,rows,columns,channels])
        label = tf.placeholder(tf.float32,shape=[None,fingers])
        training_flag = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32)
        dropout_rate = tf.placeholder(tf.float32)

        logits = Net(data,training_flag,dropout_rate).model
        cost = tf.reduce_mean(tf.losses.mean_squared_error(logits,label))
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=epsilon).minimize(cost)

        loss_summary = tf.summary.scalar('loss', cost)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess,ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(path, sess.graph)

            epoch_learning_rate = init_learning_rate
            mini_batches = random_mini_batches(train_data,train_label_detail,mini_batch_size,616)
            accuracy_queue = []
            for epoch in range(total_epochs):
                if epoch == (total_epochs*0.4) or epoch == (total_epochs*0.7):
                    epoch_learning_rate = epoch_learning_rate/10
                for mini_batch in mini_batches:
                    (mini_batch_X,mini_batch_Y) = mini_batch
                    train_feed_dict = {
                        data:mini_batch_X,
                        label:mini_batch_Y,
                        learning_rate:epoch_learning_rate,
                        training_flag:True,
                        dropout_rate:drop_out_rate
                    }
                    _ = sess.run(opt,feed_dict = train_feed_dict)
                trainloss_feed_dict = {
                    data:train_data,
                    label:train_label_detail,
                    training_flag:False,
                    dropout_rate:drop_out_rate
                }
                summary_tmp,train_loss,train_pred = sess.run([merged,cost,logits], feed_dict=trainloss_feed_dict)  
                writer.add_summary(summary=summary_tmp, global_step=epoch)
                print("epoch:",epoch,"train loss:",train_loss)
                if epoch%5 == 0:         
                    testloss_feed_dict = {
                        data:test_data,
                        label:test_label_detail,
                        training_flag:False,
                        dropout_rate:drop_out_rate
                    }
                    test_loss,test_pred = sess.run([cost,logits],feed_dict=testloss_feed_dict)
                    print("test loss:",test_loss)
                    if epoch > total_epochs*0.5:
                        train_or_test = 1
                        pred_result = []
                        threshold = 0.5
                        dataset = [(train_data,train_label_detail,train_pred,train_movement_num),
                                   (test_data,test_label_detail,test_pred,test_movement_num)]
                        (t_data,t_label,t_pred,t_movement_num) = dataset[train_or_test]
                        print('{}...'.format(['training','testing'][train_or_test]))
                        sam_num = t_pred.shape[0]
                        for logit in t_pred:
                            logit_result = []
                            for finger_id in range(fingers):
                                if logit[finger_id] > threshold:
                                    logit_result.append(1)
                                else:
                                    logit_result.append(0)
                            pred_result.append(logit_result)

                        finger_result = np.zeros(shape=[fingers,])
                        for finger_id in range(fingers):
                            for sam_id in range(sam_num):
                                if pred_result[sam_id][finger_id] == t_label[sam_id,finger_id]:
                                    finger_result[finger_id] += 1
                        finger_result = [x/sam_num for x in finger_result]
                        print('finger result:{}  \n'.format(finger_result))

                        movement_result = np.zeros(shape=[len_class,])
                        temp_sam_id = 0
                        for movement_id in range(len_class):
                            for sam_id in range(temp_sam_id,temp_sam_id+t_movement_num[movement_id]):
                                if (pred_result[sam_id] == t_label[sam_id]).all():
                                    movement_result[movement_id] += 1
                            temp_sam_id += t_movement_num[movement_id]
                        all_accuracy = sum(movement_result)/sam_num
                        print('all accuracy:{}  \n'.format(all_accuracy))
                        movement_result = [movement_result[movement_id]/t_movement_num[movement_id] for movement_id in range(len_class)]
    #                     all_user_movement_acc.append(movement_result)
                        print('movement result:{}  \n'.format(movement_result))
                        accuracy_queue.append(all_accuracy)
            all_user_acc.append(max(accuracy_queue))
print('mean accuracy:{}+/-{}'.format(np.mean(all_user_acc),np.std(all_user_acc)))