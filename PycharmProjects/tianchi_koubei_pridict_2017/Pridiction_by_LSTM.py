# -*-coding:gbk-*-
# -*-coding:utf-8-*-
# ��tensorflow�İ汾��0.10.0

from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
import config as conf  # ������һ��config.py�ļ�
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn  # ʹ��tensorflow�汾��0.10.0
from tensorflow.contrib import layers as tflayers
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def Divide_data(data_file):
    '''
    Divide_data(): �����ݷָ�ɡ�ѵ����/��֤��/���Լ�������������,�б�����
    ��������ļ��Ѵ��������,��������ļ��Ļ��ֲ������ֺ���ļ����б���.
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def divide_data():
        with open("data/process_data/user_pay_del_all_zero.csv", "r") as f:
            # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
            # split = [0.8, 0.1, 0.1]
            # shop_ids_exist = check_shop_id()
            for line in f:  # ÿһ����һ���̼ң���2000�У�����,����2000����������ϼ�Ԥ��
                context = line.replace("\n", "").split(",")
                # if shop_id in shop_ids_exist:  # �ж�ĳ���̼��Ƿ��Ѿ���Ԥ����,������������Ԥ��
                #     continue
                shop_id = context[0]
                # listing_data = [int(i) for i in listing_data]
                listing_data = [int(i) for i in context[1:] if len(i)>0]
                listing_data = list(np.array(listing_data)/float(max(listing_data)))  # Ԥ������һ��
                pro = 0.7
                len_train = int(len(listing_data) * pro)
                len_vild = int(len(listing_data) * 0.1)
                len_test = len(listing_data) - len_train - len_vild

                while len_test < 16:
                    pro -= 0.1
                    len_train = int(len(listing_data) * pro)
                    len_vild = int(len(listing_data) * (1 - pro) / 2)
                    len_test = len(listing_data) - len_train - len_vild
                data_file1 = 'data/process_data/npz_del_all_zero/norm_user_pay_{}.npz'.format(shop_id)
                data_file2 = 'data/process_data/npz_del_all_zero/norm_diff_user_pay_{}.npz'.format(shop_id)

                # �������δ�з�,��ִ�������з����
                seq1 = [[], [], []]
                seq2 = [[], [], []]
                lastAc = -1
                for i in range(len_train):  # ��һ��������ݷָ�
                    try:
                        ac = listing_data[i]
                        seq1[0].append(ac)
                    except KeyError:
                        pass
                for i in range(len_vild):
                    try:
                        ac = listing_data[i + len_train]
                        seq1[1].append(ac)
                    except KeyError:
                        pass
                for i in range(len_test):
                    try:
                        ac = listing_data[i + len_train + len_vild]
                        seq1[2].append(ac)
                    except KeyError:
                        pass
                datasets1 = [np.asarray(dat, dtype=np.float32) for dat in seq1]
                print('Saving in {}'.format(data_file1))
                np.savez(data_file1, *datasets1)
                for i in range(len_train):  # ��һ����,�ٲ�ִ��������ݷָ�
                    try:
                        ac = listing_data[i]
                        daily_return = (ac - lastAc)  # һ��һ�ײ��
                        # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��;  ��ԭ��ʽ��ac = lastAc*daily_return+lastAc
                        # if len(daily_returns) == daily_returns.maxlen:
                        #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                        # daily_returns.append(daily_return)
                        lastAc = ac
                        seq2[0].append(daily_return)  # һ��һ�ײ�����ݣ�ԭʼ���������������Ч���ܲ
                        # seq[0].append(ac)  # ԭʼ����
                    except KeyError:
                        pass
                for i in range(len_vild):
                    try:
                        ac = listing_data[i + len_train]
                        daily_return = (ac - lastAc)  # һ��һ�ײ��
                        # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
                        # if len(daily_returns) == daily_returns.maxlen:
                        #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                        # daily_returns.append(daily_return)
                        lastAc = ac
                        seq2[1].append(daily_return)  # һ��һ�ײ������
                        # seq[1].append(ac)  # ԭʼ����
                    except KeyError:
                        pass
                for i in range(len_test):
                    try:
                        ac = listing_data[i + len_train + len_vild]
                        daily_return = (ac - lastAc)  # һ��һ�ײ��
                        # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
                        # if len(daily_returns) == daily_returns.maxlen:
                        #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
                        # daily_returns.append(daily_return)
                        lastAc = ac
                        seq2[2].append(daily_return)  # һ��һ�ײ������
                        # seq[2].append(ac)  # ԭʼ����
                    except KeyError:
                        pass
                datasets2 = [np.asarray(dat, dtype=np.float32) for dat in seq2]
                print('Saving in {}'.format(data_file2))
                np.savez(data_file2, *datasets2)
    if not os.path.exists(data_file):
        divide_data()  # ���δ�ҵ������ļ�������ļ��Ļ���
    with np.load(data_file) as file_load:
        datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
    return datasets


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:  # labels==True:��ʾ�ڻ���y,ÿһ������(����Ϊtime_steps)��Ӧ(ӳ��/Ԥ��)һ��y
            try:
                rnn_df.append(data[i + time_steps])
            except AttributeError:
                rnn_df.append(data[i + time_steps])
        else:  # labels==False:��ʾ�ڻ���X(����),���ظ��Ļ��֣���һλ�������֣�
            data_ = data[i: i + time_steps]
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


def rnn_data_for_predict(data, time_steps):
    """
    # ��ȡ���һ��(����)X,������ʵԤ��.
    """
    rnn_df = []
    data_ = data[-time_steps:]
    rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df, dtype=np.float32)


def prepare_data(time_steps, data_file, labels=False):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = Divide_data(data_file)  # �ָ����ݼ�:��һά����(narray)��ʽ���
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def generate_data(time_steps, data_file):
    train_x, val_x, test_x = prepare_data(time_steps, data_file)
    train_y, val_y, test_y = prepare_data(time_steps, data_file, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def lstm_model(time_steps, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    """
    Creates a deep model based on:
        * stacked lstm cells
        * an optional dense layers
    :param time_steps: the size of the cells.
    :param rnn_layers: list of int or dict
                         * list of int: the steps used to instantiate the `BasicLSTMCell` cell
                         * list of dict: [{steps: int, keep_prob: int}, ...]
    :param dense_layers: list of nodes for each layer
    :return: the model definition
    """

    def lstm_cells(layers):
        if isinstance(layers[0], dict):
            return [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                               state_is_tuple=True),
                                                  layer['keep_prob'])
                    if layer.get('keep_prob') else tf.nn.rnn_cell.BasicLSTMCell(layer['num_units'],
                                                                                state_is_tuple=True)
                    for layer in layers]
        return [tf.nn.rnn_cell.BasicLSTMCell(steps, state_is_tuple=True) for steps in layers]

    def dnn_layers(input_layers, layers):
        if layers and isinstance(layers, dict):
            return tflayers.stack(input_layers, tflayers.fully_connected,
                                  layers['layers'],
                                  activation=layers.get('activation'),
                                  dropout=layers.get('dropout'))
        elif layers:
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        x_ = tf.unpack(X, axis=1, num=time_steps)
        output, layers = tf.nn.rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)
        prediction, loss = tflearn.models.linear_regression(output, y)
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.contrib.framework.get_global_step(), optimizer=optimizer,
            learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model


def check_shop_id(prediction_file):  # ��ȡ��Щ�̼���Ԥ����
    shop_ids = []
    if os.path.exists(prediction_file):
        with open(prediction_file, "r") as f_pri:
            for id in f_pri:
                shop_id = id.replace("\n", "").split(",")[0]
                shop_ids.append(shop_id)
    return shop_ids


def lists_Intersection(a_list, b_list):  # ���������б�Ľ���
    return list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))


def pri_weekday_weekend(pri_ids):
    # Ԥ����Щ�������պ���ĩ���Ѳ�����̼�,����5��������֮�����Ѻܽӽ�,��ĩ2���������.���͵Ľڼ��յ���
    # pri_ids:list����
    with open("data/process_data/user_pay_del_all_zero.csv","r") as f:
        with open("data/predict_data/repri_big_L.csv","w") as fw:
            for line in f:
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = [int(i) for i in context[1:] if len(i) > 0]
                if int(shop_id) in pri_ids:
                    pri_weekdays = int(np.mean(np.array(data[-8:-3])+np.array(data[-15:-10]))/2)  # �������ʮ�������յ����Ѿ�ֵ
                    pri_weekend = int(np.mean(np.array(data[-3:-1])+np.array(data[-10:-8]))/2)  # ������ܵ���ĩ�����Ѿ�ֵ
                    fw.write(shop_id+",")
                    fw.write(",".join([str(pri_weekdays) for i in range(4)])+",")  # �ܶ������ģ�Ԥ��ĵ�һ�����ܶ���
                    fw.write(",".join([str(pri_weekend) for i in range(2)])+",")  # ��ĩ
                    fw.write(",".join([str(pri_weekdays) for i in range(5)])+",")  # ��һ������
                    fw.write(",".join([str(pri_weekend) for i in range(2)])+",")  # ��ĩ
                    fw.write(",".join([str(pri_weekdays) for i in range(1)])+"\n")  # ��һ��Ԥ������һ������һ��


"""������ʼ��"""
start_time = time.time()
LOG_DIR = './ops_logs/traffic_flow1'  # ����ѵ���õõ���ģ�Ͳ�����ͼ���
TIMESTEPS = 7
# RNN_LAYERS = [{'num_units': 600}, {'num_units': 500}]  # ��ʾ�����������ز�(LSTM),���ز��ڵ�������Ԫ�����ֱ�Ϊ600,500
RNN_LAYERS = [{'num_units': 600}, {'num_units': 400}]
DENSE_LAYERS = None
TRAINING_STEPS = 100
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 20

data_format = "raw"  # ������ƴ��� �������ƣ�raw/diff/diff_normalize:���ִ���ԭ���ݷ�ʽ

# with open("data/predict_data/Neg_ids.csv", "r") as f:
#     ids = f.readline().replace("\n", "").split(",")

with open("data/process_data/user_pay_del_all_zero.csv", "r") as f:
    # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
    # split = [0.8, 0.1, 0.1]
    count_shop = 0
    shop_ids_exist = check_shop_id("data/predict_data/prediction.csv")
    for line in f:  # ÿһ����һ���̼ң���2000�У�����,����2000����������ϼ�Ԥ��
        shop_id = line.replace("\n", "").split(",")[0]
        data = line.replace("\n", "").split(",")[1:]
        data = [int(i) for i in data if len(i)>0]
        max_data = max(data)
        if shop_id in shop_ids_exist:  # �ж�ĳ���̼��Ƿ��Ѿ���Ԥ����,������������Ԥ��
            continue
        # if int(shop_id) not in ids:
        #     continue
        data_file1 = 'data/process_data/npz_del_all_zero/norm_user_pay_{}.npz'.format(shop_id)
        data_file2 = 'data/process_data/npz_del_all_zero/norm_diff_user_pay_{}.npz'.format(shop_id)

        # ����ģ�͹�����
        # regressor = tflearn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS), model_dir=LOG_DIR)  # �õ�ѵ�����ģ��ʵ����lstm_model(��Ԥ��ֵ/��ʧֵ/�Ż�������)
        regressor = tflearn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS))

        # �ָ�"����(X)"��"���(Y)"��Ӧ�����ݼ�
        if data_format == "raw":
            X, y = generate_data(TIMESTEPS, data_file1)  # ��ԭʼ����(û������ִ���)
        elif data_format == "diff":
            X, y = generate_data(TIMESTEPS, data_file2)
        # elif data_format == "diff_normalize":
        #     X, y = generate_data(TIMESTEPS, data_file3)

        # ������֤��������,create a lstm instance and validation monitor
        validation_monitor = tflearn.monitors.ValidationMonitor(X['val'], y['val'], every_n_steps=PRINT_STEPS,
                                                                early_stopping_rounds=1000)
        # print("X[train]:", X['train'][:5])
        # print("y[train]", y['train'][:5])

        # ģ�����
        regressor.fit(X['train'], y['train'],  monitors=[validation_monitor],
                      batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
        # ģ��Ԥ��
        predicted = regressor.predict(X['test'])  # �������ݼ�Ԥ��

        # ���ݻ�ԭǰԤ�����ݵĸ���ָ��,���������RMSE
        RMES = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
        MAPE = np.mean(np.abs((predicted - y['test'])/y['test']))
        print ("RMES:%f - MAPE: %f" % (RMES, MAPE))


        # ���ݻ�ԭ����
        with np.load(data_file1) as file_load:  # ���ع�һ�����ԭʼ����
            orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        priTest_data = []  # Ԥ������"len(test_data)-TIMESTEPS"������,�����
        for i, data in enumerate(predicted):  # enumerate:������������������ֵ
            if data_format == "raw":  # ��һ�����ԭʼ����,û������ִ���,���ػ�ԭ����
                priTest_data.append(data)
            elif data_format == "diff":  # "���"��Ļ�ԭ����,��ԭ��ʽ��orig_x = predict + orig_last_X
                priTest_data.append(data + orig_test_data[i + TIMESTEPS - 1])  # ��ԭ: TIMESTEPS����������Ԥ����һ������
            elif data_format == "diff_normalize":
                priTest_data.append((data + 1) * orig_test_data[i + TIMESTEPS - 1])
            else:
                print ("���ݻ�ԭʱ����Դ�����ڣ�")
        priTest_data = list(np.array(priTest_data) * max_data)  # ����һ��

        # ��ԭ��Ԥ�����ݵĸ���ָ��
        L = np.mean(abs(np.array(priTest_data) - (np.array(orig_test_data[TIMESTEPS:])*max_data))/(np.array(priTest_data) + (np.array(orig_test_data[TIMESTEPS:])*max_data)))
        pri_tRMSE = np.sqrt(np.mean((np.array(priTest_data) - np.array(orig_test_data[TIMESTEPS:])*max_data) ** 2))  # ���������RMSE:ƽ���������ݵ�ʵ��ֵ��Ԥ��ֵ֮���ƫ��
        pri_tMAPE = np.mean(np.abs(np.array(priTest_data) - np.array(orig_test_data[TIMESTEPS:])*max_data) / (np.array(orig_test_data[TIMESTEPS:])*max_data))
        print("TestData pri_tRMSE: %.3f - pri_tMAPE: %.3f - L=%.6f" % (pri_tRMSE, pri_tMAPE, L))

        # ���ӻ�
        # fig = plt.figure(figsize=(12, 8))  # ����
        # ax1 = fig.add_subplot(211)  # ����ϵ
        # ax2 = fig.add_subplot(212)
        # plot_predicted, = ax1.plot(predicted[-30:], marker="*", ms=6.0, linewidth=0.5, label='predicted', color="r")
        # plot_test, = ax1.plot(y['test'][-30:], marker="o", ms=6.0, linewidth=0.5, label='test', color="b")
        # plot_orig_predicted, = ax2.plot(priTest_data[-30:], marker="*", ms=6.0, linewidth=0.5, label='orig_predicted', color="r")
        # plot_orig_test, = ax2.plot(orig_test_data[-30:], marker="o", ms=6.0, linewidth=0.5, label='orig_test', color="b")
        # ax1.legend(handles=[plot_predicted, plot_test])  # ���������Ϸ�����label
        # ax2.legend(handles=[plot_orig_predicted, plot_orig_test])
        # plt.show()

        """��ʵ����Ԥ��(Ԥ��δ��ĳ����)"""
        x = list(y['test'])
        predict_y = []
        for i in range(14):
            predict_x = rnn_data_for_predict(np.array(x), TIMESTEPS)  # ��дrnn_data()�����Դ�����Ҫ��
            predict_y1 = regressor.predict(predict_x)
            predict_y.append(predict_y1)
            x.append(predict_y1[0])
        # ���Ԥ��ǰ�����Ǿ�����ִ����,����Ҫ�������ݻ�ԭ����
        with np.load(data_file1) as file_load:  # ����ԭʼ����
            orig_train_data, orig_valid_data, orig_test_data = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        pri_data = []  # Ԥ��δ��n�������
        orig_test_data = list(orig_test_data)
        for i, data in enumerate(predict_y):  # enumerate:������������������ֵ
            if data_format == "raw":  # ԭʼ����,û������ִ���,���ػ�ԭ����
                pri_data.append(data)
            elif data_format == "diff":  # "���"��Ļ�ԭ����,��ԭ��ʽ��orig_x = predict + orig_last_X
                pri_data.append(data + orig_test_data[-1])
                orig_test_data.append(pri_data[-1])

            elif data_format == "diff_normalize":
                pri_data.append((data + 1) * orig_test_data[- 1])
                orig_test_data.append(pri_data[-1])
            else:
                print ("���ݻ�ԭʱ����Դ�����ڣ�")
        pri_data = list(np.array(pri_data) * max_data)  # ����һ��
        pri_data = [str(int(i[0])) for i in pri_data]  # pri_data�е�Ԫ����narray����,ת��Ϊ����

        with open("data/predict_data/prediction.csv", "a") as fw:
            fw.write(shop_id+",")
            fw.write(",".join(pri_data)+","+str(L)+"\n")
        count_shop += 1
        if count_shop % 100 == 0:
            print ("The %dth shop" % count_shop)
        end_time = time.time()
        print("cost time: %dminutes" % (int(end_time - start_time) / 60))
