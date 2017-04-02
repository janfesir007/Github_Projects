# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""��ԭʼ��ʱ���������ݻ��ֳɣ�ѵ����/��֤��/���Լ�,������X-Y�����ݶ�Ҳ���ֺã�����ֱ��Ͷ��ģ��ѵ��"""
import numpy as np


def divide_data(data_set_path):

    """
        Divide_data(): �����ݷָ�ɡ�ѵ����/��֤��/���Լ�������������,�б�����
        data_set_path�ϵġ�ʱ�����С��ǡ�������(�������һ��),���������С�
        find date range for the split train, val, test (0.7, 0.1, 0.2 of total days)
    """
    with open(data_set_path, "r") as f:
        list_data = []
        first_line = True
        for line in f:
            context = line.replace("\n", "").split(",")
            if first_line:  # ȥ����һ��---�ֶ���
                first_line = False
                continue
            list_data.append(int(context[-1]))
    # list_data = list(np.array(list_data)/float(max(list_data)))  # Ԥ������һ��
    pro = 0.7
    len_train = int(len(list_data) * pro)
    len_vild = int(len(list_data) * 0.1)
    len_test = len(list_data) - len_train - len_vild
    # while len_test < 16:
    #     pro -= 0.1
    #     len_train = int(len(list_data) * pro)
    #     len_vild = int(len(list_data) * (1 - pro) / 2)
    #     len_test = len(list_data) - len_train - len_vild
    seq1 = [[], [], []]
    for i in range(len_train):  # ���ݷָ�
        try:
            ac = list_data[i]
            seq1[0].append(ac)
        except KeyError:
            pass
    for i in range(len_vild):
        try:
            ac = list_data[i + len_train]
            seq1[1].append(ac)
        except KeyError:
            pass
    for i in range(len_test):
        try:
            ac = list_data[i + len_train + len_vild]
            seq1[2].append(ac)
        except KeyError:
            pass
    data_3d = [np.asarray(dat, dtype=np.float32) for dat in seq1]
    # lastAc = -1
    # seq2 = [[], [], []]
    # for i in range(len_train):  # ��ִ��������ݷָ�
    #     try:
    #         ac = list_data[i]
    #         daily_return = (ac - lastAc)  # һ��һ�ײ��
    #         # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��;  ��ԭ��ʽ��ac = lastAc*daily_return+lastAc
    #         # if len(daily_returns) == daily_returns.maxlen:
    #         #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
    #         # daily_returns.append(daily_return)
    #         lastAc = ac
    #         seq2[0].append(daily_return)  # һ��һ�ײ�����ݣ�ԭʼ���������������Ч���ܲ
    #         # seq[0].append(ac)  # ԭʼ����
    #     except KeyError:
    #         pass
    # for i in range(len_vild):
    #     try:
    #         ac = list_data[i + len_train]
    #         daily_return = (ac - lastAc)  # һ��һ�ײ��
    #         # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
    #         # if len(daily_returns) == daily_returns.maxlen:
    #         #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
    #         # daily_returns.append(daily_return)
    #         lastAc = ac
    #         seq2[1].append(daily_return)  # һ��һ�ײ������
    #         # seq[1].append(ac)  # ԭʼ����
    #     except KeyError:
    #         pass
    # for i in range(len_test):
    #     try:
    #         ac = list_data[i + len_train + len_vild]
    #         daily_return = (ac - lastAc)  # һ��һ�ײ��
    #         # daily_return = (ac - lastAc) / lastAc  # һ��һ�ײ�ֱ�׼��
    #         # if len(daily_returns) == daily_returns.maxlen:
    #         #     seq[idx].append(daily_return / np.std(daily_returns))  # һ��һ�ײ�ֱ�׼����,�ٽ������ݳ���Ϊ50�ı�׼���һ��
    #         # daily_returns.append(daily_return)
    #         lastAc = ac
    #         seq2[2].append(daily_return)  # һ��һ�ײ������
    #         # seq[2].append(ac)  # ԭʼ����
    #     except KeyError:
    #         pass
    # datasets = [np.asarray(dat, dtype=np.float32) for dat in seq2]
    return data_3d


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


def prepare_data(time_steps, data_set_path, labels=False):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = divide_data(data_set_path)  # �ָ����ݼ�:��һά����(narray)��ʽ���
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def generate_x_y(time_steps, data_set_path):
    train_x, val_x, test_x = prepare_data(time_steps, data_set_path)
    train_y, val_y, test_y = prepare_data(time_steps, data_set_path, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

