# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""将原始的时间序列数据划分成：训练集/验证集/测试集,并将”X-Y“数据对也划分好－－－直接投入模型训练"""
import numpy as np


def divide_data(data_set_path):

    """
        Divide_data(): 将数据分割成“训练集/验证集/测试集”三部分数据,列表类型
        data_set_path上的”时间序列“是”列排列(且在最后一列),不是行排列“
        find date range for the split train, val, test (0.7, 0.1, 0.2 of total days)
    """
    with open(data_set_path, "r") as f:
        list_data = []
        first_line = True
        for line in f:
            context = line.replace("\n", "").split(",")
            if first_line:  # 去掉第一行---字段行
                first_line = False
                continue
            list_data.append(int(context[-1]))
    # list_data = list(np.array(list_data)/float(max(list_data)))  # 预处理：归一化
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
    for i in range(len_train):  # 数据分割
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
    # for i in range(len_train):  # 差分处理后的数据分割
    #     try:
    #         ac = list_data[i]
    #         daily_return = (ac - lastAc)  # 一次一阶差分
    #         # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化;  还原公式：ac = lastAc*daily_return+lastAc
    #         # if len(daily_returns) == daily_returns.maxlen:
    #         #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
    #         # daily_returns.append(daily_return)
    #         lastAc = ac
    #         seq2[0].append(daily_return)  # 一次一阶差分数据（原始不经过处理的数据效果很差）
    #         # seq[0].append(ac)  # 原始数据
    #     except KeyError:
    #         pass
    # for i in range(len_vild):
    #     try:
    #         ac = list_data[i + len_train]
    #         daily_return = (ac - lastAc)  # 一次一阶差分
    #         # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化
    #         # if len(daily_returns) == daily_returns.maxlen:
    #         #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
    #         # daily_returns.append(daily_return)
    #         lastAc = ac
    #         seq2[1].append(daily_return)  # 一次一阶差分数据
    #         # seq[1].append(ac)  # 原始数据
    #     except KeyError:
    #         pass
    # for i in range(len_test):
    #     try:
    #         ac = list_data[i + len_train + len_vild]
    #         daily_return = (ac - lastAc)  # 一次一阶差分
    #         # daily_return = (ac - lastAc) / lastAc  # 一次一阶差分标准化
    #         # if len(daily_returns) == daily_returns.maxlen:
    #         #     seq[idx].append(daily_return / np.std(daily_returns))  # 一次一阶差分标准化后,再进行数据长度为50的标准差归一化
    #         # daily_returns.append(daily_return)
    #         lastAc = ac
    #         seq2[2].append(daily_return)  # 一次一阶差分数据
    #         # seq[2].append(ac)  # 原始数据
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
        if labels:  # labels==True:表示在划分y,每一个序列(长度为time_steps)对应(映射/预测)一个y
            try:
                rnn_df.append(data[i + time_steps])
            except AttributeError:
                rnn_df.append(data[i + time_steps])
        else:  # labels==False:表示在划分X(序列),有重复的划分（隔一位继续划分）
            data_ = data[i: i + time_steps]
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df, dtype=np.float32)


def prepare_data(time_steps, data_set_path, labels=False):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = divide_data(data_set_path)  # 分割数据集:以一维数组(narray)形式存放
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))


def generate_x_y(time_steps, data_set_path):
    train_x, val_x, test_x = prepare_data(time_steps, data_set_path)
    train_y, val_y, test_y = prepare_data(time_steps, data_set_path, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

