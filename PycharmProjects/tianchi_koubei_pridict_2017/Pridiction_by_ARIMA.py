# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
时间序列模型："自回归/差分/移动平均"时间序列混合模型(ARIMA)
自回归：AR
差分：I  消除趋势和季节性
移动平均：MA
ARMA(dta,(p,q)).fit()   # ARMA模型拟合函数：dta：时间序列数据;参数p(自回归函数（AR）的条件),q是移动平均数(MA)的条件
ARIMA(dta,(p,d,q)).fit() # ARIMA模型拟合函数：参数p,q同上,d:差分的次数(取0,1或2),不是阶数！而且ARIMA中差分只是1阶！即：ARIMA模型只有d次1阶差分操作.
ARIMA模型:第一步：先找出使得数据平稳的差分次数d;第二步：利用d,根据“BIC准则”再找出参数p,q及其最优模型; 第三步：预测（无需差分还原操作）
ARMA和ARIMA模型的不同：
    如果数据不平稳,需差分处理,ARMA模型训练时使用的是差分后的数据,故预测后还需要差分还原操作.
    ARIMA模型训练时使用的是原始数据,预测后无需差分还原操作.该库对于ARIMA模型只提供了最高两次的差分ARIMA(p,d,q),即d最大只能取2.
    若某数据需要经过3次以上差分才平稳则使用ARMA模型.
"""
from __future__ import print_function  # 使得输出格式为：print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller  # ADF单位根检验
import statsmodels.api as sm


"""
     对于个数不多的时序数据,我们可以通过观察自相关图和偏相关图来进行模型识别(确定p,q值),
     倘若我们要分析的时序数据量较多,例如要预测每只股票的走势,我们就不可能逐个去调参了.
     这时我们可以依据BIC准则识别模型的p, q值,通常认为BIC值越小的模型相对更优.
     BIC准则综合考虑了残差大小和自变量的个数,残差越小BIC值越小,自变量个数越少BIC值越小.
     下面的proper_model_pq(data_ts, maxLag, d)函数：根据提供的平稳序列数据,求出其最优模型的p,q值.
"""

# with open("data/process_data/user_pay_del_zero1.csv", "r") as f:
#     count = 0
#     for line in f:
#         count += 1
#         if count == 2:
#             data_line = line.replace("\n", "").split(",")
#             shop_id = data_line[0]
#             data = [int(i) for i in data_line[1:]]
#             fig = plt.figure(figsize=(12, 8))  # 画布
#             ax1 = fig.add_subplot(111)  # 坐标系
#             plt.plot(np.arange(len(data)), data)  # plt.plot(x,y)
#             # plt.show()
#             plt.savefig("data/process_data/pics/%s.jpg" % shop_id)


def proper_model_pq(data_ts, maxLag, d):
    init_bic = sys.maxint
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARIMA(data_ts, order=(p, d, q))  # (经上第二步骤验证1次差分后数据平稳,故d=1)
            try:
                results_ARIMA = model.fit(disp=-1)  # 模型拟合,耗时
            except:
                continue
            bic = results_ARIMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARIMA
                init_bic = bic
    return init_p, init_q, init_properModel


def testStationarity(ts):
    """ADF单位根检验,验证数据的平稳性"""

    dftest = adfuller(ts)  # ADF单位根检验
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def Arima_model(datas):  # 进行模型参数确定及训练并返回训练好的模型
    d = 0  # 差分次数
    ts = pd.Series(datas, dtype=float)
    ts.index = pd.date_range(start="2016", periods=len(ts), freq="D")  # periods：索引长度 ;freq：时间索引的间隔

    # 一.检测时间序列数据是否平稳
    dfoutput = testStationarity(ts)  # 平稳性检验结果dfoutput,返回的数据类型是时间序列
    # print("原始数据p-value:", dfoutput["p-value"], "\n", dfoutput)  # p-value值越小越平稳

    # 二.对于不平稳的数据,找出其经过几次差分后才平稳（d=0,1,2）
    if (dfoutput['p-value'] >= 0.01):
        dta_diff1 = ts.diff(1)  # shift(m)：移动操作,将数据移动m个位置; diff(n)”一阶差分“处理:隔开n个位置的数据相减(先移动n个位置,再数据相减)即有：dta.diff(n) = dta - dta.shift(n)
        dta_diff1.dropna(inplace=True)  # 这句不能少:应该是diff后有无效数据(前两个数无效),直接去除
        dfoutput = testStationarity(dta_diff1)  # 再次检测数据是否平稳.平稳性检验结果dfoutput,返回的数据类型是时间序列
        if dfoutput["p-value"] < 0.05:
            d = 1
            # print("1次1阶差分处理后p-value:", dfoutput["p-value"], "\n")
        else:
            dta_diff2 = dta_diff1.diff(1)
            dta_diff2.dropna(inplace=True)
            dfoutput_end = testStationarity(dta_diff2)
            if dfoutput_end["p-value"] < 0.05:
                d = 2
                # print("2次1阶差分处理后p-value:", dfoutput["p-value"], "\n")
            else:
                raise Exception("警告：经2次1阶差分后数据仍不平稳！ARIMA模型不适用！")  # 异常被抛出,后面的语句无法执行
    # else:
    #     print("原始数据未进行差分处理：p-value:", dfoutput["p-value"], "\n")


    # 三.模型训练
    # p, q, arima_model = proper_model_pq(ts, len(ts)/10, d)  # 找出最优模型arima_model,模型训练使用原始数据ts
    p, q, arima_model = proper_model_pq(ts, 5, d)
    # arima_model = ARIMA(ts, (q, d, q)).fit(disp=-1)
    # arima_model_fit = pd.Series(arima_model.fittedvalues, copy=True)  # 模型拟合得到的拟合值(2002-2080)
    # arima_model_fit['2001'] = ts['2001']
    # arima_model_fit['2002'] = ts['2002']
    return arima_model

#  四.使用模型来预测未来的估值
# arima_model.summary2()
# ARMA/ARIMA模型的predict函数都是针对差分后平稳序列的预测,故：使用该函数最终都要进行一步“还原差分”的操作
# ARMA中forecast()与predict()类似.但ARIMA中forecast()可以直接预测,不需要差分还原操作.
# 总结1：ARMA模型:数据经差分处理后,使用predict()预测平稳序列,后进行差分还原操作.
# 总结2：ARIMA模型：使用forecast()直接预测,不需要差分还原操作.
# predict_values = arima_model.predict(str(len(datas) + 2001), str(len(datas) + 2003), dynamic=True)  # # 预测下1个值,返回类型是Serise
# forecast_values = arima_model.forecast(5)# 预测多个值


def predict_one_by_one(train_data, pri_num):  # 预测后得到的数据返回:原数据 + 预测数据,list类型
    for i in range(pri_num):  # 迭代式循环预测多个后续值,而不采用在arma_model.predict函数中一次预测多个值,出于对准确度的考虑
        arima_model = Arima_model(train_data)
        forecast_value = arima_model.forecast(1)[0][0]  # 预测下1个值
        # 新的预测值加入原始数据集,和原始数据集一起作为下一次预测的输入集
        # new_data = []
        # new_data.append(forecast_values)
        # new_datas = datas + new_data
        train_data.append(forecast_value)
    predict_data = train_data[-pri_num:]
    return predict_data


def predict_mutil_values(datas, num):
    arima_model = Arima_model(datas)
    forecast_values = arima_model.forecast(num)[0]  # 预测多个值
    return forecast_values


def check_shop_id(file_name):  # 获取那些有预测结果商家的ID
    shop_ids = []
    if os.path.exists(file_name):
        with open(file_name, "r") as f_pri:
            for id in f_pri:
                shop_id = id.replace("\n", "").split(",")[0]
                shop_ids.append(shop_id)
    return shop_ids


def check_L_value():  # 找出用ARIMA预测效果差(L_value值比较大)的那些商家,写入Exception_ids.csv文件
    with open("data/process_data/user_pay_del_all_zero_process.csv", "r") as f:
        with open("data/predict_data/Exception_ids.csv", "a") as fw:
            shop_ids_exist = check_shop_id("data/predict_data/Exception_ids.csv")
            count = 0
            for line in f:  # 每一行是一个商家（共2000行）数据,进行2000次神经网络拟合及预测
                count += 1
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = [int(i) for i in context[1:]]
                if shop_id in shop_ids_exist:  # 判断某个商家是否已经存在
                    continue
                train_data = data[:len(data)-pri_num]
                test_data = data[-pri_num:]
                predict_data1 = predict_mutil_values(train_data, pri_num)
                L1 = np.mean(abs(np.array(predict_data1) - np.array(test_data))/(np.array(predict_data1) + np.array(test_data)))
                # pri_tMAPE1 = np.mean(np.abs(np.array(predict_data1) - np.array(test_data)) / np.array(test_data))
                if count % 100 == 0:
                    time_end = time.time()
                    print ("已测试%d个商家"%count)
                    print ("cost time: %dminutes" % (int(time_end - time_start) / 60))
                if L1 > 0.1:
                    fw.write(shop_id+",L = %.6f\n" % L1)
                    # print("shopId:%s - tMAPE1: %.3f - L1: %.6f" % (shop_id, pri_tMAPE1, L1))


def L_big_data_file():  # 将指标L值（1.5-3）较大（效果差）的商家全部写入一个文件,以便后续单独预测处理
    shop_id_L_big = pd.read_csv("data/process_data/L_1.5-3.csv")["shop_id"].tolist()  # 对于小文件可以用这种读法（简单/方便）
    with open("data/process_data/user_pay_del_all_zero.csv", "r") as f:
        with open("data/process_data/user_pay_del_all_zero/data_L_1.5-3.csv", "w") as fw:
            for line in f:
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = context[1:]
                if int(shop_id) in shop_id_L_big:
                    fw.write(shop_id+",")
                    fw.write(",".join(data)+"\n")


def main():  # 真实预测未来14天的值
    shop_id_L_big = pd.read_csv("data/process_data/L_1.5-3.csv")["shop_id"].tolist()  # 对于小文件可以用这种读法（简单/方便）
    with open("data/process_data/user_pay_del_all_zero_process.csv", "r") as f:
        shop_ids_exist = check_shop_id("data/predict_data/prediction.csv")
        with open("data/predict_data/prediction.csv", "a") as fw:
            count = 0
            for line in f:  # 每一行是一个商家（共2000行）数据,进行2000次神经网络拟合及预测
                count += 1
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                train_data = [int(i) for i in context[1:]]
                if int(shop_id) in shop_id_L_big:  # 不预测L值（指标值）较大的商家（该部分商家用其他方法预测）
                    continue
                if shop_id in shop_ids_exist:  # 判断某个商家是否已经有预测结果,若有则不再重新预测
                    continue
                predict_data1 = predict_mutil_values(train_data, pri_num)
                fw.write(shop_id+",")
                fw.write(",".join([str(data) for data in [int(i) for i in predict_data1]])+"\n")
                if count % 10 == 0:
                    time_end = time.time()
                    print ("已预测%d个商家"%count)
                    print ("cost time: %dminutes" % (int(time_end - time_start) / 60))


def pri_weekday_weekend(pri_ids):
    # 预测那些“工作日和周末消费差距大的商家,但在5个工作日之间消费很接近,周末2天消费相近.典型的节假日导向”
    # pri_ids:list类型
    with open("data/process_data/user_pay_del_all_zero.csv","r") as f:
        with open("data/predict_data/pri.csv","w") as fw:
            for line in f:
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = [int(i) for i in context[1:] if len(i) > 0]
                if int(shop_id) in pri_ids:
                    pri_weekdays = int(np.mean(np.array(data[-8:-3])+np.array(data[-15:-10]))/2)  # 最近两周十个工作日的消费均值
                    pri_weekend = int(np.mean(np.array(data[-3:-1])+np.array(data[-10:-8]))/2)  # 最近两周的周末的消费均值
                    fw.write(shop_id+",")
                    fw.write(",".join([str(pri_weekdays) for i in range(4)])+",")  # 周二至周四（预测的第一天是周二）
                    fw.write(",".join([str(pri_weekend) for i in range(2)])+",")  # 周末
                    fw.write(",".join([str(pri_weekdays) for i in range(5)])+",")  # 周一至周五
                    fw.write(",".join([str(pri_weekend) for i in range(2)])+",")  # 周末
                    fw.write(",".join([str(pri_weekdays) for i in range(1)])+"\n")  # 周一（预测的最后一天是周一）


def pri_week(pri_ids):
    # 预测每周有相同规律的那些商家
    # pri_ids:list类型
    with open("data/process_data/user_pay_del_all_zero.csv","r") as f:
        with open("data/predict_data/pri_week_L_big.csv","w") as fw:
            for line in f:
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = [int(i) for i in context[1:] if len(i) > 0]
                if int(shop_id) in pri_ids:
                    pri_tuesday = int(data[-14]*0.3+data[-7]*0.7)  # 上两周 周二的加权平均值作为下周二的预测值（以下同理）
                    pri_wedesday = int(data[-13] * 0.3 + data[-6] * 0.7)
                    pri_thesday = int(data[-12] * 0.3 + data[-5] * 0.7)
                    pri_friday = int(data[-11] * 0.3 + data[-4] * 0.7)
                    pri_sateday = int(data[-10] * 0.3 + data[-3] * 0.7)
                    pri_sunday = int(data[-19] * 0.3 + data[-2] * 0.7)
                    pri_monday = int(data[-8] * 0.3 + data[-1] * 0.7)
                    pri_data = [pri_tuesday,pri_wedesday,pri_thesday,pri_friday,pri_sateday,pri_sunday,pri_monday,
                                pri_tuesday,pri_wedesday,pri_thesday,pri_friday,pri_sateday,pri_sunday,pri_monday]
                    fw.write(shop_id+",")
                    fw.write(",".join([str(i) for i in pri_data])+"\n")


"""主函数"""
pri_num = 14
time_start = time.time()

has_pri_ids0 = pd.read_csv("data/predict_data/prediction_L.csv")["shop_id"].tolist()
with open("data/process_data/user_pay_del_all_zero.csv", "r") as f:
    has_pri_ids = pd.read_csv("data/predict_data/pridiction_part1 _improve_experience.csv")["5"].tolist()
    has_pri_ids.append(5)
    count = 0
    for line in f:
        context = line.replace("\n", "").split(",")
        shipid = context[0]
        if int(shipid) in has_pri_ids:
            continue
        if int(shipid) in has_pri_ids0:
            continue
        data = [int(i) for i in context[1:] if len(i) > 0]
        train_data = data[:len(data) - pri_num]
        test_data = data[-pri_num:]
        # 正则化数据
        # data_min = min(data)
        # data_max = max(data)
        # data_normolization = [10*(i - data_min) / float(data_max) for i in data]
        # data_normolization = [np.log10(i) if i!=0 else 0 for i in data ]
        # train_data1 = data_normolization[:len(data) - pri_num]
        # test_data1 = data_normolization[-pri_num:]

        predict_data = predict_mutil_values(train_data, pri_num)
        predict_data = np.array([float(int(i)) for i in predict_data])
        L = np.mean(abs(np.array(predict_data) - np.array(test_data)) / (np.array(predict_data) + np.array(test_data)))

        # predict_data1 = predict_mutil_values(train_data1, pri_num)
        # # predict_data1 = predict_data1*data_max/10 + data_min
        # predict_data1 = 10**predict_data1
        # predict_data1 = np.array([float(int(i)) if i!=1 else 0 for i in predict_data1])
        # L1 = np.mean(abs(np.array(predict_data1) - np.array(test_data))/(np.array(predict_data1) + np.array(test_data)))
        # # print([str(i) for i in predict_data1])
        # if L > L1:
        #     predict_data0 = predict_mutil_values(data_normolization, pri_num)
        #     predict_data0 = 10 ** predict_data0
        #     predict_data0 = [str(int(i)) if i != 1 else "0" for i in predict_data0]
        #     if not int(shipid) in has_pri_ids:
        #         with open("data/predict_data/prediction_norm.csv", "a") as fw0:
        #             fw0.write(shipid+",")
        #             fw0.write(",".join(predict_data0)+","+L1+"\n")
        with open("data/predict_data/prediction_L.csv", "a") as fw0:
            fw0.write(shipid+",")
            fw0.write(",".join([str(int(i)) for i in predict_data])+"," + str(L) + "\n")
        count += 1
        if count % 10 == 0:
            time_end = time.time()
            print("run %d records \n cost %d min" % (count, int(time_end-time_start)/60))

