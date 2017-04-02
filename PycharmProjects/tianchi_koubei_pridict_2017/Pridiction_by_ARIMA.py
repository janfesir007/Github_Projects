# -*-coding:gbk-*-
# -*-coding:utf-8-*-
"""
ʱ������ģ�ͣ�"�Իع�/���/�ƶ�ƽ��"ʱ�����л��ģ��(ARIMA)
�Իع飺AR
��֣�I  �������ƺͼ�����
�ƶ�ƽ����MA
ARMA(dta,(p,q)).fit()   # ARMAģ����Ϻ�����dta��ʱ����������;����p(�Իع麯����AR��������),q���ƶ�ƽ����(MA)������
ARIMA(dta,(p,d,q)).fit() # ARIMAģ����Ϻ���������p,qͬ��,d:��ֵĴ���(ȡ0,1��2),���ǽ���������ARIMA�в��ֻ��1�ף�����ARIMAģ��ֻ��d��1�ײ�ֲ���.
ARIMAģ��:��һ�������ҳ�ʹ������ƽ�ȵĲ�ִ���d;�ڶ���������d,���ݡ�BIC׼�����ҳ�����p,q��������ģ��; ��������Ԥ�⣨�����ֻ�ԭ������
ARMA��ARIMAģ�͵Ĳ�ͬ��
    ������ݲ�ƽ��,���ִ���,ARMAģ��ѵ��ʱʹ�õ��ǲ�ֺ������,��Ԥ�����Ҫ��ֻ�ԭ����.
    ARIMAģ��ѵ��ʱʹ�õ���ԭʼ����,Ԥ��������ֻ�ԭ����.�ÿ����ARIMAģ��ֻ�ṩ��������εĲ��ARIMA(p,d,q),��d���ֻ��ȡ2.
    ��ĳ������Ҫ����3�����ϲ�ֲ�ƽ����ʹ��ARMAģ��.
"""
from __future__ import print_function  # ʹ�������ʽΪ��print()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import os
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller  # ADF��λ������
import statsmodels.api as sm


"""
     ���ڸ��������ʱ������,���ǿ���ͨ���۲������ͼ��ƫ���ͼ������ģ��ʶ��(ȷ��p,qֵ),
     ��������Ҫ������ʱ���������϶�,����ҪԤ��ÿֻ��Ʊ������,���ǾͲ��������ȥ������.
     ��ʱ���ǿ�������BIC׼��ʶ��ģ�͵�p, qֵ,ͨ����ΪBICֵԽС��ģ����Ը���.
     BIC׼���ۺϿ����˲в��С���Ա����ĸ���,�в�ԽСBICֵԽС,�Ա�������Խ��BICֵԽС.
     �����proper_model_pq(data_ts, maxLag, d)�����������ṩ��ƽ����������,���������ģ�͵�p,qֵ.
"""

# with open("data/process_data/user_pay_del_zero1.csv", "r") as f:
#     count = 0
#     for line in f:
#         count += 1
#         if count == 2:
#             data_line = line.replace("\n", "").split(",")
#             shop_id = data_line[0]
#             data = [int(i) for i in data_line[1:]]
#             fig = plt.figure(figsize=(12, 8))  # ����
#             ax1 = fig.add_subplot(111)  # ����ϵ
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
            model = ARIMA(data_ts, order=(p, d, q))  # (���ϵڶ�������֤1�β�ֺ�����ƽ��,��d=1)
            try:
                results_ARIMA = model.fit(disp=-1)  # ģ�����,��ʱ
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
    """ADF��λ������,��֤���ݵ�ƽ����"""

    dftest = adfuller(ts)  # ADF��λ������
    # ������������õ�ֵ������������
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


def Arima_model(datas):  # ����ģ�Ͳ���ȷ����ѵ��������ѵ���õ�ģ��
    d = 0  # ��ִ���
    ts = pd.Series(datas, dtype=float)
    ts.index = pd.date_range(start="2016", periods=len(ts), freq="D")  # periods���������� ;freq��ʱ�������ļ��

    # һ.���ʱ�����������Ƿ�ƽ��
    dfoutput = testStationarity(ts)  # ƽ���Լ�����dfoutput,���ص�����������ʱ������
    # print("ԭʼ����p-value:", dfoutput["p-value"], "\n", dfoutput)  # p-valueֵԽСԽƽ��

    # ��.���ڲ�ƽ�ȵ�����,�ҳ��侭�����β�ֺ��ƽ�ȣ�d=0,1,2��
    if (dfoutput['p-value'] >= 0.01):
        dta_diff1 = ts.diff(1)  # shift(m)���ƶ�����,�������ƶ�m��λ��; diff(n)��һ�ײ�֡�����:����n��λ�õ��������(���ƶ�n��λ��,���������)���У�dta.diff(n) = dta - dta.shift(n)
        dta_diff1.dropna(inplace=True)  # ��䲻����:Ӧ����diff������Ч����(ǰ��������Ч),ֱ��ȥ��
        dfoutput = testStationarity(dta_diff1)  # �ٴμ�������Ƿ�ƽ��.ƽ���Լ�����dfoutput,���ص�����������ʱ������
        if dfoutput["p-value"] < 0.05:
            d = 1
            # print("1��1�ײ�ִ����p-value:", dfoutput["p-value"], "\n")
        else:
            dta_diff2 = dta_diff1.diff(1)
            dta_diff2.dropna(inplace=True)
            dfoutput_end = testStationarity(dta_diff2)
            if dfoutput_end["p-value"] < 0.05:
                d = 2
                # print("2��1�ײ�ִ����p-value:", dfoutput["p-value"], "\n")
            else:
                raise Exception("���棺��2��1�ײ�ֺ������Բ�ƽ�ȣ�ARIMAģ�Ͳ����ã�")  # �쳣���׳�,���������޷�ִ��
    # else:
    #     print("ԭʼ����δ���в�ִ���p-value:", dfoutput["p-value"], "\n")


    # ��.ģ��ѵ��
    # p, q, arima_model = proper_model_pq(ts, len(ts)/10, d)  # �ҳ�����ģ��arima_model,ģ��ѵ��ʹ��ԭʼ����ts
    p, q, arima_model = proper_model_pq(ts, 5, d)
    # arima_model = ARIMA(ts, (q, d, q)).fit(disp=-1)
    # arima_model_fit = pd.Series(arima_model.fittedvalues, copy=True)  # ģ����ϵõ������ֵ(2002-2080)
    # arima_model_fit['2001'] = ts['2001']
    # arima_model_fit['2002'] = ts['2002']
    return arima_model

#  ��.ʹ��ģ����Ԥ��δ���Ĺ�ֵ
# arima_model.summary2()
# ARMA/ARIMAģ�͵�predict����������Բ�ֺ�ƽ�����е�Ԥ��,�ʣ�ʹ�øú������ն�Ҫ����һ������ԭ��֡��Ĳ���
# ARMA��forecast()��predict()����.��ARIMA��forecast()����ֱ��Ԥ��,����Ҫ��ֻ�ԭ����.
# �ܽ�1��ARMAģ��:���ݾ���ִ����,ʹ��predict()Ԥ��ƽ������,����в�ֻ�ԭ����.
# �ܽ�2��ARIMAģ�ͣ�ʹ��forecast()ֱ��Ԥ��,����Ҫ��ֻ�ԭ����.
# predict_values = arima_model.predict(str(len(datas) + 2001), str(len(datas) + 2003), dynamic=True)  # # Ԥ����1��ֵ,����������Serise
# forecast_values = arima_model.forecast(5)# Ԥ����ֵ


def predict_one_by_one(train_data, pri_num):  # Ԥ���õ������ݷ���:ԭ���� + Ԥ������,list����
    for i in range(pri_num):  # ����ʽѭ��Ԥ��������ֵ,����������arma_model.predict������һ��Ԥ����ֵ,���ڶ�׼ȷ�ȵĿ���
        arima_model = Arima_model(train_data)
        forecast_value = arima_model.forecast(1)[0][0]  # Ԥ����1��ֵ
        # �µ�Ԥ��ֵ����ԭʼ���ݼ�,��ԭʼ���ݼ�һ����Ϊ��һ��Ԥ������뼯
        # new_data = []
        # new_data.append(forecast_values)
        # new_datas = datas + new_data
        train_data.append(forecast_value)
    predict_data = train_data[-pri_num:]
    return predict_data


def predict_mutil_values(datas, num):
    arima_model = Arima_model(datas)
    forecast_values = arima_model.forecast(num)[0]  # Ԥ����ֵ
    return forecast_values


def check_shop_id(file_name):  # ��ȡ��Щ��Ԥ�����̼ҵ�ID
    shop_ids = []
    if os.path.exists(file_name):
        with open(file_name, "r") as f_pri:
            for id in f_pri:
                shop_id = id.replace("\n", "").split(",")[0]
                shop_ids.append(shop_id)
    return shop_ids


def check_L_value():  # �ҳ���ARIMAԤ��Ч����(L_valueֵ�Ƚϴ�)����Щ�̼�,д��Exception_ids.csv�ļ�
    with open("data/process_data/user_pay_del_all_zero_process.csv", "r") as f:
        with open("data/predict_data/Exception_ids.csv", "a") as fw:
            shop_ids_exist = check_shop_id("data/predict_data/Exception_ids.csv")
            count = 0
            for line in f:  # ÿһ����һ���̼ң���2000�У�����,����2000����������ϼ�Ԥ��
                count += 1
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = [int(i) for i in context[1:]]
                if shop_id in shop_ids_exist:  # �ж�ĳ���̼��Ƿ��Ѿ�����
                    continue
                train_data = data[:len(data)-pri_num]
                test_data = data[-pri_num:]
                predict_data1 = predict_mutil_values(train_data, pri_num)
                L1 = np.mean(abs(np.array(predict_data1) - np.array(test_data))/(np.array(predict_data1) + np.array(test_data)))
                # pri_tMAPE1 = np.mean(np.abs(np.array(predict_data1) - np.array(test_data)) / np.array(test_data))
                if count % 100 == 0:
                    time_end = time.time()
                    print ("�Ѳ���%d���̼�"%count)
                    print ("cost time: %dminutes" % (int(time_end - time_start) / 60))
                if L1 > 0.1:
                    fw.write(shop_id+",L = %.6f\n" % L1)
                    # print("shopId:%s - tMAPE1: %.3f - L1: %.6f" % (shop_id, pri_tMAPE1, L1))


def L_big_data_file():  # ��ָ��Lֵ��1.5-3���ϴ�Ч������̼�ȫ��д��һ���ļ�,�Ա��������Ԥ�⴦��
    shop_id_L_big = pd.read_csv("data/process_data/L_1.5-3.csv")["shop_id"].tolist()  # ����С�ļ����������ֶ�������/���㣩
    with open("data/process_data/user_pay_del_all_zero.csv", "r") as f:
        with open("data/process_data/user_pay_del_all_zero/data_L_1.5-3.csv", "w") as fw:
            for line in f:
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = context[1:]
                if int(shop_id) in shop_id_L_big:
                    fw.write(shop_id+",")
                    fw.write(",".join(data)+"\n")


def main():  # ��ʵԤ��δ��14���ֵ
    shop_id_L_big = pd.read_csv("data/process_data/L_1.5-3.csv")["shop_id"].tolist()  # ����С�ļ����������ֶ�������/���㣩
    with open("data/process_data/user_pay_del_all_zero_process.csv", "r") as f:
        shop_ids_exist = check_shop_id("data/predict_data/prediction.csv")
        with open("data/predict_data/prediction.csv", "a") as fw:
            count = 0
            for line in f:  # ÿһ����һ���̼ң���2000�У�����,����2000����������ϼ�Ԥ��
                count += 1
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                train_data = [int(i) for i in context[1:]]
                if int(shop_id) in shop_id_L_big:  # ��Ԥ��Lֵ��ָ��ֵ���ϴ���̼ң��ò����̼�����������Ԥ�⣩
                    continue
                if shop_id in shop_ids_exist:  # �ж�ĳ���̼��Ƿ��Ѿ���Ԥ����,������������Ԥ��
                    continue
                predict_data1 = predict_mutil_values(train_data, pri_num)
                fw.write(shop_id+",")
                fw.write(",".join([str(data) for data in [int(i) for i in predict_data1]])+"\n")
                if count % 10 == 0:
                    time_end = time.time()
                    print ("��Ԥ��%d���̼�"%count)
                    print ("cost time: %dminutes" % (int(time_end - time_start) / 60))


def pri_weekday_weekend(pri_ids):
    # Ԥ����Щ�������պ���ĩ���Ѳ�����̼�,����5��������֮�����Ѻܽӽ�,��ĩ2���������.���͵Ľڼ��յ���
    # pri_ids:list����
    with open("data/process_data/user_pay_del_all_zero.csv","r") as f:
        with open("data/predict_data/pri.csv","w") as fw:
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


def pri_week(pri_ids):
    # Ԥ��ÿ������ͬ���ɵ���Щ�̼�
    # pri_ids:list����
    with open("data/process_data/user_pay_del_all_zero.csv","r") as f:
        with open("data/predict_data/pri_week_L_big.csv","w") as fw:
            for line in f:
                context = line.replace("\n", "").split(",")
                shop_id = context[0]
                data = [int(i) for i in context[1:] if len(i) > 0]
                if int(shop_id) in pri_ids:
                    pri_tuesday = int(data[-14]*0.3+data[-7]*0.7)  # ������ �ܶ��ļ�Ȩƽ��ֵ��Ϊ���ܶ���Ԥ��ֵ������ͬ��
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


"""������"""
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
        # ��������
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

