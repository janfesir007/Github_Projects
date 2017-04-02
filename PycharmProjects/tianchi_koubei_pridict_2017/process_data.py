# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""�������������б�"""
def Date_Range():
    start = "2015-07-01"
    end = "2016-10-31"
    days = (datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1
    return [datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d") +
                                        datetime.timedelta(i), "%Y-%m-%d") for i in xrange(days)]

""" ��ÿһ���̼�һ�е����ݸ�ʽ,�������̼ҵ�ÿһ�죨1/7/2015--31/10/2016����֧����¼ͳ�Ƴ���"""
def process_orig_data():
    start_time = time.time()
    sid = []
    with open("data/orig_data/user_pay.txt", "r") as fd:
        line_counts = 0
        for line in fd:  # for line in fd ��佫 file ����ת���� iterable object,��Ȼ�ǿɵ�������һ��ֻ����һ�� item �����������Ὣ���� items �Ž��ڴ棬��˽�ʡ����Դ��
            f_line = line.replace("\n", "").split(",")
            f_sid = f_line[1]
            sid.append(f_sid)
            line_counts += 1
        print (u"���ļ���%d��", line_counts)
    sid_all_date = dict()  # ÿ���̼�ÿһ���֧������,û�м�¼��һ����Ϊ0
    shop_ids = list(set(sid))  # ͳ�Ƴ������̼ҵ�shop_id
    for id in shop_ids:
        dates = Date_Range()
        for date in dates:
            sid_all_date[id + "/" + date] = 0
    with open("data/orig_data/user_pay.txt", "r") as fd:
        for line in fd:
            f_line = line.replace("\n", "").split(",")
            f_sid = f_line[1]
            f_date = f_line[2][:10]
            dic_id = f_sid + "/" + f_date
            if dic_id in sid_all_date:
                sid_all_date[dic_id] += 1
            else:
                print (u"��ȥ��⵽���쳣���ݣ�", dic_id)
        # label = fd.tell()  # ��¼��ȡ�ļ��ĵ�ǰλ��

    with open("data/process_data/user_pay.csv", "w") as f:
        len_date = len(Date_Range())
        # len_date = 3
        i = 0
        order_dic = list(sorted(sid_all_date.iteritems(), key=lambda asd: asd[0], reverse=False))
        for line in order_dic:
            if i % len_date == 0:
                f.write(line[0].split("/")[0])
            i += 1
            f.write(","+str(sid_all_date[line[0]]))
            if i % len_date == 0:
                f.write("\n")
    end_time = time.time()
    print (end_time-start_time, "s")


"""����м�����Ϊ���ֵ"""
def process_mid_zreo():
    with open("data/process_data/user_pay_del_front_zero_process1.csv", "r") as f:
        with open("data/process_data/user_pay_del_front_zero_process.csv", "w") as fd:
            for context in f:
                line = context.replace("\n", '').split(",")
                shop_id = line[0]
                datas = [int(d) for d in line[1:]]
                for i in range(len(datas)):
                    if datas[i] == 0:
                        if i < len(datas)-22:
                            datas[i] = int(np.mean(datas[i+1:i+22]))
                        else:
                            datas[i] = int(np.mean(datas[i-21:i]))
                fd.write(shop_id+",")
                fd.write(",".join([str(i) for i in datas])+"\n")


def lists_Intersection(a_list, b_list):  # ���������б�Ľ���
    return list((set(a_list).union(set(b_list))) ^ (set(a_list) ^ set(b_list)))


# ids1 = pd.read_csv("data/predict_data/L_value_big_prediction.csv")["835"].tolist()[:500]
# ids1.append(835)
# ids2 = pd.read_csv("data/predict_data/repri_L1-1.5_weekend_weekday.csv")["2"].tolist()
# ids2.append(2)
# ids3 = pd.read_csv("data/predict_data/pridiction_part1 _improve_experience.csv")["5"].tolist()
# ids3.append(5)
# b = set(ids2+ids3)
# a = set(ids1)-b
# with open("data/predict_data/prediction.csv","r") as f:
#     with open("data/predict_data/repri_big_L.csv","a") as fw:
#         for line in f:
#             if int(line.replace("\n", "").split(",")[0]) not in a:
#                 fw.write(line)

ids2 = pd.read_csv("data/predict_data/prediction_part2_everyday_experience.csv")["2"].tolist()
ids2.append(2)
with open("data/predict_data/prediction_ARIMA_L.csv", "r") as f:
    with open("data/predict_data/prediction_part2_everyday_experience.csv", "a") as fw:
        for line in f:
            shop_id = line.replace("\n", "").split(",")[0]
            if shop_id == "shop_id":
                continue
            if int(shop_id) not in ids2:
                fw.write(line)


with open("data/process_data/user_pay_del_all_zero.csv", "r") as f:
    # ��������ͼ
    count = 0
    for context in f:
        line = context.replace("\n", "").split(",")
        shop_id = line[0]
        if int(shop_id) not in ids2:
            continue
        count += 1
        if count < 300:
            continue
        if count > 600:
            break
        data = [int(i) for i in line if len(i) > 0][-28:]
        fig = plt.figure(figsize=(12, 8))  # ����
        ax1 = fig.add_subplot(111)  # ����ϵ

        plt.plot(np.arange(len(data)), data)  # plt.plot(x,y)
        xticks_dates = [(i+1) for i in np.arange(7)]*4  # x��Ŀ̶�(ticks)��ʾ����
        del xticks_dates[0]
        xticks_dates.append(1)

        # ����x��̶��ϵı�ǩ:plt.xticks() ��1����������Ҫ��ʾ��ǩ�ĺ������λ��(int),����ʾ28��;
        #  ��2����������Ҫ��ʾ��ǩ��20��λ������Ӧ�ľ�������;��3������rotation: x��̶ȱ�ǩ��ת���ٶ�.
        plt.xticks(np.arange(28), xticks_dates, rotation=0)
        # plt.show()
        plt.savefig("data/process_data/pics_del_all_zero_ARIMA_L_large(>0.1)_3weeks/%s.jpg" % shop_id)


def process_zero(data_line):  # ��ȥ������Ϊ0�ļ�¼��,����д���ļ�
    result = []
    result1 = []
    if data_line is None or len(data_line) == 0:
        return result
    # pattern = data_line[0]
    # if int(pattern) != 0:
    #     result.append(pattern)
    # for s in data_line[1:]:
    #     if s == pattern and int(s)==0:
    #         # result[-1] += s
    #         continue
    #     else:
    #         pattern = s
    #         result.append(pattern)
    for data in data_line:
        result.append(int(data))
        if sum(result) == 0:
           result.pop(-1)
    for data1 in result:
        result1.append(str(data1))
    return result1


# with open("data/process_data/user_pay_del_zero.csv", "w") as fd:
#     # ��¼����ֵ�ź����д���ļ�
#     with open("data/process_data/user_pay.csv", "r") as f:
#         context = f.readlines()  # ����ȫ����ȡ
#         context_dic = {}
#         for i in context:
#             line = i.replace("\n", "").split(",")
#             w_line = process_zero(line[1:])  # ���ú���
#             context_dic[int(line[0])] = w_line
#         for line_data in sorted(context_dic):  # ��¼����ֵ�ź����д���ļ�
#             fd.write(str(line_data)+",")
#             # for w in w_line:
#             #     fd.write("," + w)
#             # fd.write("\n")
#             fd.write(",".join(context_dic[line_data])+"\n")  # �б���������д���ļ�; join(): split()"�ָ�"���������������


# �������ݾ�������,�����򻯺�Ч��������ȡ���򻯺������Ԥ��,����ԭ����Ԥ��
# with open("data/predict_data/prediction_ARIMA.csv","r") as f:
#     norm_ids = pd.read_csv("data/predict_data/prediction_norm.csv")["shop_id"].tolist()
#     with open("data/predict_data/prediction_norm.csv", "a") as fw:
#         for line in f:
#             context = line.replace("\n", "").split(",")
#             shop_id = context[0]
#             if int(shop_id) in norm_ids:
#                 continue
#             fw.write(line)


# with open("data/predict_data/prediction.csv", "r") as f:  # �ҳ�Ԥ��ֵΪ��ֵ����Щ�̼�
#     with open("data/predict_data/Neg_ids.csv", "w") as fw:
#         # i=0
#         for line in f:
#             # i+=1
#             context = line.replace("\n", "").split(",")
#             shop_id = context[0]
#             for i in context[1:-1]:
#                 if int(i)<0:
#                     fw.write(shop_id+",")
#                     break

#
# ids2 = pd.read_csv("data/process_data/L_1-1.5.csv")["shop_id"].tolist()
# ids_lstm = pd.read_csv("data/predict_data/prediction.csv")["835"].tolist()[:1331]
# ids_lstm.append(835)
# a = lists_Intersection(ids_lstm, ids2)
# with open("data/predict_data/prediction.csv", "r") as f:
#     with open("data/predict_data/L_value_big_prediction.csv", "w") as fw:
#         for line in f:
#             context = line.replace("\n", "").split(",")
#             if int(context[0]) not in ids2:
#                 if float(context[-1])>0.1:
#                     fw.write(line)
#
#
#
# ids = [2,3,4,6,7,11,17,21,23,28,30,31,47,53,54,59,65,68,73,75,77,83,85,92,104,106,107,110,112,123,125,127,132,141,150,166,167,169,173,186,201,203,204,218,222,223,232,234,244,245,250,258,260,276,288,290,295,300,303,307,308,311,318,321,325,327,336,351,359,369,372,373,393,410,411,417,418,419,421,422,426,429,435,439,443,446,453,454,458,459,462,463,464,465,470,471,477,480,481,485,487,490,491,492,493,498,505,510,516,524,537,545,556,557,558,559,561,574,576,580,584,585,588,594,596,601,602,607,610,611,613,614,635,641,643,644,647,654,657,664,671,677,682,684,687,689,691,697,700,707,708,712,723,726,727,739,744,749,753,754,756,759,760,764,765,766,771,780,781,782,786,797,798,812,813,819,823,829,830,832,833,835,837,838,839,841,842,857,863,869,873,877,878,881,882,887,890,895,899,901,910,913,918,920,928,929,944,946,953,955,964,981,982,983,984,988,990,993,1001,1003,1006,1007,1011,1013,1017,1022,1024,1025,1034,1040,1044,1049,1054,1056,1058,1063,1064,1065,1066,1074,1084,1092,1093,1099,1108,1109,1112,1113,1123,1128,1135,1136,1138,1143,1156,1157,1166,1173,1179,1182,1184,1188,1194,1197,1198,1205,1215,1217,1219,1231,1237,1239,1241,1242,1247,1248,1255,1257,1263,1274,1275,1290,1301,1306,1307,1309,1311,1316,1318,1320,1321,1322,1323,1325,1328,1332,1337,1342,1355,1367,1369,1379,1388,1392,1396,1397,1400,1404,1405,1407,1417,1420,1425,1426,1429,1430,1431,1435,1446,1448,1451,1452,1455,1476,1485,1486,1488,1490,1495,1499,1507,1513,1515,1516,1517,1521,1523,1527,1528,1529,1534,1541,1545,1547,1556,1559,1565,1568,1569,1575,1580,1582,1586,1589,1592,1599,1601,1603,1614,1619,1621,1629,1630,1631,1632,1635,1638,1639,1641,1642,1651,1658,1666,1669,1672,1683,1684,1692,1694,1697,1698,1706,1708,1711,1716,1718,1719,1728,1730,1735,1737,1738,1741,1750,1752,1753,1755,1757,1763,1780,1788,1797,1798,1799,1800,1801,1806,1811,1816,1821,1822,1824,1834,1835,1839,1840,1841,1843,1845,1846,1847,1848,1850,1851,1853,1857,1861,1862,1867,1869,1871,1872,1873,1876,1879,1882,1896,1902,1908,1909,1910,1913,1915,1916,1917,1918,1922,1924,1925,1933,1943,1952,1966,1969,1970,1972,1973,1980,1983,1987,1989,1990,1993,1994]
# with open("data/predict_data/prediction_0.csv", "r") as f:
#     with open("data/predict_data/prediction.csv", "a") as fw:
#         for line in f:
#             context = line.replace("\n", "").split(",")
#             shop_id = context[0]
#             if int(shop_id) not in ids:
#                 fw.write(line)
#
