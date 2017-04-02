# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""每20min划分一个时段,生成时段列表:2016-09-19 00:00; 2016-09-19 00:20; 2016-09-19 00:40"""
def Date_Range():
    start = "2016-09-19 00:00"
    end = "2016-10-18 00:00"
    _20min_counts = (datetime.datetime.strptime(end, "%Y-%m-%d %H:%M") - datetime.datetime.strptime(start, "%Y-%m-%d %H:%M")).days*72+1  # 1day==72*20min
    return [datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d %H:%M") +
                                       datetime.timedelta(minutes=20*i), "%Y-%m-%d %H:%M") for i in xrange(_20min_counts)]


""" 按某个收费站点(Tollgate)在每20分钟的时间间隔内通过的车辆数量统计出来：
    tollgate_id,direction,time_window,volume
    1,0,[2016-10-18 08:00:00,2016-10-18 08:20:00),57
"""
def process_orig_data():
    start_time = time.time()
    ids = ["1/0", "1/1", "2/0", "3/0", "3/1"]  # tollgate_id/direction
    sid_all_date = dict()  # keys: tollgate_id/direction/time_window
    for id in ids:
        dates = Date_Range()
        for date in dates:
            sid_all_date[id + "/" + date] = 0
    with open("data/orig_data/training/volume(table 6)_training.csv", "r") as fd:
        for line in fd:
            f_line = line.replace("\n", "").replace("\"", "").split(",")
            if f_line[0] == "time":  # 去掉首行－－－字段行
                continue
            if int(f_line[0][14]) <= 1:
                dic_id = f_line[1] + "/" + f_line[2]+"/"+f_line[0][:14]+"00"
            elif int(f_line[0][14]) >= 4:
                dic_id = f_line[1] + "/" + f_line[2]+"/"+f_line[0][:14]+"40"
            else:
                dic_id = f_line[1] + "/" + f_line[2]+"/"+f_line[0][:14]+"20"
            if dic_id in sid_all_date:
                sid_all_date[dic_id] += 1
            else:
                print u"舍去检测到的异常数据：", dic_id
    with open("data/process_data/traffic_flow_20min.csv", "w") as f:
        order_dic = list(sorted(sid_all_date.iteritems(), key=lambda asd: asd[0], reverse=False))
        f.write("tollgate_id,direction,time_window,volume\n")
        for line in order_dic:
            f.write(line[0].replace("/", ",")+","+str(line[1])+"\n")
    end_time = time.time()
    print end_time-start_time, "s"
process_orig_data()