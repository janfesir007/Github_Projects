# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ����ԭʼ����,������listing_data��DataFrame����,������Ĭ�ϵġ�int������
with open("data/process_data/user_pay_del_zero.csv","r") as f:
    listing_data = f.readline().replace("\n", "").split(",")
"""ԭʼ���ݲ��ֿ��ӻ�"""
def Data_Visualization():
    # ��һ�ֻ�������ͳplt
    show_data = listing_data[1:]
    fig = plt.figure(figsize=(12, 8))  # ����
    ax1 = fig.add_subplot(111)  # ����ϵ
    plt.plot(np.arange(len(show_data)), show_data, color="b")  # plt.plot(x,y)
    plt.show()

Data_Visualization()