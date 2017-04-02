# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读入原始数据,保存在listing_data：DataFrame类型,索引是默认的“int”类型
with open("data/process_data/user_pay_del_zero.csv","r") as f:
    listing_data = f.readline().replace("\n", "").split(",")
"""原始数据部分可视化"""
def Data_Visualization():
    # 第一种画法：传统plt
    show_data = listing_data[1:]
    fig = plt.figure(figsize=(12, 8))  # 画布
    ax1 = fig.add_subplot(111)  # 坐标系
    plt.plot(np.arange(len(show_data)), show_data, color="b")  # plt.plot(x,y)
    plt.show()

Data_Visualization()