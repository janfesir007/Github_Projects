# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import os
import pandas as pd

# """使用fromkeys()初始化字典奇怪的地方"""
# b = ["1","2"]
# a = {}.fromkeys(b, [0])  # a = dict().fromkeys(b, [1])：key是ｂ中的所有元素,value初始化为列表：[0]
# a["1"].append(2)  # 所有的key所对应的value都填充２（好像此时分辨不了key,所有key都一样）
# a["1"] = [1]  # 执行完赋值后,所有key才区分开了
# a["2"].append(3)  # 此时只有key为２的才填充３

def list_files(rootDir):  # 遍历一个文件夹
    file_names = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        if os.path.isdir(path):
            continue
        file_names.append(path)
    return file_names

file_names = list_files("orig_bus_stops_data")
error_files_counts = 0
# for name in file_names:
#     # name = "orig_bus_stops_data/CT8_2.txt"
#     with open(name, "r") as fr:  # 获得所有唯一的stop_ids
#         for line in fr:
#             da = line.replace("\n", "").split(",")
#             first_stop_id = da[0]
#             break
#         try:
#             stop_ids = pd.read_csv(name, dtype=str)[first_stop_id].tolist()  # 要设置dtype=str,否则默认读成整型,会将001读成1.
#         except:
#             error_files_counts += 1
#             print name+":Error file"+str(error_files_counts)
#             continue
#         stop_ids.append(first_stop_id)
#         stop_ids = [str(i) for i in set(stop_ids)]
#         stop_dic = dict().fromkeys(stop_ids, [])
#         count = 0
#     with open(name, "r") as fr:
#         lines_count = 0
#         for line in fr:
#             lines_count += 1
#             line_data = line.replace("\n", "").split(",")
#             stop_id = line_data[0]
#             if len(line_data) < 6:  # 没有经纬度信息
#                 stop_longitude = "0"  # 经度
#                 stop_latitude = "0"  # 纬度
#             else:
#                 stop_longitude = line_data[5]  # 经度
#                 stop_latitude = line_data[4]  # 纬度
#             try:
#                 if len(stop_latitude) > 0 and (float(stop_latitude) >= 0):
#                     if len(stop_longitude) > 0 and (float(stop_longitude) >= 0):
#                         count += 1
#                     else:
#                         stop_latitude = "0"
#                         stop_longitude = "0"
#                 else:
#                     stop_latitude = "0"
#                     stop_longitude = "0"
#             except :
#                 stop_latitude = "0"
#                 stop_longitude = "0"
#             if len(stop_dic[stop_id]) <= 0:
#                 stop_dic[stop_id] = [stop_latitude + "," + stop_longitude]
#             else:
#                 stop_dic[stop_id].append(stop_latitude + "," + stop_longitude)
#             if count > 120:
#                 break
#     with open("processed_bus_stops_data/%s" % name.split("/")[-1], "w") as fw:
#         for data in stop_dic:
#             w_data = list(set([i for i in stop_dic[data] if i != "0,0"]))
#             if len(w_data) > 0:
#                 fw.write(str(data)+","+",".join(w_data)+"\n")
#             else:
#                 fw.write(str(data)+"\n")
#     # a=1


for name in file_names:
    # name = "orig_bus_stops_data/RWS8_1.txt"
    with open(name, "r") as fr:
        stop_dict = {}
        for line in fr:
            line_data = line.replace("\n", "").split(",")
            stop_id = line_data[0]
            if len(line_data) >= 6:
                stop_longitude = line_data[5]  # 经度
                stop_latitude = line_data[4]  # 纬度
            else:
                stop_longitude = "0"  # 经度
                stop_latitude = "0"  # 纬度
            if stop_id in stop_dict:
                stop_dict[stop_id].append(stop_latitude+","+stop_longitude)
            else:
                stop_dict[stop_id] = [stop_latitude+","+stop_longitude]

    with open("processed_bus_stops_data/%s" % name.split("/")[-1], "w") as fw:
        for data in stop_dict:
                    w_data = list(set([i for i in stop_dict[data] if (i != "0,0" and i != "#,#" and i != ",")]))
                    avg_points = 150/len(stop_dict)
                    if len(w_data) > avg_points:
                        fw.write(str(data)+","+",".join(w_data[:avg_points])+"\n")
                    elif len(w_data)>0:
                        fw.write(str(data) + "," + ",".join(w_data) + "\n")
                    else:
                        fw.write(str(data)+"\n")

