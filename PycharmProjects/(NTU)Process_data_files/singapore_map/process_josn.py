# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import json
# with open('singapore_ways', 'r') as f:
#     with open("singapore_way.txt","w") as fw:
#         i=0
#         for line in f:
#             if line == "\n":
#                 continue
#             for i in range(len(line)):
#                 if line[len(line)-i-1]=="\"":
#                     break
#             fw.write(line[:len(line)-i]+"\n")
with open('singapore_way0.txt', 'r') as f:
    with open("singapore_way.txt","w") as fw:
        for line in f:
            if line=="\n":
                continue
            else:
                fw.write(line[:-2]+"\n")