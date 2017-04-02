#!/usr/bin/python
# -*- coding:utf-8 -*-
"""本示例：　线性回归(Linear regression)：三种手段的广告费　与　销售额的线性关系
　　本程序中采取了正则化处理（以防止过拟合）
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('8.Advertising.csv')    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print x
    print y

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    # print x_train, y_train

    model = Lasso()  # 经过L1正则化：“损失函数”加了一项"正则项(也叫惩罚因子,带有一个超参数alpha)", 防止过拟合
    # model = Ridge()  # 岭回归,经过L2正则化(另一种正则化方法)

    alpha_can = np.logspace(-3, 2, 10)  # 超参数可选值
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)  # 网格搜索法：５折验证,找出最佳超参数值,得到经过正则化处理后的lasso模型
    lasso_model.fit(x, y)
    print u'超参数：\n', lasso_model.best_params_

    y_hat = lasso_model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mape = np.average(abs(y_hat - np.array(y_test)) / np.array(y_test))
    print "MSE=%.6f - RMSE=%.6f - MAPE=%.6f" % (mse, rmse, mape)

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
