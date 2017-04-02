# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import Divide_X_Y
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor

if __name__ == "__main__":
    time_steps = 3
    data_set_path = "data/process_data/traffic_flow_20min.csv"
    x, y = Divide_X_Y.generate_x_y(time_steps, data_set_path)  # 函数返回字典类型
    x_train = x["train"]
    y_train = y["train"]
    x_test = x["test"]
    y_test = y["test"]
    reg = DecisionTreeRegressor(criterion="mse", max_depth=8)
    dtr = reg.fit(x, y)
    pri_test = dtr.predict(x_test)
    mape = np.mean((pri_test-y_test)/y_test)
    print mape

