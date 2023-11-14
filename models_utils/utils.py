import pandas as pd
import numpy as np
import glob
import torch
import time
from sklearn.model_selection import train_test_split
from models_utils.models import RNNModel, LSTMModel, GRUModel, MLSTMModel, MGRUModel
from models_utils.models import AttenLSTMModel, AttenGRUModel, MAttenLSTMModel, MAttenGRUModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(path=" ", sequence_length=120):
    data_all = pd.DataFrame()
    try:
        all_files = glob.glob(path)
        for file in all_files:
            df = pd.read_csv(file, encoding='gbk')
            data_all = pd.concat([data_all, df], ignore_index=True)
    except FileNotFoundError:
        print(f"None files : {path}")
    print("the total numbers of all data: ")
    print(data_all.shape)
    data_Car = data_all[data_all[' Type'] == ' Car']
    print("the total numbers of car data: ")
    print(data_Car.shape)
    data_ahead, data_right, data_left = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ahead_list = ["东西", "西东", "南北", "北南"]
    right_list = ["南东", "东北", "北西", "西南"]
    left_list = ["南西", "西北", "北东", "东南"]
    for i in data_Car.index:
        a = data_Car.loc[i, ' Entry Gate'][2:3]
        b = data_Car.loc[i, ' Exit Gate'][2:3]
        s = a + b
        temp_df = data_Car.loc[[i]].iloc[:, 8:(sequence_length + 1) * 7 + 8]
        if any(s in t for t in ahead_list):
            data_ahead = pd.concat([data_ahead, temp_df])
        elif any(s in t for t in right_list):
            data_right = pd.concat([data_right, temp_df])
        elif any(s in t for t in left_list):
            data_left = pd.concat([data_left, temp_df])
        else:
            pass
    data_intersection = pd.concat([data_ahead, data_right, data_left])
    print(data_ahead.shape)
    print(data_right.shape)
    print(data_left.shape)
    print(data_intersection.shape)

    return data_intersection, data_ahead, data_right, data_left


def prepare_data(p_data, input_size=7, output_size=2, test_size=0.2):
    data_num = p_data.shape[0]  # 数据总量
    sequence_length = int(p_data.shape[1] / input_size) - 1
    numpy_data = p_data.values
    data_2tensor = torch.tensor(numpy_data.astype('float'))
    data_3tensor = data_2tensor.view([data_num, -1, input_size])
    x = data_3tensor[:, :sequence_length, :]
    y = data_3tensor[:, sequence_length, :output_size]
    train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size=test_size, random_state=10)
    train_data = train_data.float().to(device)
    train_labels = train_labels.float().to(device)
    test_data = test_data.float().to(device)
    test_labels = test_labels.float().to(device)

    return train_data, train_labels, test_data, test_labels, data_num


def select_model(model_name="GRU", input_size=7, hidden_size=32, num_layers=1, output_size=2):
    if model_name == "RNN":
        model = RNNModel(input_size, hidden_size, output_size).to(device)
    elif model_name == "LSTM":
        model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "MCLSTM":
        model = MLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "SALSTM":
        model = AttenLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "MCSALSTM":
        model = MAttenLSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "GRU":
        model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "MCGRU":
        model = MGRUModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "SAGRU":
        model = AttenGRUModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_name == "MCSAGRU":
        model = MAttenGRUModel(input_size, hidden_size, num_layers, output_size).to(device)
    else:
        model = MAttenGRUModel(input_size, hidden_size, num_layers, output_size).to(device)

    return model


def trainer(model, train_data, train_labels, num_epochs, sequence_length, criterion, optimizer, save_model=True, save_name=""):
    t_loss, t_mse, t_mae, t_r2, t_time = [], [], [], [], []
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        for i in range(sequence_length):
            optimizer.zero_grad()  # 清空梯度缓存
            output = model(train_data[:, i:i + 1, :])  # 前向传播
            if i != (sequence_length-1):
                loss = criterion(output, train_data[:, i + 1, 0:2])  # 计算损失
            else:
                loss = criterion(output, train_labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新权重
            total_loss += loss.item()
        end_time = time.time()
        temp_time = end_time - start_time
        temp_loss = total_loss / sequence_length
        temp_mse, temp_mae, temp_r2 = model_predict(model, train_data, train_labels, sequence_length)
        t_loss.append(temp_loss)
        t_mse.append(temp_mse)
        t_mae.append(temp_mae)
        t_r2.append(temp_r2)
        t_time.append(temp_time)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train times {temp_time:.4f}, Train Loss: {temp_loss:.4f}, "
              f"Train MSE: {temp_mse:.4f}, Train MAE: {temp_mae:.4f}, Train R2: {temp_r2:.4f}")

    # print(f"Train average time is : {np.mean(t_time):.4f}")

    if save_model:
        torch.save(model, save_name)
        print('Model saved as {}'.format(save_name))

    return model, t_loss, t_mse, t_mae, t_r2, t_time


def model_predict(model, data, labels, sequence_length):
    total_mse, total_mae, total_r2 = 0, 0, 0
    for i in range(sequence_length):
        p = model(data[:, i:i + 1, :])
        if i != (sequence_length-1):
            y = data[:, i + 1, 0:2]
        else:
            y = labels
        pp = p.cpu().detach().numpy()
        yy = y.cpu().detach().numpy()
        mse = mean_squared_error(pp, yy)
        mae = mean_absolute_error(pp, yy)
        r2 = r2_score(pp, yy)
        total_mse += mse
        total_mae += mae
        total_r2 += r2

    return total_mse/sequence_length, total_mae/sequence_length, total_r2/sequence_length


