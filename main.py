from models_utils.utils import read_data, prepare_data, select_model, trainer, model_predict
from models_utils.draw_utils import draw_scatter
from torch import nn, optim
import torch
import pandas as pd
import numpy as np

# 设置参数
sequence_length = 120
input_size = 7  # 输入维度
hidden_size = 32  # 隐藏状态大小
num_layers = 1  # LSTM层数
output_size = 2  # 输出维度
# train_batch_size = int(data_num*0.8)  # 批处理大小
test_size = 0.2
num_epochs = 20  # 训练周期数
learning_rate = 0.001  # 学习率
model_name = "LSTM"
data_name = "intersection"
# data_name = "ahead"


# data_intersection, data_ahead, data_right, data_left = read_data(path="./dataset1/*.csv",
#                                                                  sequence_length=sequence_length)
# data_intersection.to_csv('./dataset1_pd/intersection.csv', index=False)
# data_ahead.to_csv('./dataset1_pd/ahead.csv', index=False)
# data_right.to_csv('./dataset1_pd/right.csv', index=False)
# data_left.to_csv('./dataset1_pd/left.csv', index=False)

p_data = pd.read_csv('./dataset3_pd/'+data_name+'.csv', encoding='gbk')
print("#1 read data is Done.")

train_data, train_labels, test_data, test_labels, data_num = prepare_data(p_data,
                                                                          input_size=input_size,
                                                                          output_size=output_size,
                                                                          test_size=test_size)
print("#2 train data and test data is Done.")


# 训练模型
train_loss, train_mse, train_mae, train_r2, train_time, test_mse, test_mae, test_r2 = [], [], [], [], [], [], [], []
for i in range(1):
    model = select_model(model_name, input_size=input_size, hidden_size=hidden_size,
                         num_layers=num_layers, output_size=output_size)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    t_model, Loss, MSE, MAE, R2, t_Time = trainer(model, train_data, train_labels, num_epochs, sequence_length,
                                                  criterion, optimizer, save_model=False,
                                                  save_name="./model_save/" + model_name + ".pth")
    with torch.no_grad():
        mse, mae, r2 = model_predict(t_model, test_data, test_labels, sequence_length)

    draw_scatter(t_model, test_data, test_labels, sq=sequence_length, save_state=True,
                 save_path='./picture/scatter_' + model_name + '.pdf')

    # train_loss.append(Loss[-1])
    # train_mse.append(MSE[-1])
    # train_mae.append(MAE[-1])
    # train_r2.append(R2[-1]*100)
    # train_time.append(t_Time[-1])
    # test_mse.append(mse)
    # test_mae.append(mae)
    # test_r2.append(r2*100)

# print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}")


# print(f"Train average loss is : {np.mean(train_loss):.4f} ± {np.var(train_loss):.4f}")
# print(f"Train average MSE is : {np.mean(train_mse):.4f} ± {np.var(train_mse):.4f}")
# print(f"Train average MAE is : {np.mean(train_mae):.4f} ± {np.var(train_mae):.4f}")
# print(f"Train average R2 is : {np.mean(train_r2):.4f} ± {np.var(train_r2):.4f}")
# print(f"Train average time is : {np.mean(train_time):.4f} ± {np.var(train_time):.4f}")
# print(f"Test average MSE is : {np.mean(test_mse):.4f} ± {np.var(test_mse):.4f}")
# print(f"Test average MAE is : {np.mean(test_mae):.4f} ± {np.var(test_mae):.4f}")
# print(f"Test average R2 is : {np.mean(test_r2):.4f} ± {np.var(test_r2):.4f}")



