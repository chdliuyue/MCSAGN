import torch
import torch.nn as nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义RNN模型
# 定义一个简单的RNN
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN层的初始化
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层，用于将RNN的输出转化为最终输出

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)  # 初始化隐藏状态
        out, _ = self.rnn(x, h0)  # RNN的前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出，送入全连接层进行分类或回归等任务

        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始细胞状态
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出，通过全连接层得到输出

        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # n*2

        return out


class MLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size*2, num_layers, batch_first=True)
        # self.lstm3 = nn.LSTM(input_size, hidden_size*3, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*3, output_size)

    def forward(self, x):
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始细胞状态
        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*2).float().to(device)  # 初始隐藏状态
        c2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*2).float().to(device)  # 初始细胞状态
        # h3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*3).float().to(device)  # 初始隐藏状态
        # c3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*3).float().to(device)  # 初始细胞状态
        out1, _ = self.lstm1(x, (h1, c1))  # LSTM前向传播
        out2, _ = self.lstm2(x, (h2, c2))  # LSTM前向传播
        # out3, _ = self.lstm3(x, (h3, c3))  # LSTM前向传播
        out = torch.cat((out1, out2), dim=2)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出，通过全连接层得到输出

        return out


class MGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru2 = nn.GRU(input_size, hidden_size*2, num_layers, batch_first=True)
        # self.gru3 = nn.GRU(input_size, hidden_size*3, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*3, output_size)

    def forward(self, x):
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*2).float().to(device)  # 初始隐藏状态
        # h3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*3).float().to(device)  # 初始隐藏状态
        out1, hn1 = self.gru1(x, h1)
        out2, hn2 = self.gru2(x, h2)
        # out3, hn3 = self.gru3(x, h3)
        out = torch.cat((out1, out2), dim=2)
        out = self.fc(out[:, -1, :])
        return out


class AttenLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AttenLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始细胞状态
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        query = self.query(out)
        key = self.key(out)
        value = self.value(out)
        scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(key.shape[-1]).float())
        attention = self.softmax(scores)
        context = torch.matmul(attention, value)
        out = self.fc(context[:, -1, :])  # 取最后一个时间步的输出，通过全连接层得到输出
        return out


class AttenGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(AttenGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Self-attention layer
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU layer
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        out, _ = self.gru(x, h0)
        # Self-attention layer
        query = self.query(out)
        key = self.key(out)
        value = self.value(out)
        scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(key.shape[-1]).float())
        attention = self.softmax(scores)
        context = torch.matmul(attention, value)
        # Output layer

        out = self.fc(context[:, -1, :])  # n*2
        return out


class MAttenLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MAttenLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size*2, num_layers, batch_first=True)
        # self.lstm3 = nn.LSTM(input_size, hidden_size*3, num_layers, batch_first=True)
        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)
        self.query2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.key2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.value2 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.query3 = nn.Linear(hidden_size*3, hidden_size*3)
        # self.key3 = nn.Linear(hidden_size*3, hidden_size*3)
        # self.value3 = nn.Linear(hidden_size*3, hidden_size*3)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        # self.softmax3 = nn.Softmax(dim=-1)
        self.fc = nn.Linear(hidden_size*3, output_size)

    def forward(self, x):
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始细胞状态
        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*2).float().to(device)  # 初始隐藏状态
        c2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*2).float().to(device)  # 初始细胞状态
        # h3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*3).float().to(device)  # 初始隐藏状态
        # c3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*3).float().to(device)  # 初始细胞状态
        out1, _ = self.lstm1(x, (h1, c1))  # LSTM前向传播
        out2, _ = self.lstm2(x, (h2, c2))  # LSTM前向传播
        # out3, _ = self.lstm3(x, (h3, c3))  # LSTM前向传播
        query1 = self.query1(out1)
        key1 = self.key1(out1)
        value1 = self.value1(out1)
        query2 = self.query2(out2)
        key2 = self.key2(out2)
        value2 = self.value2(out2)
        # query3 = self.query3(out3)
        # key3 = self.key3(out3)
        # value3 = self.value3(out3)
        scores1 = torch.matmul(query1, key1.transpose(-1, -2)) / torch.sqrt(torch.tensor(key1.shape[-1]).float())
        scores2 = torch.matmul(query2, key2.transpose(-1, -2)) / torch.sqrt(torch.tensor(key2.shape[-1]).float())
        # scores3 = torch.matmul(query3, key3.transpose(-1, -2)) / torch.sqrt(torch.tensor(key3.shape[-1]).float())
        attention1 = self.softmax1(scores1)
        attention2 = self.softmax2(scores2)
        # attention3 = self.softmax3(scores3)
        context1 = torch.matmul(attention1, value1)
        context2 = torch.matmul(attention2, value2)
        # context3 = torch.matmul(attention3, value3)
        out = torch.cat((context1, context2), dim=2)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出，通过全连接层得到输出
        return out


class MAttenGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MAttenGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU layer
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru2 = nn.GRU(input_size, hidden_size*2, num_layers, batch_first=True)
        # self.gru3 = nn.GRU(input_size, hidden_size*3, num_layers, batch_first=True)
        # Self-attention layer
        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)
        self.query2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.key2 = nn.Linear(hidden_size*2, hidden_size*2)
        self.value2 = nn.Linear(hidden_size*2, hidden_size*2)
        # self.query3 = nn.Linear(hidden_size*3, hidden_size*3)
        # self.key3 = nn.Linear(hidden_size*3, hidden_size*3)
        # self.value3 = nn.Linear(hidden_size*3, hidden_size*3)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        # self.softmax3 = nn.Softmax(dim=-1)
        self.fc = nn.Linear(hidden_size*3, output_size)

    def forward(self, x):
        # GRU layer
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)  # 初始隐藏状态
        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*2).float().to(device)  # 初始隐藏状态
        # h3 = torch.zeros(self.num_layers, x.size(0), self.hidden_size*3).float().to(device)  # 初始隐藏状态
        out1, _ = self.gru1(x, h1)
        out2, _ = self.gru2(x, h2)
        # out3, _ = self.gru3(x, h3)
        # Self-attention layer
        query1 = self.query1(out1)
        key1 = self.key1(out1)
        value1 = self.value1(out1)
        query2 = self.query2(out2)
        key2 = self.key2(out2)
        value2 = self.value2(out2)
        # query3 = self.query3(out3)
        # key3 = self.key3(out3)
        # value3 = self.value3(out3)
        scores1 = torch.matmul(query1, key1.transpose(-1, -2)) / torch.sqrt(torch.tensor(key1.shape[-1]).float())
        scores2 = torch.matmul(query2, key2.transpose(-1, -2)) / torch.sqrt(torch.tensor(key2.shape[-1]).float())
        # scores3 = torch.matmul(query3, key3.transpose(-1, -2)) / torch.sqrt(torch.tensor(key3.shape[-1]).float())
        attention1 = self.softmax1(scores1)
        attention2 = self.softmax2(scores2)
        # attention3 = self.softmax3(scores3)
        context1 = torch.matmul(attention1, value1)
        context2 = torch.matmul(attention2, value2)
        # context3 = torch.matmul(attention3, value3)
        # Output layer
        out = torch.cat((context1, context2), dim=2)
        out = self.fc(out[:, -1, :])  # n*2
        return out

