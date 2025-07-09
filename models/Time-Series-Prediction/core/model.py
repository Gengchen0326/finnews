import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # 添加绘图功能

class GRUNet(nn.Module):
    """PyTorch实现的GRU模型"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout_rate=0.2):
        """初始化GRU网络
        
        参数:
        - input_dim: 输入特征的维度
        - hidden_dim: 隐藏层的维度
        - output_dim: 输出的维度
        - n_layers: GRU层的数量
        - dropout_rate: Dropout的比率
        """
        super(GRUNet, self).__init__()
        
        # GRU层
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 保存配置
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        """前向传播
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_len, input_dim]
        
        返回:
        - 预测输出，形状为 [batch_size, output_dim]
        """
        # 初始化隐藏状态
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # GRU前向传播
        gru_out, _ = self.gru(x, h0)
        
        # 获取最后一个时间步的输出
        out = gru_out[:, -1, :]
        
        # 应用dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        
        return out

class LSTMModel(nn.Module):
    """PyTorch实现的LSTM模型"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        """初始化LSTM网络
        
        参数:
        - input_dim: 输入特征的维度
        - hidden_dim: 隐藏层的维度
        - output_dim: 输出的维度
        - num_layers: LSTM层的数量
        - dropout: Dropout的比率
        """
        super(LSTMModel, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 保存配置
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        """前向传播
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_len, input_dim]
        
        返回:
        - 预测输出，形状为 [batch_size, output_dim]
        """
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 获取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # 应用dropout
        out = self.dropout(out)
        
        # 全连接层
        out = self.fc(out)
        
        return out

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """前向传播
        
        参数:
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]
        
        返回:
        - 添加位置编码后的张量
        """
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    """PyTorch实现的Transformer模型"""

    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.2):
        """初始化Transformer网络
        
        参数:
        - input_dim: 输入特征的维度
        - d_model: Transformer的模型维度
        - nhead: 注意力头的数量
        - num_layers: Transformer编码器层的数量
        - output_dim: 输出的维度
        - dropout: Dropout的比率
        """
        super(TransformerModel, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 保存配置
        self.d_model = d_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        """前向传播
        
        参数:
        - x: 输入数据，形状为 [batch_size, seq_len, input_dim]
        
        返回:
        - 预测输出，形状为 [batch_size, output_dim]
        """
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        x = self.transformer_encoder(x)
        
        # 获取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 应用dropout
        x = self.dropout(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class Model():
    """用于构建和推理GRU模型的类"""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Model] 使用设备: {self.device}")
        self.optimizer = None
        self.criterion = None

    def load_model(self, filepath):
        """从文件加载模型
        
        参数:
        - filepath: 模型文件路径
        
        返回:
        - 是否成功加载模型
        """
        print('[Model] 从文件加载模型 %s' % filepath)
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            if isinstance(checkpoint, nn.Module):
                self.model = checkpoint
            else:
                print(f"加载的对象不是PyTorch模型: {type(checkpoint)}")
                return False
                
            # 初始化优化器和损失函数
            self.optimizer = optim.Adam(self.model.parameters())
            self.criterion = nn.MSELoss()
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("将构建新模型...")
            return False

    def build_model(self, configs):
        """根据配置构建模型
        
        参数:
        - configs: 模型配置
        """
        timer = Timer()
        timer.start()
        
        # 从配置中提取GRU层数量和尺寸
        layers_config = configs['model']['layers']
        n_layers = sum(1 for layer in layers_config if layer['type'] == 'gru')
        
        # 查找第一个GRU层获取输入尺寸
        input_dim = next(layer['input_dim'] for layer in layers_config if layer['type'] == 'gru')
        
        # 获取隐藏层尺寸（假设所有GRU层使用相同的神经元数量）
        hidden_dim = next(layer['neurons'] for layer in layers_config if layer['type'] == 'gru')
        
        # 获取Dropout率
        dropout_rate = next((layer['rate'] for layer in layers_config if layer['type'] == 'dropout'), 0.2)
        
        # 创建模型
        self.model = GRUNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # 时间序列预测通常是单变量输出
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        
        # 将模型移至适当的设备
        self.model.to(self.device)
        
        # 设置优化器
        optimizer_name = configs['model']['optimizer']
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters())
        elif optimizer_name.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters())
        else:
            self.optimizer = optim.Adam(self.model.parameters())
        
        # 设置损失函数
        loss_name = configs['model']['loss']
        if loss_name.lower() == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
        
        print('[Model] 模型编译完成')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        """使用内存中的数据训练模型
        
        参数:
        - x: 输入特征
        - y: 目标值
        - epochs: 训练周期数
        - batch_size: 批次大小
        - save_dir: 保存模型的目录
        """
        timer = Timer()
        timer.start()
        print('[Model] 开始训练')
        print('[Model] %s 周期, %s 批次大小' % (epochs, batch_size))

        # 准备数据
        x_tensor = torch.FloatTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 训练模型
        self.model.train()
        best_loss = float('inf')
        save_fname = os.path.join(save_dir, '%s-e%s.pt' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # 打印训练信息
            avg_loss = epoch_loss / len(dataloader)
            print(f'周期 {epoch+1}/{epochs}, 损失: {avg_loss:.6f}')
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model, save_fname)
                print(f'模型已改进，保存于 {save_fname}')

        print('[Model] 训练完成。模型保存为 %s' % save_fname)
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir, sentiment_type, model_name, num_csvs, patience=10, min_delta=0.0001):
        """使用生成器训练模型（用于大型数据集）
        
        参数:
        - data_gen: 数据生成器
        - epochs: 训练周期数
        - batch_size: 批次大小
        - steps_per_epoch: 每个周期的步数
        - save_dir: 保存模型的目录
        - sentiment_type: 情感类型
        - model_name: 模型名称
        - num_csvs: CSV文件数量
        - patience: 早停耐心值，默认10
        - min_delta: 损失改进的最小阈值，默认0.0001
        """
        timer = Timer()
        timer.start()
        print('[Model] 开始训练')
        print('[Model] %s 周期, %s 批次大小, %s 每周期批次' % (epochs, batch_size, steps_per_epoch))
        print(f'[Model] 早停参数 - 耐心值: {patience}, 最小改进阈值: {min_delta}')
        model_path = f"{model_name}_{sentiment_type}_{num_csvs}.pt"
        save_fname = os.path.join(save_dir, model_path)
        
        # 创建结果文件夹用于保存训练损失图
        result_folder = f"test_result_{num_csvs}"
        os.makedirs(result_folder, exist_ok=True)
        
        # 检查模型是否已加载
        if self.model is None:
            print("错误：没有可用的模型。请先构建或加载模型。")
            timer.stop()
            return
            
        # 检查优化器和损失函数是否已初始化
        if self.optimizer is None:
            print("初始化优化器...")
            self.optimizer = optim.Adam(self.model.parameters())
            
        if self.criterion is None:
            print("初始化损失函数...")
            self.criterion = nn.MSELoss()
        
        # 训练模型
        self.model.train()
        best_loss = float('inf')
        training_losses = []  # 用于记录每个epoch的损失
        
        # 早停计数器
        patience_counter = 0
        
        try:
            for epoch in range(epochs):
                epoch_loss = 0
                for _ in range(steps_per_epoch):
                    batch_x, batch_y = next(data_gen)
                    batch_x = torch.FloatTensor(batch_x).to(self.device)
                    batch_y = torch.FloatTensor(batch_y).to(self.device)
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    
                    # 反向传播
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # 计算并记录平均损失
                avg_loss = epoch_loss / steps_per_epoch
                training_losses.append(avg_loss)
                print(f'周期 {epoch+1}/{epochs}, 损失: {avg_loss:.6f}')
                
                # 保存最佳模型
                if best_loss - avg_loss > min_delta:
                    best_loss = avg_loss
                    torch.save(self.model, save_fname)
                    print(f'模型已改进 {best_loss - avg_loss:.6f} > {min_delta}，保存于 {save_fname}')
                    patience_counter = 0  # 重置早停计数器
                else:
                    patience_counter += 1
                    print(f'模型未足够改进 ({best_loss - avg_loss:.6f} <= {min_delta})，早停计数: {patience_counter}/{patience}')
                
                # 早停检查
                if patience_counter >= patience:
                    print(f'早停触发: 最后 {patience} 个周期内损失没有显著改进')
                    break
        
        except Exception as e:
            print(f"训练时出错: {e}")
            # 保存当前模型
            if self.model is not None:
                torch.save(self.model, save_fname)
                print(f"尽管出现错误，模型仍保存于 {save_fname}")
        
        # 绘制并保存训练损失图
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label='训练损失')
        plt.title(f'训练损失 - {model_name}_{sentiment_type}')
        plt.xlabel('周期')
        plt.ylabel('损失')
        plt.grid(True)
        plt.legend()
        
        # 保存损失图
        loss_plot_path = os.path.join(result_folder, f"{model_name}_{sentiment_type}_{num_csvs}_loss.png")
        plt.savefig(loss_plot_path)
        print(f'训练损失图已保存至 {loss_plot_path}')
        
        print('[Model] 训练完成。模型保存为 %s' % save_fname)
        timer.stop()

    def predict_point_by_point(self, data):
        """逐点预测
        
        参数:
        - data: 输入数据
        
        返回:
        - 预测结果
        """
        print('[Model] 逐点预测...')
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            predicted = self.model(data_tensor).cpu().numpy()
            predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        """预测多个序列
        
        参数:
        - data: 输入数据
        - window_size: 窗口大小
        - prediction_len: 预测长度
        
        返回:
        - 多个序列的预测结果
        """
        print('[Model] 预测多个序列...')
        self.model.eval()
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                with torch.no_grad():
                    tensor_frame = torch.FloatTensor(curr_frame[newaxis, :, :]).to(self.device)
                    result = self.model(tensor_frame).cpu().numpy()[0, 0]
                    predicted.append(result)
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequences_multiple_modified(self, data, window_size, prediction_len):
        """修改版多序列预测
        
        参数:
        - data: 输入数据
        - window_size: 窗口大小
        - prediction_len: 预测长度
        
        返回:
        - 多个序列的预测结果
        """
        self.model.eval()
        prediction_seqs = []
        for i in range(0, len(data), prediction_len):
            # 确保不会超出数据范围
            if i >= len(data):
                break
                
            curr_frame = data[i]
            predicted = []
            for j in range(prediction_len):
                with torch.no_grad():
                    tensor_frame = torch.FloatTensor(curr_frame[newaxis, :, :]).to(self.device)
                    result = self.model(tensor_frame).cpu().numpy()[0, 0]
                    predicted.append(result)
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        """预测完整序列
        
        参数:
        - data: 输入数据
        - window_size: 窗口大小
        
        返回:
        - 完整序列的预测结果
        """
        print('[Model] 预测完整序列...')
        self.model.eval()
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            with torch.no_grad():
                tensor_frame = torch.FloatTensor(curr_frame[newaxis, :, :]).to(self.device)
                result = self.model(tensor_frame).cpu().numpy()[0, 0]
                predicted.append(result)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted