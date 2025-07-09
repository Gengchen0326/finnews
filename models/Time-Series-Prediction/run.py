import os
import json
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from torch.autograd import Variable
from core.data_processor import DataLoader, CombinedDataLoader
from core.model import GRUNet, LSTMModel, TransformerModel
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

print("Starting Time Series Prediction Model")

# 设置随机种子以确保结果可重现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 生成带有当前时间的文件夹名
current_time = datetime.now().strftime("%Y%m%d%H")

# 特殊数据处理说明
print("\n" + "="*80)
print("数据预处理信息:")
print("- Volume列会自动进行对数变换处理，以减小数值范围并提高模型稳定性")
print("- 对数变换公式: log(Volume)，且Volume中的0值会被替换为1以避免log(0)")
print("="*80 + "\n")

def get_directional_accuracy(y_true, y_pred):
    """计算方向准确率 - 预测股价涨跌方向的准确率"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 计算方向
    y_true_direction = np.sign(y_true)
    y_pred_direction = np.sign(y_pred)
    
    # 计算方向匹配的比例
    correct_direction = np.sum(y_true_direction == y_pred_direction)
    total = len(y_true)
    
    return correct_direction / total if total > 0 else 0

def plot_results(y_true, y_pred, pred_type='Returns', results_dir='results'):
    """绘制结果和方向准确率"""
    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 计算指标
    dir_acc = get_directional_accuracy(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制真实值和预测值比较
    axes[0].plot(y_true, label=f'True {pred_type}')
    axes[0].plot(y_pred, label=f'Predicted {pred_type}')
    axes[0].set_title(f'{pred_type} Prediction Results (Direction Accuracy: {dir_acc:.4f}, MAE: {mae:.6f}, R2: {r2:.4f})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(f'{pred_type}')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制真实方向和预测方向
    axes[1].plot(np.sign(y_true), label=f'True {pred_type} Direction')
    axes[1].plot(np.sign(y_pred), label=f'Predicted {pred_type} Direction')
    axes[1].set_title(f'{pred_type} Direction Prediction')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Direction (+1/-1)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # 确保结果目录存在
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    plt.savefig(f'{results_dir}/{pred_type}_prediction_{timestamp}.png')
    plt.close()

def plot_results_by_stock(data_loader, y_true, y_pred, stock_ids, results_dir='results'):
    """按股票分组绘制结果"""
    # 获取时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if len(stock_ids) == 0:
        print("No stock ID information available, cannot plot by stock")
        return
    
    unique_stock_ids = np.unique(stock_ids)
    
    # 创建保存目录
    stocks_dir = f'{results_dir}/stocks'
    if not os.path.exists(stocks_dir):
        os.makedirs(stocks_dir)
    
    for stock_id in unique_stock_ids:
        # 获取此股票的索引
        indices = np.where(stock_ids == stock_id)[0]
        if len(indices) == 0:
            continue
            
        stock_y_true = np.array(y_true)[indices]
        stock_y_pred = np.array(y_pred)[indices]
        
        # 获取股票名称
        if 0 <= stock_id < len(data_loader.stock_names):
            stock_name = data_loader.stock_names[int(stock_id)]
        else:
            stock_name = f"Unknown_Stock_{stock_id}"
        
        # 计算此股票的方向准确率
        dir_acc = get_directional_accuracy(stock_y_true, stock_y_pred)
        mae = np.mean(np.abs(stock_y_true - stock_y_pred))
        # 计算R2决定系数
        r2 = r2_score(stock_y_true, stock_y_pred)
        
        # 绘制此股票的结果
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        axes[0].plot(stock_y_true, label='True Returns')
        axes[0].plot(stock_y_pred, label='Predicted Returns')
        axes[0].set_title(f'{stock_name} Returns Prediction (Direction Accuracy: {dir_acc:.4f}, MAE: {mae:.6f}, R2: {r2:.4f})')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Returns')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(np.sign(stock_y_true), label='True Direction')
        axes[1].plot(np.sign(stock_y_pred), label='Predicted Direction')
        axes[1].set_title(f'{stock_name} Direction Prediction')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Direction (+1/-1)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plt.savefig(f'{stocks_dir}/{stock_name}_prediction_{timestamp}.png')
        plt.close()

def unnormalize(normalized_values, base_values):
    return normalized_values * base_values + base_values

def predict_sequence_full(model, x_data, return_true=False):
    """
    进行全序列预测
    对于收益率预测，我们不需要将预测结果反归一化
    """
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # x_data: (batch_size, seq_len-1, num_features)
    
    # 创建一个空的预测结果列表
    pred_y = []
    
    # 遍历每个批次
    for i in range(len(x_data)):
        # 获取当前批次的序列
        curr_frame = x_data[i]
        # 使用模型预测下一个值
        input_tensor = torch.from_numpy(np.expand_dims(curr_frame, axis=0)).float().to(device)
        predicted = model(input_tensor)
        # 将预测结果添加到列表中 (先将其移回CPU)
        pred_y.append(predicted.detach().cpu().numpy()[0][0])
    
    # 返回完整的预测序列
    return np.array(pred_y)

def train(data_loader, config, model=None, config_name="model"):
    """
    训练一个新模型或微调现有模型
    """
    # 确保存在保存结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 打印模型配置信息
    print("\n" + "="*50)
    print(f"开始训练 {config['model_type'].upper()} 模型")
    print("="*50)
    print("\n模型配置信息:")
    print(f"- 模型类型: {config['model_type']}")
    print(f"- 输入维度: {config['input_dim']}")
    print(f"- 输出维度: {config['output_dim']}")
    print(f"- 隐藏层维度: {config['hidden_dim']}")
    print(f"- 层数: {config['num_layers']}")
    print(f"- Dropout率: {config['dropout']}")
    
    if config['model_type'] == 'transformer':
        print(f"- Transformer模型维度: {config.get('d_model', 512)}")
        print(f"- 注意力头数: {config.get('nhead', 8)}")
    
    print("\n训练配置信息:")
    print(f"- 训练周期: {config['epochs']}")
    print(f"- 批次大小: {config['batch_size']}")
    print(f"- 学习率: {config['learning_rate']}")
    print(f"- 优化器: {config.get('optimizer', 'adam')}")
    print(f"- 损失函数: {config.get('loss', 'mse')}")
    print(f"- 早停参数: 耐心值={config.get('early_stopping', {}).get('patience', 5)}, 最小改进={config.get('early_stopping', {}).get('min_delta', 0.001)}")
    print("="*50 + "\n")
    
    # 如果没有传入模型，创建一个新模型
    if model is None:
        # 配置模型
        if config["model_type"] == "gru":
            model = GRUNet(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                n_layers=config["num_layers"],
                dropout_rate=config["dropout"]
            )
        elif config["model_type"] == "lstm":
            model = LSTMModel(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
        elif config["model_type"] == "transformer":
            model = TransformerModel(
                input_dim=config["input_dim"],
                d_model=config.get("d_model", 512),
                nhead=config.get("nhead", 8),
                num_layers=config["num_layers"],
                output_dim=config["output_dim"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Unsupported model type: {config['model_type']}")
    
    # 将模型移动到设备上
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    learning_rate = config["learning_rate"]
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 获取训练和测试数据
    if config.get("use_combined_dataloader", False):
        x, y, _ = data_loader.get_train_data(
            seq_len=config["sequence_length"],
            normalise=config["normalise"]
        )
        x_test, y_test, _, _ = data_loader.get_test_data(
            seq_len=config["sequence_length"],
            normalise=config["normalise"],
            cols_to_norm=config["columns_to_normalise"]
        )
    else:
        x, y = data_loader.get_train_data(
            seq_len=config["sequence_length"],
            normalise=config["normalise"]
        )
        x_test, y_test, _ = data_loader.get_test_data(
            seq_len=config["sequence_length"],
            normalise=config["normalise"],
            cols_to_norm=config["columns_to_normalise"]
        )
    
    # 检查测试数据是否为空
    has_test_data = x_test.size > 0
    
    # 将训练数据转换为torch张量并移动到设备上
    x_train = torch.from_numpy(x).float().to(device)
    y_train = torch.from_numpy(y).float().to(device)
    
    if has_test_data:
        x_test_tensor = torch.from_numpy(x_test).float().to(device)
        y_test_tensor = torch.from_numpy(y_test).float().to(device)
    
    # 训练模型
    min_loss = float('inf')
    model.train()
    
    # 用于记录训练和测试损失
    train_losses = []
    test_losses = []
    
    for epoch in range(config["epochs"]):
        start_time = time.time()
        
        # 分批训练
        total_loss = 0
        for i in range(0, len(x_train), config["batch_size"]):
            batch_x = x_train[i:i + config["batch_size"]]
            batch_y = y_train[i:i + config["batch_size"]]
            
            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            # 反向传播和优化
            optimiser.zero_grad()
            loss.backward()
            # 添加梯度裁剪，防止梯度爆炸引起的NaN损失
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
        
        # 计算平均训练损失
        avg_train_loss = total_loss / (len(x_train) / config["batch_size"])
        train_losses.append(avg_train_loss)
        
        # 在测试集上评估
        test_loss = None
        if has_test_data:
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor).item()
                test_losses.append(test_loss)
            model.train()
        
        # 保存最佳模型
        if avg_train_loss < min_loss:
            min_loss = avg_train_loss
            torch.save(model.state_dict(), f'saved_models/{config["model_type"]}_{config_name}.pth')
        
        # 打印训练进度
        elapsed_time = time.time() - start_time
        if has_test_data:
            print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}, Time: {elapsed_time:.2f}s')
        else:
            print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {avg_train_loss:.6f}, Time: {elapsed_time:.2f}s')
    
    # 绘制训练和测试损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    if has_test_data:
        plt.plot(test_losses, label='Test Loss', color='red')
    plt.title(f'{config["model_type"].upper()} Model - {config_name} - Training and Test Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存损失曲线图
    loss_curve_path = f'results/{config["model_type"]}_{config_name}_loss_curves_{current_time}.png'
    plt.savefig(loss_curve_path)
    print(f'Training and test loss curves saved to {loss_curve_path}')
    
    return model

def predict(model, data_loader, config, is_combined_dataloader=False, config_name="model"):
    """
    使用训练好的模型进行预测
    """
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # 创建保存结果的目录
    results_dir = f'results/{config_name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 为每个股票创建保存目录
    stocks_dir = f'{results_dir}/stocks'
    if not os.path.exists(stocks_dir):
        os.makedirs(stocks_dir)
    
    # 获取测试数据
    try:
        if is_combined_dataloader:
            x_test, y_test, y_base, stock_ids = data_loader.get_test_data(
                seq_len=config["sequence_length"],
                normalise=config["normalise"],
                cols_to_norm=config["columns_to_normalise"]
            )
        else:
            x_test, y_test, y_base = data_loader.get_test_data(
                seq_len=config["sequence_length"],
                normalise=config["normalise"],
                cols_to_norm=config["columns_to_normalise"]
            )
            stock_ids = None
            
        # 当测试数据为空时，打印警告并返回
        if x_test.size == 0 or y_test.size == 0:
            print("Warning: Test data is empty, cannot make predictions")
            return None, None, None
            
        # 将numpy数组转换为torch张量并移动到设备上
        x_test_tensor = torch.from_numpy(x_test).float().to(device)
        
        # 设置模型为评估模式
        model.eval()
        
        # 进行预测
        with torch.no_grad():
            predictions = model(x_test_tensor)
        
        # 将预测结果转换为numpy数组（先将其移回CPU）
        predictions = predictions.cpu().numpy()
        
        # 以下是评估指标计算
        y_test_flat = y_test.flatten()
        predictions_flat = predictions.flatten()
        
        # 计算均方误差
        mse = np.mean((y_test_flat - predictions_flat)**2)
        # 计算均方根误差
        rmse = math.sqrt(mse)
        # 计算平均绝对误差
        mae = np.mean(np.abs(y_test_flat - predictions_flat))
        # 计算方向准确率
        directional_accuracy = get_directional_accuracy(y_test_flat, predictions_flat)
        # 计算R2决定系数
        r2 = r2_score(y_test_flat, predictions_flat)
        
        # 打印评估指标
        print(f'Mean Squared Error (MSE): {mse:.8f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.8f}')
        print(f'Mean Absolute Error (MAE): {mae:.8f}')
        print(f'Directional Accuracy: {directional_accuracy:.4f}')
        print(f'R-squared (R2): {r2:.4f}')
        
        # 保存评估指标
        metrics_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'Directional_Accuracy', 'R2'],
            'Value': [mse, rmse, mae, directional_accuracy, r2]
        })
        metrics_path = f'{results_dir}/metrics_{current_time}.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f'Metrics saved to {metrics_path}')
        
        # 绘制结果
        plot_results(y_test_flat, predictions_flat, pred_type='Returns', results_dir=results_dir)
        
        # 如果是组合数据加载器，还按股票绘制结果
        if is_combined_dataloader and stock_ids is not None and stock_ids.size > 0:
            plot_results_by_stock(data_loader, y_test_flat, predictions_flat, stock_ids.flatten(), results_dir=results_dir)
        
        return predictions, y_test, y_base
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None, None

def main():
    """
    主函数，加载配置并执行训练和预测
    """
    try:
        # 检查GPU可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*50}")
        print(f"设备信息:")
        print(f"- 使用设备: {device}")
        if torch.cuda.is_available():
            print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"- 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        print(f"{'='*50}\n")
        
        # 获取当前脚本所在目录，并拼接config.json路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')

        # 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
            
        configs = json.load(open(config_path, 'r'))

        
        # 确保保存模型的目录存在
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        # 依次运行无情绪和有情绪模型
        for config_name in ["nonsentiment_model", "sentiment_model"]:
            print(f"\n{'='*50}")
            print(f"Running {config_name}")
            print(f"{'='*50}\n")
            
            # 加载当前配置
            if config_name not in configs:
                print(f"Warning: Configuration for {config_name} not found in config.json, skipping...")
                continue
                
            config = configs[config_name]
            
            # 检查必要的配置项
            required_configs = ["data_path", "model_type", "input_dim", "hidden_dim", "output_dim", "num_layers"]
            missing_configs = [cfg for cfg in required_configs if cfg not in config]
            if missing_configs:
                print(f"Warning: Missing required configurations for {config_name}: {missing_configs}, skipping...")
                continue
            
            # 检查数据路径是否存在
            if not os.path.exists(config["data_path"]):
                print(f"Warning: Data path {config['data_path']} does not exist, skipping {config_name}")
                continue
            
            # 加载数据
            print(f"Loading data from: {config['data_path']}")

            dpath = config['data_path']
            # 检查是否包含分隔符并以 data_5y 结尾
            if ('/' in dpath or '\\' in dpath) and os.path.basename(dpath) == "data_5y":
                # 用合适的方式替换为 data_1y
                data_path_2 = os.path.join(os.path.dirname(dpath), "data_1y")
            else:
                data_path_2 = dpath  # 或保留原路径
            
            try:
                # 检查是否使用组合数据加载器
                if config.get("use_combined_dataloader", False):
                    data_loader = CombinedDataLoader(
                        data_folder=data_path_2,
                        split=config["train_test_split"],
                        cols=config["columns"],
                        cols_to_norm=config["columns_to_normalise"],
                        pred_len=config["sequence_length"]
                    )
                    is_combined_dataloader = True
                else:
                    data_loader = DataLoader(
                        filename=data_path_2,
                        split=config["train_test_split"],
                        cols=config["columns"],
                        cols_to_norm=config["columns_to_normalise"],
                        pred_len=config["sequence_length"]
                    )
                    is_combined_dataloader = False
            except Exception as e:
                print(f"Error loading data for {config_name}: {str(e)}")
                continue
            
            # 模型路径
            model_path = f'saved_models/{config["model_type"]}_{config_name}.pth'
            
            # 如果不是从头开始训练，尝试加载现有模型
            model = None
            if not config.get("train_from_scratch", True) and os.path.exists(model_path):
                print(f"Loading existing model: {model_path}")
                try:
                    if config["model_type"].lower() == "gru":
                        model = GRUNet(
                            input_dim=config["input_dim"],
                            hidden_dim=config["hidden_dim"],
                            output_dim=config["output_dim"],
                            n_layers=config["num_layers"],
                            dropout_rate=config.get("dropout", 0.0)
                        )
                    elif config["model_type"].lower() == "lstm":
                        model = LSTMModel(
                            input_dim=config["input_dim"],
                            hidden_dim=config["hidden_dim"],
                            output_dim=config["output_dim"],
                            num_layers=config["num_layers"],
                            dropout=config.get("dropout", 0.0)
                        )
                    elif config["model_type"].lower() == "transformer":
                        model = TransformerModel(
                            input_dim=config["input_dim"],
                            d_model=config.get("d_model", 512),
                            nhead=config.get("nhead", 8),
                            num_layers=config["num_layers"],
                            output_dim=config["output_dim"],
                            dropout=config.get("dropout", 0.0)
                        )
                    else:
                        raise ValueError(f"Unsupported model type: {config['model_type']}")
                        
                    model.load_state_dict(torch.load(model_path))
                except Exception as e:
                    print(f"Error loading model from {model_path}: {str(e)}")
                    model = None
            
            # 训练模型
            if config.get("train", False):
                print("Starting model training...")
                try:
                    model = train(data_loader, config, model, config_name)
                    
                    # 保存模型
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved to {model_path}")
                    
                    # 在训练后自动进行预测评估
                    print("Training complete, starting detailed evaluation on test set...")
                    predictions, ground_truth, base_values = predict(model, data_loader, config, is_combined_dataloader, config_name)
                except Exception as e:
                    print(f"Error during training for {config_name}: {str(e)}")
                    continue
                    
            elif config.get("predict", False):
                # 如果不训练但需要预测，确保有可用的模型
                if model is None:
                    if not os.path.exists(model_path):
                        print(f"Error: No model available for prediction. Model path {model_path} does not exist.")
                        continue
                        
                    try:
                        if config["model_type"].lower() == "gru":
                            model = GRUNet(
                                input_dim=config["input_dim"],
                                hidden_dim=config["hidden_dim"],
                                output_dim=config["output_dim"],
                                n_layers=config["num_layers"],
                                dropout_rate=config.get("dropout", 0.0)
                            )
                        elif config["model_type"].lower() == "lstm":
                            model = LSTMModel(
                                input_dim=config["input_dim"],
                                hidden_dim=config["hidden_dim"],
                                output_dim=config["output_dim"],
                                num_layers=config["num_layers"],
                                dropout=config.get("dropout", 0.0)
                            )
                        elif config["model_type"].lower() == "transformer":
                            model = TransformerModel(
                                input_dim=config["input_dim"],
                                d_model=config.get("d_model", 512),
                                nhead=config.get("nhead", 8),
                                num_layers=config["num_layers"],
                                output_dim=config["output_dim"],
                                dropout=config.get("dropout", 0.0)
                            )
                        else:
                            raise ValueError(f"Unsupported model type: {config['model_type']}")
                            
                        model.load_state_dict(torch.load(model_path))
                    except Exception as e:
                        print(f"Error loading model for prediction: {str(e)}")
                        continue
                
                print("Starting prediction...")
                try:
                    predictions, ground_truth, base_values = predict(model, data_loader, config, is_combined_dataloader, config_name)
                except Exception as e:
                    print(f"Error during prediction for {config_name}: {str(e)}")
                    continue
            
            print(f"\n{'='*50}")
            print(f"Completed {config_name}")
            print(f"{'='*50}\n")
            
    except Exception as e:
        print(f"Critical error in main function: {str(e)}")
        raise

if __name__ == '__main__':
    main()
