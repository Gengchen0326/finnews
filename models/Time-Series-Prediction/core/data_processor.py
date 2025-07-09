import math
import numpy as np
import pandas as pd
import os

class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, cols_to_norm, pred_len):
        dataframe = pd.read_csv(filename)
        
        # 对Volume列进行对数化处理
        if 'Volume' in dataframe.columns:
            dataframe['Volume'] = self._log_transform_volume(dataframe['Volume'])
            print("已对Volume列进行对数变换")
            
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        # print(self.data_train[:10])
        self.data_test  = dataframe.get(cols).values[i_split:]
        # print(len(self.data_test))
        # print(self.data_test)
        self.cols_to_norm = cols_to_norm
        self.pred_len = pred_len
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def _log_transform_volume(self, volume_series):
        """对交易量数据进行对数变换"""
        # 确保所有值都是正数，对于0值特殊处理
        volume_series = volume_series.replace(0, 1)  # 将0替换为1，避免log(0)
        return np.log(volume_series)

    def get_test_data(self, seq_len, normalise, cols_to_norm):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        if len(data_windows) == 0:
            print("警告: 测试数据不足，无法创建窗口")
            # 创建空数组
            empty_x = np.array([]).reshape(0, seq_len-1, len(self.data_test[0]))
            empty_y = np.array([]).reshape(0, 1)
            empty_base = np.array([]).reshape(0, 1)
            return empty_x, empty_y, empty_base
            
        data_windows = np.array(data_windows).astype(float)
        
        # 保存基准值（用于反归一化）
        if data_windows.ndim >= 3:
            y_base = data_windows[:, 0, [0]]
        else:
            print(f"警告: data_windows 维度异常: {data_windows.shape}")
            # 处理一维数组情况
            if data_windows.ndim == 1:
                y_base = np.array([[data_windows[0]]])
            else:
                y_base = data_windows[:, [0]]
        
        # 归一化数据
        data_windows = self.normalise_selected_columns(data_windows, cols_to_norm, single_window=False) if normalise else data_windows
        
        # 分割为X和Y
        if data_windows.ndim >= 3:
            x = data_windows[:, :-1, :]
            y = data_windows[:, -1, [0]]
        else:
            print(f"警告: 归一化后的data_windows维度异常: {data_windows.shape}")
            # 创建空数组
            x = np.array([]).reshape(0, seq_len-1, len(self.data_test[0]) if len(self.data_test) > 0 else 0)
            y = np.array([]).reshape(0, 1)
            
        return x, y, y_base

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len,normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        # window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        window = self.normalise_selected_columns(window, self.cols_to_norm, single_window=True)[0] if normalise else window       
        # x = window[:-1]
        x = window[:-1]
        # y = window[0][2][0]
        y = window[-1, [0]]
        return x, y
# 
    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                w = window[0, col_i]
                if w == 0:
                  w = 1
                normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    # Modified normalization function to normalize only specific columns
    def normalise_selected_columns(self, window_data, columns_to_normalise, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if col_i in columns_to_normalise:
                    # Normalize only if the column index is in the list of columns to normalize
                    w = window[0, col_i]
                    if w == 0:
                        w = 1
                    normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                else:
                    # Keep the original data for columns not in the list
                    normalised_col = window[:, col_i].tolist()
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)


class CombinedDataLoader():
    """一次性加载所有股票数据并按时间顺序划分训练集和测试集"""
    
    def __init__(self, data_folder, split, cols, cols_to_norm, pred_len):
        """
        初始化组合数据加载器
        
        参数:
        - data_folder: 包含所有股票CSV文件的文件夹路径
        - split: 训练集/测试集划分比例
        - cols: 要使用的列名列表
        - cols_to_norm: 要归一化的列索引列表
        - pred_len: 预测长度
        """
        self.cols_to_norm = cols_to_norm
        self.pred_len = pred_len
        
        # 加载所有股票数据
        print(f"从 {data_folder} 加载所有股票数据...")
        all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        
        all_data = []
        self.stock_names = []
        
        for file in all_files:
            file_path = os.path.join(data_folder, file)
            try:
                df = pd.read_csv(file_path)
                
                # 对Volume列进行对数化处理
                if 'Volume' in df.columns:
                    df['Volume'] = self._log_transform_volume(df['Volume'])
                    print(f"对 {file} 的Volume列进行对数变换")
                
                # 计算收益率（如果收盘价列存在）
                if 'Close' in df.columns:
                    # 计算收益率: (p_t - p_{t-1})/p_{t-1}
                    df['Returns'] = df['Close'].pct_change()
                    # 将第一行的NaN值替换为0
                    df['Returns'].fillna(0, inplace=True)
                    
                    # 数据清洗：修剪极端值（可选）
                    # df['Returns'] = df['Returns'].clip(-0.2, 0.2)  # 限制在-20%到+20%之间
                    
                    # 创建包含收益率的新列列表
                    new_cols = ['Returns']  # 将收益率作为第一列
                    
                    # 添加原始列
                    for col in cols:
                        if col not in new_cols:  # 避免重复
                            new_cols.append(col)
                    
                    # 使用新的列获取数据
                    stock_data = df.get(new_cols).values
                else:
                    print(f"警告: 文件 {file_path} 中没有找到'Close'列，使用原始列")
                    stock_data = df.get(cols).values
                
                # 保存股票名称
                stock_name = file.split('.')[0]
                
                # 添加股票标识列
                stock_id_column = np.full((len(stock_data), 1), len(self.stock_names))
                stock_data_with_id = np.hstack((stock_data, stock_id_column))
                
                all_data.append(stock_data_with_id)
                self.stock_names.append(stock_name)
                print(f"加载了股票 {stock_name}")
            except Exception as e:
                print(f"加载 {file} 时出错: {e}")
        
        # 合并所有数据
        if all_data:
            self.combined_data = np.vstack(all_data)
            print(f"合并后的数据形状: {self.combined_data.shape}")
            
            # 划分训练集和测试集
            i_split = int(len(self.combined_data) * split)
            self.data_train = self.combined_data[:i_split]
            self.data_test = self.combined_data[i_split:]
            
            self.len_train = len(self.data_train)
            self.len_test = len(self.data_test)
            self.len_train_windows = None
            
            print(f"训练集大小: {self.len_train}, 测试集大小: {self.len_test}")
        else:
            raise ValueError("没有找到有效的股票数据")

    def get_test_data(self, seq_len, normalise, cols_to_norm):
        '''创建测试数据窗口'''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        if len(data_windows) == 0:
            print("警告: 测试数据不足，无法创建窗口")
            # 创建空数组
            empty_x = np.array([]).reshape(0, seq_len-1, self.data_test.shape[1]-1)  # 减1是因为去掉股票ID列
            empty_y = np.array([]).reshape(0, 1)
            empty_base = np.array([]).reshape(0, 1)
            return empty_x, empty_y, empty_base, np.array([]).reshape(0, 1)
            
        data_windows = np.array(data_windows).astype(float)
        
        # 保存收盘价基准值（用于参考）
        # 收盘价现在可能是第二列(索引1)，因为第一列是收益率
        close_col_idx = 1  # 假设收盘价是第二列
        y_base = data_windows[:, 0, [close_col_idx]] if data_windows.shape[2] > close_col_idx else data_windows[:, 0, [0]]
        
        # 获取股票ID信息，用于评估时区分不同股票
        stock_ids = data_windows[:, 0, [-1]].astype(int)
        
        # 删除股票ID列后再归一化
        data_windows_without_id = data_windows[:, :, :-1]
        
        # 归一化数据（注意：收益率列通常不需要归一化，因为它已经是相对值）
        if normalise:
            # 创建新的cols_to_norm列表，排除收益率列(索引0)
            modified_cols_to_norm = [i for i in cols_to_norm if i > 0]
            data_windows_norm = self.normalise_selected_columns(
                data_windows_without_id, modified_cols_to_norm, single_window=False
            )
        else:
            data_windows_norm = data_windows_without_id
        
        # 分割为X和Y，其中Y是收益率（第一列，索引0）
        x = data_windows_norm[:, :-1, :]
        y = data_windows_norm[:, -1, [0]]  # 收益率作为预测目标
        
        return x, y, y_base, stock_ids

    def get_train_data(self, seq_len, normalise):
        '''创建训练数据窗口'''
        data_x = []
        data_y = []
        stock_ids = []
        
        for i in range(self.len_train - seq_len):
            x, y, stock_id = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
            stock_ids.append(stock_id)
            
        return np.array(data_x), np.array(data_y), np.array(stock_ids)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''生成训练数据批次'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # 如果数据不能均匀分割，则为最后一个较小的批次
                    if x_batch:
                        yield np.array(x_batch), np.array(y_batch)
                    i = 0
                    break
                x, y, _ = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            if x_batch:
                yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''从给定索引位置生成下一个数据窗口'''
        window = self.data_train[i:i+seq_len]
        
        # 保存股票ID
        stock_id = window[-1, -1]
        
        # 去掉股票ID列
        window_without_id = window[:, :-1]
        
        # 归一化窗口数据（注意：收益率列通常不需要归一化）
        if normalise:
            # 创建新的cols_to_norm列表，排除收益率列(索引0)
            modified_cols_to_norm = [i for i in self.cols_to_norm if i > 0]
            window_norm = self.normalise_selected_columns(
                window_without_id, modified_cols_to_norm, single_window=True
            )[0]
        else:
            window_norm = window_without_id
        
        x = window_norm[:-1]
        y = window_norm[-1, [0]]  # 收益率作为预测目标（第一列）
        
        return x, y, stock_id

    def normalise_selected_columns(self, window_data, columns_to_normalise, single_window=False):
        """只归一化指定的列"""
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if col_i in columns_to_normalise:
                    # 仅当列索引在要归一化的列列表中时才归一化
                    w = window[0, col_i]
                    if abs(w) < 1e-10:  # 避免除以接近零的值
                        w = 1.0
                    normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                else:
                    # 对于不在列表中的列，保留原始数据
                    normalised_col = window[:, col_i].tolist()
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def _log_transform_volume(self, volume_series):
        """对交易量数据进行对数变换"""
        # 确保所有值都是正数，对于0值特殊处理
        volume_series = volume_series.replace(0, 1)  # 将0替换为1，避免log(0)
        return np.log(volume_series)
