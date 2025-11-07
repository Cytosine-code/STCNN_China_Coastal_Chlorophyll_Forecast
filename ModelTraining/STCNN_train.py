import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import datetime
import logging
from torch.utils.data import DataLoader, TensorDataset
import time
import math

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 只有在CUDA可用时才尝试获取设备名称
if torch.cuda.is_available():
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到CUDA支持的GPU，将使用CPU进行训练")

# 初始化日志记录
def setup_logging():
    """设置日志记录，同时输出到终端和文件"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建日志文件路径
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(script_dir, f'training_log_{timestamp}.txt')
    
    # 创建两个不同的handler，一个用于文件（带时间戳），一个用于控制台（不带时间戳）
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))  # 只输出消息内容，不带时间戳
    
    # 配置根logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除现有的handler
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # 替换print函数为logging.info
    global print
    original_print = print
    
    def log_print(*args, **kwargs):
        # 构建要打印的消息
        message = ' '.join(str(arg) for arg in args)
        # 使用logging.info记录消息
        logging.info(message)
    
    print = log_print
    
    return log_file

# 数据加载函数
def load_data():
    """加载训练和验证数据集"""
    print("正在加载数据...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')
    
    # =============================================
    # 加载训练数据批次
    # =============================================
    
    # 检查是否存在train.npz
    train_path = os.path.join(dataset_dir, 'train.npz')
    if os.path.exists(train_path):
        # 使用完整训练集
        print(f"  加载完整训练集 {train_path}")
        train_data = np.load(train_path)
        # 使用正确的键名 'features' 和 'labels'
        X_train = train_data['features']
        Y_train = train_data['labels']
    else:
        print(f"  未找到{train_path}，训练暂停")
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    
    # 加载验证数据
    val_path = os.path.join(dataset_dir, 'val.npz')
    val_data = np.load(val_path)
    # 使用正确的键名 'features' 和 'labels'
    X_val = val_data['features']
    Y_val = val_data['labels']
    
    # 加载陆地掩码
    mask_path = os.path.join(dataset_dir, 'land_mask.npz')
    mask_data = np.load(mask_path)
    # 使用正确的键名 'island'
    land_mask = mask_data['island']  # (lat_size, lon_size)
    
    print(f"\n数据加载完成:")
    print(f"  训练集: X形状={X_train.shape}, Y形状={Y_train.shape}")
    print(f"  验证集: X形状={X_val.shape}, Y形状={Y_val.shape}")
    print(f"  陆地掩码形状={land_mask.shape}")
    
    return X_train, Y_train, X_val, Y_val, land_mask

# STCNN模型定义
class STCNN(nn.Module):
    """
    时空卷积神经网络模型
    用于预测中国近海叶绿素浓度的7天时空预测
    """
    def __init__(self):
        super(STCNN, self).__init__()
        
        # 3D卷积层，用于捕捉时空特征
        self.conv1 = nn.Conv3d(
            in_channels=8,  # 8个特征通道
            out_channels=32,
            kernel_size=(3, 3, 3),  # 时间步3+空间3×3
            padding='same'
        )
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding='same'
        )
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3, 3),
            padding='same'
        )
        self.relu3 = nn.ReLU()
        
        # 输出层，预测未来7天
        self.output_conv = nn.Conv3d(
            in_channels=32,
            out_channels=7,  # 输出7天的预测
            kernel_size=(1, 1, 1)  # 不改变空间维度
        )
    
    def forward(self, x):
        """
        前向传播
        x: 输入张量，形状为 (batch_size, features, time_steps, lat, lon)
        """
        # 第一层3D卷积
        x = self.conv1(x)
        x = self.relu1(x)
        
        # 第二层3D卷积
        x = self.conv2(x)
        x = self.relu2(x)
        
        # 第三层3D卷积
        x = self.conv3(x)
        x = self.relu3(x)
        
        # 输出层，预测未来7天
        output = self.output_conv(x)
        
        return output

# 自定义损失函数（考虑陆地掩码）
class MaskedMSELoss(nn.Module):
    """带陆地掩码的均方误差损失函数，忽略掉陆地区域的预测"""
    def __init__(self, land_mask):
        super(MaskedMSELoss, self).__init__()
        # 将陆地掩码转换为海洋掩码（True表示海洋，False表示陆地）
        # 确保掩码类型正确，并移至设备
        self.ocean_mask = torch.tensor(~land_mask, dtype=torch.bool, device=device)
    
    def forward(self, pred, target):
        # 添加额外的安全检查，确保输入不包含nan
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 调整目标的维度顺序以匹配预测
        # 目标需要复制到所有7个时间步
        target_reshaped = target.permute(0, 3, 1, 2).unsqueeze(2)  # (batch_size, 7, 1, lat, lon)
        target_reshaped = target_reshaped.expand(-1, -1, pred.size(2), -1, -1)  # 扩展到与预测相同的时间维度
        
        # 应用海洋掩码
        batch_size = pred.size(0)
        time_steps = pred.size(2)  # 这是7
        
        # 扩展掩码以匹配批次和所有时间维度
        # 使用clone以避免潜在的内存共享问题
        mask_expanded = self.ocean_mask.clone().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, lat, lon)
        mask_expanded = mask_expanded.expand(batch_size, 7, time_steps, -1, -1)  # (batch_size, 7, 7, lat, lon)
        
        # 计算海洋区域的均方误差
        # 先计算平方差，再应用掩码
        squared_diff = (pred - target_reshaped) ** 2
        
        # 应用掩码并计算平均值
        masked_squared_diff = squared_diff * mask_expanded.float()
        
        # 计算掩码中True值的数量
        valid_count = mask_expanded.sum().float()
        
        # 添加安全检查，避免除以零
        if valid_count > 0:
            loss = masked_squared_diff.sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=device)
        
        # 确保返回的损失值不是nan或inf
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        
        return loss

# 数据预处理函数
def preprocess_data(X_train, Y_train, X_val, Y_val):
    """预处理数据，调整维度以适应PyTorch的输入格式"""
    print("正在预处理数据...")
    
    # PyTorch期望的输入格式: (batch_size, channels, time_steps, lat, lon)
    # 原始格式: (batch_size, lat, lon, time_steps, channels)
    
    # 调整训练数据维度
    X_train_pytorch = X_train.transpose(0, 4, 3, 1, 2)  # (batch_size, 8, 7, lat_size, lon_size)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_pytorch, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.transpose(0, 4, 3, 1, 2), dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    
    print(f"数据预处理完成:")
    print(f"  训练数据形状: {X_train_tensor.shape}")
    print(f"  验证数据形状: {X_val_tensor.shape}")
    
    return train_dataset, val_dataset

# 计算评估指标
def calculate_metrics(pred, target, land_mask):
    """
    计算评估指标：RMSE, R (皮尔逊相关系数), MAE
    
    参数:
    - pred: 预测值张量
    - target: 真实值张量
    - land_mask: 陆地掩码
    
    返回:
    - metrics: 包含RMSE, R, MAE的字典
    """
    # 确保输入是numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(land_mask, torch.Tensor):
        land_mask = land_mask.cpu().numpy()
    
    # 应用海洋掩码（只考虑海洋区域）
    ocean_mask = ~land_mask
    
    # 调整目标的维度顺序以匹配预测
    # 将陆地掩码扩展到与预测相同的维度
    mask_expanded = np.expand_dims(np.expand_dims(ocean_mask, axis=0), axis=0)  # (1, 1, lat, lon)
    mask_expanded = np.repeat(mask_expanded, pred.shape[0], axis=0)  # (batch, 1, lat, lon)
    mask_expanded = np.repeat(mask_expanded, pred.shape[1], axis=1)  # (batch, 7, lat, lon)
    
    # 应用掩码到预测和目标
    # 首先处理预测值 (batch, 7, 7, lat, lon) -> 简化为 (batch, 7, lat, lon)
    pred_flat = pred.mean(axis=2)  # 简化时间维度
    
    # 对海洋区域的值进行筛选
    pred_values = pred_flat[mask_expanded]
    # 目标值需要调整形状以匹配 (batch, 7, lat, lon)
    target_reshaped = target.transpose(0, 3, 1, 2)  # (batch, 7, lat, lon)
    target_values = target_reshaped[mask_expanded]
    
    # 移除可能存在的NaN值
    valid_mask = ~(np.isnan(pred_values) | np.isnan(target_values))
    pred_values = pred_values[valid_mask]
    target_values = target_values[valid_mask]
    
    # 计算指标
    metrics = {}
    
    # 计算MSE和RMSE
    if len(pred_values) > 0:
        mse = np.mean((pred_values - target_values) ** 2)
        rmse = math.sqrt(mse)
        
        # 计算MAE
        mae = np.mean(np.abs(pred_values - target_values))
        
        # 计算皮尔逊相关系数（R）
        if np.std(pred_values) > 0 and np.std(target_values) > 0:
            r = np.corrcoef(pred_values, target_values)[0, 1]
        else:
            r = 0.0
        
        metrics = {
            'RMSE': rmse,
            'R': r,
            'MAE': mae
        }
    else:
        metrics = {
            'RMSE': 0.0,
            'R': 0.0,
            'MAE': 0.0
        }
    
    return metrics

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, land_mask, num_epochs=50, save_path='best_model.pth'):
    """
    训练模型并保存最佳模型
    
    参数:
    - model: 要训练的模型
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - land_mask: 陆地掩码，用于计算评估指标
    - num_epochs: 训练轮次
    - save_path: 最佳模型保存路径
    
    返回:
    - 训练历史记录（训练损失和验证损失）
    """
    # 初始化历史记录
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_metrics = {}
    
    # 训练循环
    print(f"开始训练，共{num_epochs}轮...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练模式
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # 将数据移至设备
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(X_batch)
            
            # 计算损失
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item() * X_batch.size(0)
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_loss_history.append(train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        
        # 用于计算评估指标的累积变量
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # 将数据移至设备
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 前向传播
                outputs = model(X_batch)
                
                # 计算损失
                loss = criterion(outputs, y_batch)
                
                # 累计损失
                val_loss += loss.item() * X_batch.size(0)
                
                # 保存预测和真实值用于后续计算指标
                all_preds.append(outputs)
                all_targets.append(y_batch)
        
        # 计算平均验证损失
        val_loss /= len(val_loader.dataset)
        val_loss_history.append(val_loss)
        
        # 合并批次并计算评估指标
        if all_preds and all_targets:
            combined_preds = torch.cat(all_preds)
            combined_targets = torch.cat(all_targets)
            metrics = calculate_metrics(combined_preds, combined_targets, land_mask)
            
            # 打印评估指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        else:
            metrics = {}
            metrics_str = "无有效数据计算指标"
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics.copy()
            torch.save(model.state_dict(), save_path)
            print(f"  第{epoch+1}轮: 验证损失降低，保存最佳模型")
        
        # 打印当前轮次信息
        end_time = time.time()
        print(f"  第{epoch+1}/{num_epochs}轮 - 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 评估指标: {metrics_str}, 耗时: {end_time-start_time:.2f}秒")
    
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳模型已保存至: {save_path}")
    if best_metrics:
        best_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_metrics.items()])
        print(f"最佳评估指标: {best_metrics_str}")
    
    return train_loss_history, val_loss_history, best_metrics

# 绘制损失曲线
def plot_loss_history(train_loss_history, val_loss_history, metrics=None):
    """绘制训练和验证损失曲线并保存到脚本所在目录"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('STCNN Model Training Process')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss Value (MSE)')
    
    # 如果有评估指标，添加到图表标题或注释
    if metrics:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        plt.figtext(0.5, 0.01, f"Best Metrics: {metrics_str}", ha="center", fontsize=10)
    
    plt.legend()
    plt.grid(True)
    
    # 保存到脚本所在目录
    loss_plot_path = os.path.join(script_dir, 'loss_history.png')
    plt.savefig(loss_plot_path)
    plt.close()  # 关闭图表以释放内存
    print(f"损失曲线已保存为: {loss_plot_path}")

# 主函数
def main():
    # 设置日志记录
    log_file = setup_logging()
    print(f"日志将同时记录到: {log_file}")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 超参数设置
    # 临时配置 - CPU训练使用最小的batch_size以节省内存
    batch_size = 4  # 进一步减小以解决内存不足问题
    learning_rate = 0.0001  # 初始学习率
    
    # 临时使用最少的训练轮次以便快速测试
    num_epochs = 20  # 减小训练轮次以便快速测试
    
    # 确保模型保存到脚本所在目录
    save_path = os.path.join(script_dir, 'stcnn_model.pth')  # 最佳模型保存路径
    
    print(f"\n使用配置batch_size: {batch_size}, epochs: {num_epochs}")
    print(f"模型将保存到: {save_path}")
    
    try:
        # 加载数据
        X_train, Y_train, X_val, Y_val, land_mask = load_data()
        
        # 预处理数据
        train_dataset, val_dataset = preprocess_data(X_train, Y_train, X_val, Y_val)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = STCNN().to(device)
        print("模型创建完成:")
        print(model)
        
        # 创建损失函数和优化器
        criterion = MaskedMSELoss(land_mask)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        train_loss_history, val_loss_history, best_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, land_mask, num_epochs, save_path
        )
        
        # 绘制损失曲线
        plot_loss_history(train_loss_history, val_loss_history, best_metrics)
        
        # 打印最终损失
        print(f"\n最终训练损失: {train_loss_history[-1]:.6f}")
        print(f"最终验证损失: {val_loss_history[-1]:.6f}")
        
        # 打印最佳评估指标
        if best_metrics:
            print("\n模型性能评估:")
            print(f"  RMSE: {best_metrics.get('RMSE', 0):.4f}")
            print(f"  皮尔逊相关系数 (R): {best_metrics.get('R', 0):.4f}")
            print(f"  MAE: {best_metrics.get('MAE', 0):.4f}")
            
        print(f"\n训练完成！所有结果已记录到日志文件: {log_file}")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 关闭日志文件
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        print("日志记录已关闭")
    

if __name__ == "__main__":
    main()