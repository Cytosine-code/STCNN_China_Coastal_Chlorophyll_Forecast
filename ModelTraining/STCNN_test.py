import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# STCNN模型定义（从训练脚本复制）
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
    
    # 重塑预测结果，移除额外的时间维度（因为预测形状是[6, 7, 7, 600, 480]）
    # 我们只需要第一个时间步的预测结果
    pred_day1 = pred[:, 0, 0]  # 获取每个样本的第一个预测结果 (6, 600, 480)
    
    # 重塑目标数据以匹配预测
    target_day1 = target[:, :, :, 0]  # 获取每个样本的第一个目标值 (6, 600, 480)
    
    # 扩展掩码以匹配批次维度
    batch_size = pred_day1.shape[0]
    mask_3d = np.expand_dims(ocean_mask, axis=0)  # (1, 600, 480)
    mask_3d = np.repeat(mask_3d, batch_size, axis=0)  # (batch, 600, 480)
    
    # 对海洋区域的值进行筛选
    pred_values = pred_day1[mask_3d]
    target_values = target_day1[mask_3d]
    
    # 移除可能存在的NaN值
    valid_mask = ~(np.isnan(pred_values) | np.isnan(target_values))
    pred_values = pred_values[valid_mask]
    target_values = target_values[valid_mask]
    
    # 计算指标
    metrics = {}
    
    if len(pred_values) > 0:
        mse = np.mean((pred_values - target_values) ** 2)
        rmse = math.sqrt(mse)
        mae = np.mean(np.abs(pred_values - target_values))
        
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

# 加载测试数据
def load_test_data():
    """加载测试数据集"""
    print("正在加载测试数据...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')
    
    # 加载测试数据
    test_path = os.path.join(dataset_dir, 'test.npz')
    if os.path.exists(test_path):
        print(f"  加载测试集 {test_path}")
        test_data = np.load(test_path)
        X_test = test_data['features']
        Y_test = test_data['labels']
    else:
        print(f"  未找到{test_path}")
        raise FileNotFoundError(f"测试文件不存在: {test_path}")
    
    # 加载陆地掩码
    mask_path = os.path.join(dataset_dir, 'land_mask.npz')
    mask_data = np.load(mask_path)
    land_mask = mask_data['island']  # (lat_size, lon_size)
    
    print(f"\n测试数据加载完成:")
    print(f"  测试集: X形状={X_test.shape}, Y形状={Y_test.shape}")
    print(f"  陆地掩码形状={land_mask.shape}")
    
    return X_test, Y_test, land_mask

# 预处理测试数据
def preprocess_test_data(X_test):
    """预处理测试数据，调整维度以适应PyTorch的输入格式"""
    print("正在预处理测试数据...")
    
    # 调整维度顺序: (batch_size, lat, lon, time_steps, channels) -> (batch_size, channels, time_steps, lat, lon)
    X_test_pytorch = X_test.transpose(0, 4, 3, 1, 2)  # (batch_size, 8, 7, lat_size, lon_size)
    
    # 转换为PyTorch张量
    X_test_tensor = torch.tensor(X_test_pytorch, dtype=torch.float32)
    
    print(f"  测试数据预处理完成，形状: {X_test_tensor.shape}")
    
    return X_test_tensor

# 加载模型
def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    
    # 创建模型实例
    model = STCNN().to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    
    print("模型加载完成")
    return model

# 预测函数
def predict(model, X_test_tensor, device):
    """使用模型进行预测"""
    print("开始预测...")
    
    with torch.no_grad():
        # 将数据移至设备
        X_test_tensor = X_test_tensor.to(device)
        
        # 创建数据加载器（批次处理以节省内存）
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 存储所有预测结果
        all_preds = []
        
        for i, (X_batch,) in enumerate(test_loader):
            # 前向传播
            outputs = model(X_batch)
            all_preds.append(outputs)
            
            if (i + 1) % 10 == 0 or i + 1 == len(test_loader):
                print(f"  已处理 {i + 1}/{len(test_loader)} 个样本")
    
    # 合并所有预测结果
    if all_preds:
        combined_preds = torch.cat(all_preds)
        print(f"预测完成，预测结果形状: {combined_preds.shape}")
        return combined_preds
    else:
        print("预测结果为空")
        return None

def create_chlorophyll_cmap():
    """
    创建叶绿素专用的绿色渐变颜色映射
    使用更浅的绿色配置
    """
    # 使用更浅的绿色渐变配置
    colors = [(0.9, 1.0, 0.9),  # 超浅绿
              (0.7, 0.9, 0.7),  # 浅绿
              (0.5, 0.8, 0.5),  # 中浅绿
              (0.3, 0.7, 0.3)]  # 浅中绿
    return LinearSegmentedColormap.from_list('light_greens', colors, N=256)

# 可视化预测结果
def visualize_results(pred, target, land_mask, num_samples=6):
    """
    可视化预测结果与真实结果的对比
    为前num_samples个样本生成对比图
    """
    print(f"\n生成可视化结果，共 {num_samples} 张对比图")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建保存结果的文件夹
    output_dir = os.path.join(script_dir, 'predictionVis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取经纬度信息
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset')
    spatial_info_path = os.path.join(dataset_dir, 'spatial_info.npz')
    spatial_data = np.load(spatial_info_path)
    lon = spatial_data['lon']
    lat = spatial_data['lat']
    
    # 创建叶绿素专用颜色映射
    cmap = create_chlorophyll_cmap()
    
    # 确保pred是numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(land_mask, torch.Tensor):
        land_mask = land_mask.cpu().numpy()
    
    # 限制样本数量
    num_samples = min(num_samples, len(pred))
    
    # 为每个样本生成对比图
    for sample_idx in range(num_samples):
        plt.figure(figsize=(14, 6))
        
        # 获取预测值（直接使用预测的第一天结果）
        # 预测形状是[6, 8, 7, 600, 480]，我们需要获取正确的2D切片
        pred_day = pred[sample_idx, 0, 0]  # 获取每个样本的第一个预测结果 (600, 480)
        
        # 获取真实值
        target_day = target[sample_idx, :, :, 0]  # 获取每个样本的第一个目标值 (600, 480)
        
        # 应用陆地掩码（将陆地设为NaN以便在可视化中显示为浅灰色）
        pred_day = np.where(land_mask, np.nan, pred_day)
        target_day = np.where(land_mask, np.nan, target_day)
        
        # 创建颜色限制（更细化地展示低浓度叶绿素值）
        # 基于数据统计特性，更好地突出低浓度变化
        pred_min = np.nanmin(pred_day)
        pred_max = np.nanmax(pred_day)
        target_min = np.nanmin(target_day)
        target_max = np.nanmax(target_day)
        
        # 使用稍小于最小值和稍大于最大值的范围，但更关注低浓度区域
        vmin = min(pred_min, target_min)
        vmax = max(pred_max, target_max)
        
        # 如果最大值远大于最小值，我们可以限制上限以更好地显示低浓度区域的细节
        # 例如，如果最大值是最小值的5倍以上，我们可以使用90%分位数作为上限
        if vmax > 5 * abs(vmin):
            # 提取有效数据
            valid_pred = pred_day[~np.isnan(pred_day)]
            valid_target = target_day[~np.isnan(target_day)]
            all_valid = np.concatenate([valid_pred, valid_target])
            # 使用90%分位数作为新的上限
            vmax = np.percentile(all_valid, 90) if len(all_valid) > 0 else vmax
        
        # 设置NaN值的颜色（浅灰色）
        cmap.set_bad(color='lightgray', alpha=0.8)
        
        # 左侧：预测结果
        ax1 = plt.subplot(121, projection=ccrs.PlateCarree())
        ax1.add_feature(cfeature.COASTLINE, linewidth=1.0)
        
        masked_pred = np.ma.masked_where(np.isnan(pred_day), pred_day)
        im1 = ax1.pcolormesh(lon, lat, masked_pred, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax)
        
        gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl1.top_labels = False
        gl1.right_labels = False
        
        # 设置地图范围（中国近海区域）
        ax1.set_extent([100, 140, 0, 50], crs=ccrs.PlateCarree())
        ax1.set_title(f"Prediction Day 1")
        
        # 右侧：真实结果
        ax2 = plt.subplot(122, projection=ccrs.PlateCarree())
        ax2.add_feature(cfeature.COASTLINE, linewidth=1.0)
        
        masked_target = np.ma.masked_where(np.isnan(target_day), target_day)
        im2 = ax2.pcolormesh(lon, lat, masked_target, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=vmin, vmax=vmax)
        
        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl2.top_labels = False
        gl2.right_labels = False
        ax2.set_extent([100, 140, 0, 50], crs=ccrs.PlateCarree())
        ax2.set_title(f"Ground Truth Day 1")
        
        # 设置整体标题
        plt.suptitle(f"Sample {sample_idx + 1} Comparison", y=0.98)  # 调整标题位置
        
        # 将颜色条移到右侧而不是底部，避免覆盖图表内容
        # 调整布局以给右侧颜色条留出空间
        plt.subplots_adjust(left=0.05, right=0.85, top=0.92, bottom=0.1, wspace=0.15)
        
        # 添加右侧垂直颜色条
        cbar_ax = plt.axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(im2, cax=cbar_ax)
        cbar.set_label('Chlorophyll Concentration', fontsize=10)
        cbar.ax.tick_params(labelsize=8)  # 调整颜色条刻度标签大小
        
        # 保存图片（使用英文文件名到专门的文件夹）
        save_path = os.path.join(output_dir, f'predictionVis_{sample_idx + 1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  已保存对比图: {save_path}")
    
    print("可视化完成！")

# 主函数
def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径
    model_path = os.path.join(script_dir, 'stcnn_model.pth')
    
    try:
        # 加载测试数据
        X_test, Y_test, land_mask = load_test_data()
        
        # 预处理测试数据
        X_test_tensor = preprocess_test_data(X_test)
        
        # 加载模型
        model = load_model(model_path, device)
        
        # 进行预测
        predictions = predict(model, X_test_tensor, device)
        
        if predictions is not None:
            # 计算评估指标
            print("\n计算评估指标...")
            metrics = calculate_metrics(predictions, Y_test, land_mask)
            print("测试集评估指标:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  皮尔逊相关系数 (R): {metrics['R']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            
            # 生成可视化结果
            visualize_results(predictions, Y_test, land_mask)
    
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()