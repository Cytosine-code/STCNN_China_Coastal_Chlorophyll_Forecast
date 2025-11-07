import numpy as np
import os

class DatasetChecker:
    def __init__(self, dataset_dir='dataset'):
        """
        初始化数据集检查器
        :param dataset_dir: 数据集目录路径
        """
        self.dataset_dir = dataset_dir
        self.dataset_files = {
            'train': 'train.npz',
            'val': 'val.npz',
            'test': 'test.npz'
        }
        self.spatial_info_file = 'spatial_info.npz'
        
    def check_file_exists(self):
        """
        检查数据集文件是否存在
        """
        for dataset_name, filename in self.dataset_files.items():
            filepath = os.path.join(self.dataset_dir, filename)
            if not os.path.exists(filepath):
                print(f"错误：{dataset_name}数据集文件不存在: {filepath}")
                return False
        
        spatial_filepath = os.path.join(self.dataset_dir, self.spatial_info_file)
        if not os.path.exists(spatial_filepath):
            print(f"错误：空间信息文件不存在: {spatial_filepath}")
            return False
        
        print("✓ 所有数据集文件存在")
        return True
    
    def load_dataset(self, dataset_name):
        """
        加载指定的数据集
        :param dataset_name: 数据集名称 ('train', 'val', 'test')
        :return: 数据集字典
        """
        if dataset_name not in self.dataset_files:
            raise ValueError(f"未知的数据集名称: {dataset_name}")
        
        filepath = os.path.join(self.dataset_dir, self.dataset_files[dataset_name])
        return np.load(filepath)
    
    def load_spatial_info(self):
        """
        加载空间信息
        :return: 空间信息字典
        """
        filepath = os.path.join(self.dataset_dir, self.spatial_info_file)
        return np.load(filepath)
    
    def check_dataset_shape(self, data, dataset_type):
        """
        检查时空数据集的形状
        """
        # 尝试使用不同的键名
        if 'X' in data and 'Y' in data:
            features = data['X']
            labels = data['Y']
            key_type = 'X/Y'
        elif 'features' in data and 'labels' in data:
            features = data['features']
            labels = data['labels']
            key_type = 'features/labels'
        else:
            raise ValueError(f"数据集 {dataset_type} 中未找到特征和标签键")
        
        print(f"\n{dataset_type} 数据集形状 ({key_type}):")
        print(f"特征形状: {features.shape}")
        print(f"标签形状: {labels.shape}")
        
        # 检查维度是否正确（时空结构）
        # 新的特征工程脚本中特征为5维 (samples, lat, lon, time_step, features)
        if len(features.shape) != 5:
            raise ValueError(f"特征应为5维 (samples, lat, lon, time_step, features)，实际为{len(features.shape)}维")
        
        # 标签为4维 (samples, lat, lon, prediction_days)
        if len(labels.shape) != 4:
            raise ValueError(f"标签应为4维 (samples, lat, lon, prediction_days)，实际为{len(labels.shape)}维")
        
        # 验证样本数量一致
        assert features.shape[0] == labels.shape[0], f"样本数量不匹配: 特征={features.shape[0]}, 标签={labels.shape[0]}"
        print(f"✓ 样本数量一致: {features.shape[0]}个样本")
        
        # 检查空间维度是否一致
        assert features.shape[1] == labels.shape[1] and features.shape[2] == labels.shape[2], "空间维度不匹配"
        print(f"✓ 空间维度一致: {features.shape[1]}x{features.shape[2]} (lat x lon)")
        
        # 验证时间步维度
        assert features.shape[3] == 7, f"时间步维度应为7，实际为{features.shape[3]}"
        print(f"✓ 时间步维度正确: {features.shape[3]}")
        
        # 验证特征维度
        assert features.shape[4] == 8, f"特征维度应为8，实际为{features.shape[4]}"
        print(f"✓ 特征维度正确: {features.shape[4]}")
        
        # 验证标签维度
        assert labels.shape[-1] == 7, f"标签维度应为7，实际为{labels.shape[-1]}"
        print(f"✓ 标签维度正确: {labels.shape[-1]}")
        
        return features, labels
    
    def check_first_sample(self, data, dataset_type):
        """
        检查第一个样本的详细信息
        """
        # 尝试使用不同的键名
        if 'X' in data and 'Y' in data:
            features = data['X']
            labels = data['Y']
        elif 'features' in data and 'labels' in data:
            features = data['features']
            labels = data['labels']
        else:
            raise ValueError(f"数据集 {dataset_type} 中未找到特征和标签键")
        
        print(f"\n{dataset_type} 第一个样本信息:")
        
        # 获取第一个样本
        first_feature = features[0]
        first_label = labels[0]
        
        # 检查特征
        print("特征 (第一个样本):")
        print(f"  形状: {first_feature.shape}")
        print(f"  数据类型: {first_feature.dtype}")
        print(f"  最小值: {np.nanmin(first_feature)}")
        print(f"  最大值: {np.nanmax(first_feature)}")
        print(f"  平均值: {np.nanmean(first_feature)}")
        print(f"  NaN值数量: {np.isnan(first_feature).sum()}")
        
        # 检查标签
        print("标签 (第一个样本):")
        print(f"  形状: {first_label.shape}")
        print(f"  数据类型: {first_label.dtype}")
        print(f"  最小值: {np.nanmin(first_label)}")
        print(f"  最大值: {np.nanmax(first_label)}")
        print(f"  平均值: {np.nanmean(first_label)}")
        print(f"  NaN值数量: {np.isnan(first_label).sum()}")
        
        # 选择中心位置
        lat_idx, lon_idx = (first_feature.shape[0] // 2), (first_feature.shape[1] // 2)+100
        print(f"\n选择中心点 (lat_idx={lat_idx}, lon_idx={lon_idx})")
        
        # 显示该空间点的所有时间步和特征
        print(f"\n特征值示例 (空间点[{lat_idx},{lon_idx}]):")
        for t in range(first_feature.shape[2]):  # 时间步
            print(f"  时间步 {t+1}:")
            for i in range(first_feature.shape[3]):  # 特征维度
                feature_name = self._get_feature_name(i)
                value = first_feature[lat_idx, lon_idx, t, i]
                print(f"    特征{i+1} ({feature_name}): {value}")
        
        # 显示该空间点的所有7天预测值
        print(f"\n标签值示例 (空间点[{lat_idx},{lon_idx}]):")
        for i in range(7):
            value = first_label[lat_idx, lon_idx, i]
            print(f"  预测第{i+1}天: {value}")
    
    def _get_feature_name(self, index):
        """
        获取特征名称
        """
        feature_names = [
            "叶绿素均值",
            "叶绿素趋势",
            "海流u分量",
            "海流v分量",
            "铁离子浓度",
            "硝酸盐浓度",
            "磷酸盐浓度",
            "硅酸盐浓度"
        ]
        if 0 <= index < len(feature_names):
            return feature_names[index]
        return f"特征{index+1}"
    
    def check_spatial_info(self):
        """
        检查空间信息
        """
        spatial_info = self.load_spatial_info()
        print("\n空间信息:")
        
        # 检查必要的键
        required_keys = ['lat', 'lon', 'salt_vars', 'space_shape']
        for key in required_keys:
            if key not in spatial_info:
                raise ValueError(f"缺少空间信息键: {key}")
        
        # 显示空间信息
        lat = spatial_info['lat']
        lon = spatial_info['lon']
        salt_vars = spatial_info['salt_vars']
        space_shape = spatial_info['space_shape']
        
        print(f"纬度范围: {lat.min()}° 到 {lat.max()}°")
        print(f"经度范围: {lon.min()}° 到 {lon.max()}°")
        print(f"纬度点数: {len(lat)}")
        print(f"经度点数: {len(lon)}")
        print(f"空间形状: {space_shape}")
        print(f"营养盐变量: {salt_vars}")
        
        # 验证空间形状是否与经纬度数组长度一致
        if len(lat) != space_shape[0] or len(lon) != space_shape[1]:
            raise ValueError(f"空间形状与经纬度数组长度不一致: 空间形状={space_shape}, 经纬度长度=({len(lat)}, {len(lon)})")
        print(f"✓ 空间形状与经纬度数组长度一致")
        
        return spatial_info
    
    def run(self):
        """
        运行完整的检查流程
        """
        print("=== 数据集检查开始 ===")
        
        # 检查文件存在性
        if not self.check_file_exists():
            print("检查失败：文件不存在")
            return False
        
        try:
            # 检查空间信息
            self.check_spatial_info()
            
            # 检查每个数据集
            all_valid = True
            for dataset_name in ['train', 'val', 'test']:
                try:
                    print(f"\n=== 检查 {dataset_name} 数据集 ===")
                    data = self.load_dataset(dataset_name)
                    
                    # 检查数据集形状
                    self.check_dataset_shape(data, dataset_name)
                    
                    # 检查第一个样本
                    self.check_first_sample(data, dataset_name)
                    
                except Exception as e:
                    print(f"检查 {dataset_name} 数据集时出错: {str(e)}")
                    all_valid = False
            
            print("\n" + "=" * 50)
            if all_valid:
                print("✅ 所有检查通过！数据集满足STCNN时空特征工程要求。")
                print("特征形状: (samples, lat, lon, time_step, features)")
                print("标签形状: (samples, lat, lon, prediction_days)")
                return True
            else:
                print("❌ 部分检查失败，请检查上面的详细信息。")
                return False
                
        except Exception as e:
            print(f"检查空间信息时出错: {str(e)}")
            return False

if __name__ == "__main__":
    checker = DatasetChecker()
    checker.run()