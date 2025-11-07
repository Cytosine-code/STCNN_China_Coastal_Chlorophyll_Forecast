import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta
import gc  # 导入垃圾回收模块

class FeatureEngineering:
    def __init__(self):
        self.data_dir = 'data'
        self.output_dir = 'dataset'
        self.time_step = 7  # 使用过去7天的数据
        self.prediction_days = 7  # 预测未来7天
        self.batch_size = 5  # 批处理大小，每处理5个样本保存一次
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_date_range(self, start_date_str, end_date_str):
        """生成日期字符串列表"""
        start_date = datetime.strptime(start_date_str, '%Y%m%d')
        end_date = datetime.strptime(end_date_str, '%Y%m%d')
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        return date_range
    
    def load_data_by_date(self, date_str):
        """加载指定日期的所有数据"""
        try:
            # 加载叶绿素数据
            chl_file = os.path.join(self.data_dir, 'Chl', f'chl_{date_str}.nc')
            if not os.path.exists(chl_file):
                print(f"叶绿素文件不存在: {chl_file}")
                return None
            
            ds_chl = xr.open_dataset(chl_file)
            chl_var = 'chlor_a'
            chl_data = ds_chl[chl_var].values
            
            # 处理可能的三维数据（时间+空间），只取第一个时间点
            if len(chl_data.shape) == 3:
                chl_data = chl_data[0]
            
            # 加载海流数据
            cur_file = os.path.join(self.data_dir, 'OC', f'cur_{date_str}.nc')
            if not os.path.exists(cur_file):
                print(f"海流文件不存在: {cur_file}")
                return None
            
            ds_cur = xr.open_dataset(cur_file)
            uo_data = ds_cur['uo'].values
            vo_data = ds_cur['vo'].values
            
            # 处理可能的三维数据
            if len(uo_data.shape) == 3:
                uo_data = uo_data[0]
                vo_data = vo_data[0]
            
            # 加载营养盐数据
            salt_file = os.path.join(self.data_dir, 'salt', f'salt_{date_str}.nc')
            if not os.path.exists(salt_file):
                print(f"营养盐文件不存在: {salt_file}")
                return None
            
            ds_salt = xr.open_dataset(salt_file)
            # 获取营养盐变量（排除坐标变量）
            salt_vars = [var for var in ds_salt.data_vars if var not in ['lat', 'lon', 'time']]
            salt_data = {}
            
            for var in salt_vars:
                var_data = ds_salt[var].values
                # 处理可能的三维数据
                if len(var_data.shape) == 3:
                    var_data = var_data[0]
                salt_data[var] = var_data
            
            # 关闭数据集
            ds_chl.close()
            ds_cur.close()
            ds_salt.close()
            
            return {
                'chlorophyll': chl_data,
                'uo': uo_data,
                'vo': vo_data,
                'salt_vars': salt_vars,
                'salt_data': salt_data,
                'lat': ds_chl.lat.values,
                'lon': ds_chl.lon.values
            }
        except Exception as e:
            print(f"加载日期{date_str}的数据时出错: {e}")
            return None
    
    def extract_features(self, date_list):
        """提取特征和标签 - 保持时空结构用于STCNN，使用批处理优化内存"""
        # 获取空间维度信息（从第一个有效日期的数据中）
        spatial_info = None
        temp_data = self.load_data_by_date(date_list[0])
        if temp_data is not None:
            # 获取数据形状，确保我们只取空间维度
            chl_shape = temp_data['chlorophyll'].shape
            if len(chl_shape) > 2:
                # 如果有时间维度，只取空间维度
                shape = chl_shape[-2:]
            else:
                shape = chl_shape
            
            spatial_info = {
                'lat': temp_data['lat'],
                'lon': temp_data['lon'],
                'space_shape': shape,
                'salt_vars': temp_data['salt_vars']
            }
            # 释放临时数据内存
            del temp_data
            gc.collect()
        
        if spatial_info is None:
            print("无法获取空间维度信息，退出程序")
            return None
        
        lat_len, lon_len = spatial_info['space_shape']
        
        # 总样本数
        total_samples = len(date_list) - self.time_step - self.prediction_days + 1
        
        # 预先创建批处理输出文件
        batch_dir = os.path.join(self.output_dir, 'batches')
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        
        # 记录所有批处理文件
        batch_files = []
        
        # 当前批处理的特征和标签
        batch_features = []
        batch_labels = []
        
        # 遍历所有可能的起始日期（需要保证有过去7天和未来7天的数据）
        for i in range(self.time_step, len(date_list) - self.prediction_days + 1):
            sample_idx = i - self.time_step
            print(f"处理样本 {sample_idx + 1}/{total_samples}")
            
            # 获取过去7天的日期
            past_dates = date_list[i - self.time_step:i]
            
            # 获取未来7天的日期（用于标签）
            future_dates = date_list[i:i + self.prediction_days]
            
            # 存储过去7天的数据
            past_chl_data = []
            past_uo_data = []
            past_vo_data = []
            past_salt_data = {var: [] for var in spatial_info['salt_vars']}
            
            # 加载过去7天的数据
            valid_past_data = True
            for date in past_dates:
                data = self.load_data_by_date(date)
                if data is None:
                    valid_past_data = False
                    break
                
                past_chl_data.append(data['chlorophyll'])
                past_uo_data.append(data['uo'])
                past_vo_data.append(data['vo'])
                
                for var in spatial_info['salt_vars']:
                    if var in data['salt_data']:
                        past_salt_data[var].append(data['salt_data'][var])
                
                # 释放单个日期的数据
                del data
            
            if not valid_past_data:
                print(f"跳过日期 {past_dates[0]} 至 {past_dates[-1]}：缺失数据")
                # 清空已加载的数据
                del past_chl_data, past_uo_data, past_vo_data, past_salt_data
                gc.collect()
                continue
            
            # 转换为numpy数组 [time, lat, lon]
            chl_stack = np.stack(past_chl_data)  # 形状: [7, lat, lon]
            
            # 释放原始列表数据
            del past_chl_data
            
            # 构建特征张量 [lat, lon, time_step, features]
            features = np.zeros((lat_len, lon_len, self.time_step, 8))
            
            # 为每个时间步填充特征
            for t in range(self.time_step):
                # 1. 到该时间步为止的历史均值（计算前t+1个时间步的均值）
                mean_temp = np.zeros((lat_len, lon_len))
                for lat_idx in range(lat_len):  # 使用不同的变量名避免与外部循环冲突
                    for lon_idx in range(lon_len):
                        ts = chl_stack[:t+1, lat_idx, lon_idx]
                        # 检查是否有非NaN值
                        if np.any(~np.isnan(ts)):
                            mean_temp[lat_idx, lon_idx] = np.nanmean(ts)
                        else:
                            mean_temp[lat_idx, lon_idx] = 0  # 全部是NaN时设置为0
                features[..., t, 0] = mean_temp
                
                # 释放临时数组
                del mean_temp
                
                # 2. 到该时间步为止的趋势
                time_indices = np.arange(t+1)
                trend_temp = np.zeros((lat_len, lon_len))
                
                for lat_idx in range(lat_len):
                    for lon_idx in range(lon_len):
                        ts = chl_stack[:t+1, lat_idx, lon_idx]
                        if np.all(np.isnan(ts)):
                            trend_temp[lat_idx, lon_idx] = 0
                            continue
                        
                        valid_idx = ~np.isnan(ts)
                        valid_count = np.sum(valid_idx)
                        
                        if valid_count > 1:  # 有多个有效点时计算趋势
                            try:
                                slope = np.polyfit(time_indices[valid_idx], ts[valid_idx], 1)[0]
                                trend_temp[lat_idx, lon_idx] = slope
                            except:
                                trend_temp[lat_idx, lon_idx] = 0
                
                features[..., t, 1] = trend_temp
                
                # 释放临时数组
                del trend_temp
                
                # 3-4. 海流数据 - 将NaN值替换为0
                uo_data = past_uo_data[t].copy()
                uo_data[np.isnan(uo_data)] = 0
                features[..., t, 2] = uo_data
                
                vo_data = past_vo_data[t].copy()
                vo_data[np.isnan(vo_data)] = 0
                features[..., t, 3] = vo_data
                
                # 5-8. 营养盐数据 - 将NaN值替换为0
                for var, idx in zip(['fe', 'no3', 'po4', 'si'], [4, 5, 6, 7]):
                    if var in past_salt_data and len(past_salt_data[var]) > t:
                        salt_data = past_salt_data[var][t].copy()
                        salt_data[np.isnan(salt_data)] = 0
                        features[..., t, idx] = salt_data
            
            # 将特征张量中所有NaN值替换为0
            features[np.isnan(features)] = 0
            
            # 释放海流和营养盐数据
            del past_uo_data, past_vo_data, past_salt_data
            del chl_stack
            
            # 加载未来7天的叶绿素数据作为标签
            future_chl_data = []
            valid_future_data = True
            for date in future_dates:
                data = self.load_data_by_date(date)
                if data is None:
                    valid_future_data = False
                    break
                future_chl_data.append(data['chlorophyll'])
                # 释放单个日期的数据
                del data
            
            if not valid_future_data:
                # 释放特征数据
                del features
                gc.collect()
                continue
            
            # 构建标签张量 [lat, lon, prediction_days]
            labels = np.stack(future_chl_data, axis=-1)  # 形状: [lat, lon, 7]
            
            # 将标签张量中所有NaN值替换为0（陆地或无效区域）
            labels[np.isnan(labels)] = 0
            
            # 释放原始列表数据
            del future_chl_data
            
            # 添加到当前批处理
            batch_features.append(features)
            batch_labels.append(labels)
            
            # 释放当前样本的数据
            del features, labels
            
            # 当批处理达到指定大小时，保存并清空
            if len(batch_features) >= self.batch_size or sample_idx == total_samples - 1:
                # 转换为numpy数组
                batch_features_array = np.stack(batch_features) if batch_features else np.array([])
                batch_labels_array = np.stack(batch_labels) if batch_labels else np.array([])
                
                # 保存批处理文件
                batch_file = os.path.join(batch_dir, f'batch_{len(batch_files)}.npz')
                np.savez(batch_file, features=batch_features_array, labels=batch_labels_array)
                batch_files.append(batch_file)
                
                print(f"保存批处理 {len(batch_files)}，包含 {len(batch_features)} 个样本")
                
                # 清空批处理并释放内存
                del batch_features, batch_labels, batch_features_array, batch_labels_array
                batch_features = []
                batch_labels = []
                gc.collect()
        
        print(f"特征提取完成，共生成 {len(batch_files)} 个批处理文件")
        
        return {
            'batch_files': batch_files,
            'spatial_info': spatial_info
        }
    
    def split_dataset(self, all_data, train_end_date, val_end_date):
        """分割训练集、验证集和测试集 - 返回索引范围而不是加载所有数据"""
        # 生成完整的日期范围
        full_date_list = self.get_date_range('20250401', '20250531')
        
        # 计算每个集的起始和结束索引
        total_samples = len(full_date_list) - self.time_step - self.prediction_days + 1
        
        train_start_idx = 0  # 第一个样本
        train_end_idx = full_date_list.index(train_end_date) - self.time_step - self.prediction_days + 1
        
        val_start_idx = train_end_idx + 1
        val_end_idx = full_date_list.index(val_end_date) - self.time_step - self.prediction_days + 1
        
        test_start_idx = val_end_idx + 1
        test_end_idx = total_samples - 1  # 最后一个样本
        
        # 确保索引有效
        train_end_idx = min(train_end_idx, total_samples - 1)
        val_end_idx = min(val_end_idx, total_samples - 1)
        test_end_idx = min(test_end_idx, total_samples - 1)
        
        # 统计每个集的样本数
        train_samples = train_end_idx - train_start_idx + 1
        val_samples = val_end_idx - val_start_idx + 1
        test_samples = test_end_idx - test_start_idx + 1
        
        print(f"数据集分割: 训练集 {train_samples} 样本, 验证集 {val_samples} 样本, 测试集 {test_samples} 样本")
        
        # 返回索引信息和批处理文件列表，而不是加载所有数据
        return {
            'batch_files': all_data['batch_files'],
            'train_indices': (train_start_idx, train_end_idx),
            'val_indices': (val_start_idx, val_end_idx),
            'test_indices': (test_start_idx, test_end_idx)
        }
    
    def save_dataset(self, dataset_split, spatial_info):
        """保存数据集 - 直接从批处理文件读取并保存，避免一次性加载所有数据"""
        # 保存空间信息
        np.savez(os.path.join(self.output_dir, 'spatial_info.npz'),
                 lat=spatial_info['lat'],
                 lon=spatial_info['lon'],
                 space_shape=spatial_info['space_shape'],
                 salt_vars=np.array(spatial_info['salt_vars']))
        
        # 从批处理文件中提取并保存特定范围的样本
        def save_samples_by_indices(batch_files, start_idx, end_idx, output_file):
            print(f"正在处理 {output_file}，样本范围: {start_idx} 到 {end_idx}")
            
            # 首先创建临时文件来保存每个符合条件的样本
            temp_files = []
            sample_count = 0
            current_idx = 0
            
            for batch_file in batch_files:
                batch = np.load(batch_file)
                batch_features = batch['features']
                batch_labels = batch['labels']
                batch_size = len(batch_features)
                
                # 计算这个batch中需要提取的样本范围
                batch_start = max(0, start_idx - current_idx)
                batch_end = min(batch_size, end_idx - current_idx + 1)
                
                if batch_start < batch_end:
                    # 提取这个batch中的相关样本
                    selected_features = batch_features[batch_start:batch_end]
                    selected_labels = batch_labels[batch_start:batch_end]
                    
                    # 保存到临时文件
                    temp_file = f"{output_file}_temp_{sample_count}.npz"
                    np.savez(temp_file, features=selected_features, labels=selected_labels)
                    temp_files.append(temp_file)
                    sample_count += len(selected_features)
                
                current_idx += batch_size
                if current_idx > end_idx:
                    break
            
            if not temp_files:
                # 没有找到样本，创建空文件
                np.savez(output_file, features=np.array([]), labels=np.array([]))
                return
            
            # 逐块合并到最终文件
            print(f"合并 {len(temp_files)} 个临时文件到 {output_file}")
            
            # 第一次保存时创建文件
            first_temp = np.load(temp_files[0])
            first_features = first_temp['features']
            first_labels = first_temp['labels']
            
            # 如果只有一个临时文件，直接重命名
            if len(temp_files) == 1:
                # 关闭文件再操作
                del first_temp, first_features, first_labels
                # 删除可能存在的目标文件
                if os.path.exists(output_file):
                    os.remove(output_file)
                # 重命名临时文件
                os.rename(temp_files[0], output_file)
            else:
                # 多个临时文件，需要合并
                # 创建一个新文件，先写入第一个临时文件的数据
                np.savez(output_file, features=first_features, labels=first_labels)
                
                # 逐次追加剩余临时文件的数据
                for i in range(1, len(temp_files)):
                    temp = np.load(temp_files[i])
                    temp_features = temp['features']
                    temp_labels = temp['labels']
                    
                    # 加载现有数据
                    existing = np.load(output_file)
                    existing_features = existing['features']
                    existing_labels = existing['labels']
                    
                    # 合并数据
                    merged_features = np.concatenate([existing_features, temp_features], axis=0)
                    merged_labels = np.concatenate([existing_labels, temp_labels], axis=0)
                    
                    # 保存合并后的数据
                    np.savez(output_file, features=merged_features, labels=merged_labels)
                    
                    # 释放内存
                    del temp, temp_features, temp_labels
                    del existing, existing_features, existing_labels
                    del merged_features, merged_labels
                    gc.collect()
            
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        print(f"警告: 无法删除临时文件 {temp_file}")
            
            # 验证保存的文件
            try:
                saved = np.load(output_file)
                saved_features = saved['features']
                print(f"成功保存 {len(saved_features)} 个样本到 {output_file}")
                del saved, saved_features
            except Exception as e:
                print(f"警告: 验证文件 {output_file} 时出错: {e}")
        
        # 保存各个数据集
        batch_files = dataset_split['batch_files']
        
        # 保存训练集
        train_file = os.path.join(self.output_dir, 'train.npz')
        train_start, train_end = dataset_split['train_indices']
        save_samples_by_indices(batch_files, train_start, train_end, train_file)
        
        # 保存验证集
        val_file = os.path.join(self.output_dir, 'val.npz')
        val_start, val_end = dataset_split['val_indices']
        save_samples_by_indices(batch_files, val_start, val_end, val_file)
        
        # 保存测试集
        test_file = os.path.join(self.output_dir, 'test.npz')
        test_start, test_end = dataset_split['test_indices']
        save_samples_by_indices(batch_files, test_start, test_end, test_file)
        
        print(f"数据集保存完成！")
    
    def load_existing_batches(self):
        """加载已存在的批处理文件和空间信息"""
        batch_dir = os.path.join(self.output_dir, 'batches')
        
        # 检查批处理目录是否存在
        if not os.path.exists(batch_dir):
            print(f"批处理目录 {batch_dir} 不存在")
            return None
        
        # 获取所有批处理文件
        batch_files = sorted([
            os.path.join(batch_dir, f) 
            for f in os.listdir(batch_dir) 
            if f.endswith('.npz')
        ])
        
        if not batch_files:
            print("没有找到批处理文件")
            return None
        
        # 尝试加载第一个批处理文件来获取空间信息
        # 或者尝试从已保存的spatial_info.npz文件加载
        spatial_info_path = os.path.join(self.output_dir, 'spatial_info.npz')
        spatial_info = None
        
        if os.path.exists(spatial_info_path):
            print("从保存的文件加载空间信息...")
            try:
                loaded = np.load(spatial_info_path)
                spatial_info = {
                    'lat': loaded['lat'],
                    'lon': loaded['lon'],
                    'space_shape': tuple(loaded['space_shape']),
                    'salt_vars': loaded['salt_vars'].tolist()
                }
                del loaded
                print("空间信息加载成功")
            except Exception as e:
                print(f"加载空间信息失败: {e}")
        
        # 如果无法从文件加载空间信息，尝试从第一个批处理文件获取
        if spatial_info is None:
            print("从第一个批处理文件获取空间信息...")
            try:
                # 加载第一个日期的数据来获取空间信息
                date_list = self.get_date_range('20250401', '20250401')
                temp_data = self.load_data_by_date(date_list[0])
                if temp_data is not None:
                    chl_shape = temp_data['chlorophyll'].shape
                    if len(chl_shape) > 2:
                        shape = chl_shape[-2:]
                    else:
                        shape = chl_shape
                    
                    spatial_info = {
                        'lat': temp_data['lat'],
                        'lon': temp_data['lon'],
                        'space_shape': shape,
                        'salt_vars': temp_data['salt_vars']
                    }
                    del temp_data
                    gc.collect()
                    print("从数据文件获取空间信息成功")
            except Exception as e:
                print(f"从数据文件获取空间信息失败: {e}")
        
        if spatial_info is None:
            print("无法获取空间信息")
            return None
        
        print(f"找到 {len(batch_files)} 个批处理文件")
        return {
            'batch_files': batch_files,
            'spatial_info': spatial_info
        }
    
    def run(self):
        """运行特征工程流程"""
        print("开始特征工程...")
        
        # 选择模式：直接使用现有批处理文件进行分割和保存
        print("尝试加载现有批处理文件...")
        all_data = self.load_existing_batches()
        
        if all_data is None:
            # 如果没有找到批处理文件，可以选择退出或运行完整的特征提取
            print("无法加载现有批处理文件，请先运行完整的特征提取")
            # 以下是原始的特征提取代码，取消注释即可进行特征工程
            date_list = self.get_date_range('20250401', '20250531')
            all_data = self.extract_features(date_list)
            if all_data is None:
                print("特征提取失败")
                return
        
        # 分割数据集
        print("分割数据集...")
        dataset_split = self.split_dataset(
            all_data, '20250520', '20250525')
        
        # 保存数据集
        self.save_dataset(dataset_split, all_data['spatial_info'])
        
        print("数据集分割和保存完成！")

if __name__ == "__main__":
    feature_engineer = FeatureEngineering()
    feature_engineer.run()