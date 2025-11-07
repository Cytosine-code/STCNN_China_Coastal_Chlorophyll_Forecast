import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from netCDF4 import Dataset
from matplotlib.colors import LinearSegmentedColormap
import glob

class DataVisualizer:
    
    def __init__(self, output_dir='visualization_results'):
        # 定义中国近海区域范围
        self.lon_min, self.lon_max = 110, 135
        self.lat_min, self.lat_max = 15, 45
        # 创建保存结果的文件夹
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 定义标准填充值
        self.fill_value = 9.96921e+36
    
    # 通用数据加载函数
    def load_data(self, file_path, var_name=None):
        """从文件加载数据，返回经纬度和数据数组"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        print(f"正在读取文件: {file_path}")
        
        try:
            with Dataset(file_path, 'r') as nc:
                # 尝试不同的经纬度变量名
                lon = nc.variables['lon'][:] if 'lon' in nc.variables else nc.variables['longitude'][:]
                lat = nc.variables['lat'][:] if 'lat' in nc.variables else nc.variables['latitude'][:]
                
                # 如果没有指定变量名，直接返回None
                if var_name is None:
                    print("必须指定变量名")
                    return None
                
                if var_name not in nc.variables:
                    print(f"变量 {var_name} 不存在于文件中")
                    return None
                
                data = nc.variables[var_name][:]
                
                # 检查是否有_FillValue属性
                if '_FillValue' in nc.variables[var_name].ncattrs():
                    self.fill_value = nc.variables[var_name]._FillValue
                    print(f"从文件中读取填充值: {self.fill_value}")
                
                # 处理数据形状
                if len(data.shape) == 3:
                    data = data[0]  # 对于(time, lat, lon)格式
                elif len(data.shape) == 4:
                    data = data[0, 0]  # 对于(time, depth, lat, lon)格式
                
                # 将填充值转换为NaN
                if hasattr(data, 'mask'):
                    data = data.filled(np.nan)
                else:
                    if isinstance(self.fill_value, (int, float)) and not np.isnan(self.fill_value):
                        data = np.where(np.isclose(data, self.fill_value, rtol=1e-7), np.nan, data)
                
                print(f"加载的数据变量: {var_name}")
                print(f"数据形状: {data.shape}")
            
            # 筛选中国近海区域
            lon_mask = (lon >= self.lon_min) & (lon <= self.lon_max)
            lat_mask = (lat >= self.lat_min) & (lat <= self.lat_max)
            
            lon_idx = np.where(lon_mask)[0]
            lat_idx = np.where(lat_mask)[0]
            
            if len(lon_idx) > 0 and len(lat_idx) > 0:
                lon_subset = lon[lon_mask]
                lat_subset = lat[lat_mask]
                data_subset = data[np.ix_(lat_idx, lon_idx)]
                
                return lon_subset, lat_subset, data_subset
            else:
                print("没有找到中国近海区域的数据")
                return None
                
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            return None
    
    # 通用可视化函数
    def _create_visualization(self, lon, lat, data, file_path, var_name, data_name, cmap):
        """创建基础可视化"""
        # 创建可视化
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 添加海岸线背景
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        
        # 创建掩码数组
        masked_data = np.ma.masked_where(np.isnan(data), data)
        
        # 设置NaN值的颜色
        cmap.set_bad(color='lightgray', alpha=0.5)
        
        # 找到有效数据的范围
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            # 根据数据分布设置颜色范围
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            
            # 处理不同数据范围的情况
            if data_max > 0 and data_min > 0 and data_max / data_min > 100:
                log_data = np.log10(valid_data)
                vmin_log = np.percentile(log_data, 0.01)
                vmax_log = np.percentile(log_data, 99.9)
                vmin = 10 ** vmin_log
                vmax = 10 ** vmax_log
            else:
                vmin = np.percentile(valid_data, 0.01)
                vmax = np.percentile(valid_data, 99.9)
                
            # 对于浓度数据确保最小值不为负
            if data_name != 'current':
                vmin = max(vmin, 0)
            
            # 如果数据范围太小，手动扩展
            if vmax - vmin < 1e-10:
                vmin = max(0, vmax - 0.5)
                vmax = vmax + 0.5
        else:
            vmin, vmax = 0, 1
        
        # 显示数据
        im = ax.pcolormesh(lon, lat, masked_data, transform=ccrs.PlateCarree(), 
                          cmap=cmap, vmin=vmin, vmax=vmax)
        
        # 添加经纬度网格
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # 设置地图范围
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max], crs=ccrs.PlateCarree())
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(f'{var_name.capitalize()} Value')
        
        # 添加标题
        plt.title(f'{data_name.capitalize()} ({var_name}) - {os.path.basename(file_path)}')
        
        # 保存图像
        output_file = os.path.join(self.output_dir, f'{data_name}_{var_name}_{os.path.splitext(os.path.basename(file_path))[0]}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"图像已保存至: {output_file}")
        
        # 计算统计信息
        total_cells = data.size
        nan_cells = np.isnan(data).sum()
        valid_cells = total_cells - nan_cells
        
        nan_percentage = (nan_cells / total_cells) * 100 if total_cells > 0 else 0
        valid_percentage = (valid_cells / total_cells) * 100 if total_cells > 0 else 0
        
        print(f"总单元格数: {total_cells}")
        print(f"有效数据单元格数: {valid_cells} ({valid_percentage:.2f}%)")
        print(f"NaN值单元格数: {nan_cells} ({nan_percentage:.2f}%)")
        print(f"使用的填充值阈值: {self.fill_value}")
    
    # 1. 叶绿素数据可视化函数
    def visualize_chlorophyll(self, file_path):
        """可视化叶绿素数据"""
        print(f"\n正在处理叶绿素文件: {os.path.basename(file_path)}")
        
        # 加载数据
        data = self.load_data(file_path, var_name='chlor_a')
        if data is None:
            return
            
        lon, lat, chl_data = data
        
        # 创建叶绿素专用颜色映射
        colors = [(0.7, 0.9, 0.7),  # 浅绿
                  (0.3, 0.8, 0.3),  # 中绿
                  (0.0, 0.6, 0.0),  # 深绿
                  (0.0, 0.4, 0.0)]  # 更深绿
        cmap = LinearSegmentedColormap.from_list('deep_greens', colors, N=256)
        
        # 创建可视化
        self._create_visualization(lon, lat, chl_data, file_path, 'chlor_a', 'chlorophyll', cmap)
    
    # 2. 海流数据可视化函数
    def visualize_current(self, file_path):
        """可视化海流数据，分别显示uo和vo"""
        print(f"\n正在处理海流文件: {os.path.basename(file_path)}")
        
        # 创建海流专用颜色映射（发散色图）
        colors = [(0.2, 0.3, 0.7),  # 深蓝色（负速度）
                  (1.0, 1.0, 1.0),  # 白色（零）
                  (0.7, 0.2, 0.2)]  # 红色（正速度）
        cmap = LinearSegmentedColormap.from_list('current_cmap', colors, N=256)
        
        # 可视化u分量
        print("\n处理u分量:")
        u_data = self.load_data(file_path, var_name='uo')
        if u_data is not None:
            lon, lat, uo_data = u_data
            self._create_visualization(lon, lat, uo_data, file_path, 'uo', 'current', cmap)
        
        # 可视化v分量
        print("\n处理v分量:")
        v_data = self.load_data(file_path, var_name='vo')
        if v_data is not None:
            lon, lat, vo_data = v_data
            self._create_visualization(lon, lat, vo_data, file_path, 'vo', 'current', cmap)
    
    # 3. 营养盐数据可视化函数（分开表示四种成分）
    def visualize_nutrient(self, file_path):
        """可视化营养盐数据，分别显示四种成分"""
        print(f"\n正在处理营养盐文件: {os.path.basename(file_path)}")
        
        # 创建营养盐专用颜色映射
        nutrient_cmaps = {
            'fe': LinearSegmentedColormap.from_list('fe_cmap', [(0.9, 0.9, 1.0), (0.6, 0.3, 0.8)], N=256),
            'no3': LinearSegmentedColormap.from_list('no3_cmap', [(0.9, 0.9, 1.0), (0.4, 0.2, 0.7)], N=256),
            'po4': LinearSegmentedColormap.from_list('po4_cmap', [(0.9, 0.9, 1.0), (0.3, 0.1, 0.6)], N=256),
            'si': LinearSegmentedColormap.from_list('si_cmap', [(0.9, 0.9, 1.0), (0.2, 0.1, 0.5)], N=256)
        }
        
        # 分别可视化四种营养盐成分
        for nutrient_type, cmap in nutrient_cmaps.items():
            print(f"\n处理{nutrient_type}:")
            data = self.load_data(file_path, var_name=nutrient_type)
            if data is not None:
                lon, lat, nut_data = data
                self._create_visualization(lon, lat, nut_data, file_path, nutrient_type, 'nutrient', cmap)
    
    # 批量处理函数
    def process_files(self, file_paths, data_type):
        """批量处理指定类型的文件"""
        if data_type == 'chlorophyll':
            cnt=0
            for file_path in file_paths:
                self.visualize_chlorophyll(file_path)
                cnt+=1
                if(cnt==5):
                    break
        elif data_type == 'current':
            cnt=0
            for file_path in file_paths:
                self.visualize_current(file_path)
                cnt+=1
                if(cnt==5):
                    break
        elif data_type == 'nutrient':
            cnt=0
            for file_path in file_paths:
                self.visualize_nutrient(file_path)
                cnt+=1
                if(cnt==5):
                    break
        print(f"\n{data_type} 数据可视化完成！")
    
    # 目录处理函数
    def process_directory(self, directory, data_type):
        """处理目录中的所有nc文件"""
        if not os.path.exists(directory):
            print(f"目录 {directory} 不存在")
            return []
        
        file_paths = sorted(glob.glob(os.path.join(directory, '*.nc')))
        if not file_paths:
            print(f"在目录 {directory} 中未找到NetCDF文件")
            return []
        
        print(f"找到 {len(file_paths)} 个文件")
        return file_paths

# 示例用法
if __name__ == "__main__":
    # 创建可视化器实例
    visualizer = DataVisualizer()
    
    # 选择要运行的可视化类型
    # 1: 叶绿素数据可视化
    # 2: 海流数据可视化
    # 3: 营养盐数据可视化（四种成分分开显示）
    visualization_type = 2  # 可以修改这个值来切换不同的可视化类型
    
    if visualization_type == 1:
        # 叶绿素数据可视化
        print("=== 开始叶绿素数据可视化 ===")
        chl_dir = 'data/Chl'
        file_paths = visualizer.process_directory(chl_dir, 'chlorophyll')
        
        if file_paths:
            visualizer.process_files(file_paths, 'chlorophyll')
        else:
            print("在目录中未找到叶绿素文件")
    
    elif visualization_type == 2:
        # 海流数据可视化
        print("=== 开始海流数据可视化 ===")
        current_dir = 'data/OC'
        file_paths = visualizer.process_directory(current_dir, 'current')
        
        if file_paths:
            visualizer.process_files(file_paths, 'current')
        else:
            print("在目录中未找到海流文件")
    
    elif visualization_type == 3:
        # 营养盐数据可视化（四种成分分开显示）
        print("=== 开始营养盐数据可视化 ===")
        nutrient_dir = 'data/salt'
        file_paths = visualizer.process_directory(nutrient_dir, 'nutrient')
        
        if file_paths:
            visualizer.process_files(file_paths, 'nutrient')
        else:
            print("在目录中未找到营养盐文件")
    
    print("\n所有可视化任务完成！")