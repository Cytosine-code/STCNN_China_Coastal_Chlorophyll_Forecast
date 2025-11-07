import numpy as np
import os
from netCDF4 import Dataset

class LandMaskGenerator:
    def __init__(self, data_dir='data/Chl', output_dir='dataset'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fill_value = 9.96921e+36
        self.land_mask = None
        self.lon = None
        self.lat = None
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def find_all_nc_files(self):
        """
        查找目录中所有的.nc文件
        """
        nc_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.nc'):
                nc_files.append(os.path.join(self.data_dir, file))
        return sorted(nc_files)
    
    def load_data_file(self, file_path):
        """
        加载单个数据文件
        """
        try:
            with Dataset(file_path, 'r') as nc:
                # 获取经纬度
                if self.lon is None:
                    self.lon = nc.variables['lon'][:] if 'lon' in nc.variables else nc.variables['longitude'][:]
                if self.lat is None:
                    self.lat = nc.variables['lat'][:] if 'lat' in nc.variables else nc.variables['latitude'][:]
                
                # 获取叶绿素数据
                chl_var_names = ['chlorophyll', 'chlor_a', 'CHL', 'chl']
                chl = None
                for var_name in chl_var_names:
                    if var_name in nc.variables:
                        chl = nc.variables[var_name][:]
                        # 检查是否有_FillValue属性
                        if '_FillValue' in nc.variables[var_name].ncattrs():
                            self.fill_value = nc.variables[var_name]._FillValue
                        break
                
                if chl is None:
                    print(f"未找到叶绿素变量，可用变量: {list(nc.variables.keys())}")
                    return None
                
                # 处理数据形状
                if len(chl.shape) == 3:
                    chl = chl[0]
                elif len(chl.shape) == 4:
                    chl = chl[0, 0]
                
                # 将填充值转换为NaN
                if hasattr(chl, 'mask'):
                    chl = chl.filled(np.nan)
                else:
                    chl = np.where(np.isclose(chl, self.fill_value, rtol=1e-7), np.nan, chl)
                
                return chl
        except Exception as e:
            print(f"读取文件时出错 {file_path}: {str(e)}")
            return None
    
    def generate_land_mask(self):
        """
        生成陆地掩码，将所有日期都是NaN的区域识别为陆地
        """
        print("开始生成陆地掩码...")
        
        # 获取所有NC文件
        nc_files = self.find_all_nc_files()
        if not nc_files:
            print(f"在目录 {self.data_dir} 中未找到NC文件")
            return False
        
        print(f"找到 {len(nc_files)} 个数据文件")
        
        # 初始化陆地掩码（假设所有区域都是海洋）
        # 先加载第一个文件获取形状
        first_chl = self.load_data_file(nc_files[0])
        if first_chl is None:
            return False
        
        # 初始陆地掩码：False表示海洋，True表示陆地
        # 开始假设所有区域都是海洋
        self.land_mask = np.zeros_like(first_chl, dtype=bool)
        
        # 跟踪每个单元格的NaN出现情况
        nan_count = np.zeros_like(first_chl, dtype=int)
        total_files = len(nc_files)
        
        # 处理第一个文件
        nan_count[np.isnan(first_chl)] += 1
        
        # 处理剩余文件
        for i, file_path in enumerate(nc_files[1:], 2):
            print(f"处理文件 {i}/{total_files}: {os.path.basename(file_path)}")
            chl = self.load_data_file(file_path)
            if chl is None:
                continue
            
            # 更新NaN计数
            nan_count[np.isnan(chl)] += 1
        
        # 将在所有文件中都是NaN的区域标记为陆地
        self.land_mask = (nan_count == total_files)
        
        # 统计陆地和海洋区域
        total_cells = self.land_mask.size
        land_cells = np.sum(self.land_mask)
        ocean_cells = total_cells - land_cells
        
        print(f"陆地掩码生成完成！")
        print(f"总单元格数: {total_cells}")
        print(f"陆地单元格数: {land_cells} ({land_cells/total_cells*100:.2f}%)")
        print(f"海洋单元格数: {ocean_cells} ({ocean_cells/total_cells*100:.2f}%)")
        
        return True
    
    def save_land_mask(self, output_file='land_mask.npz'):
        """
        保存陆地掩码为NPZ格式，包含lat, lon和island数组
        """
        if self.land_mask is None:
            print("请先生成陆地掩码")
            return False
        
        # 创建island数组：True表示陆地，False表示海洋
        island = self.land_mask
        
        # 构建完整的输出路径
        full_output_path = os.path.join(self.output_dir, output_file)
        
        # 保存为NPZ文件
        np.savez(full_output_path, lat=self.lat, lon=self.lon, island=island)
        print(f"陆地掩码已保存至: {full_output_path}")
        print(f"数据形状: lat={len(self.lat)}, lon={len(self.lon)}, island={island.shape}")
        
        # 验证保存的掩码
        try:
            loaded = np.load(full_output_path)
            print(f"验证：加载的陆地掩码形状: {loaded['island'].shape}")
            print(f"验证：陆地像素数量: {np.sum(loaded['island'])}")
            del loaded
        except Exception as e:
            print(f"验证陆地掩码时出错: {str(e)}")
        
        return True
    
    def run(self, output_file='land_mask.npz'):
        """
        运行整个陆地掩码生成流程
        """
        if self.generate_land_mask():
            return self.save_land_mask(output_file)
        return False

# 主函数
if __name__ == "__main__":
    # 创建陆地掩码生成器
    mask_generator = LandMaskGenerator()
    
    # 运行生成过程并保存结果
    mask_generator.run()
    
    # 可选：加载并显示陆地掩码的基本信息
    try:
        data = np.load('land_mask.npz')
        print("\n陆地掩码文件信息:")
        print(f"lat形状: {data['lat'].shape}")
        print(f"lon形状: {data['lon'].shape}")
        print(f"island形状: {data['island'].shape}")
        print(f"陆地像素数量: {np.sum(data['island'])}")
    except Exception as e:
        print(f"读取陆地掩码文件时出错: {str(e)}")