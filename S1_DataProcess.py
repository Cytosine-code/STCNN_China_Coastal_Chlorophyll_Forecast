import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import os
import glob
from scipy.interpolate import griddata, RegularGridInterpolator

class Chl:
    """
    处理叶绿素数据的类
    数据特征：全球范围，每天一个文件，需要筛选出中国近海区域(100~180经度，0~60纬度)
    """
    def showStruct(self):
        ncf = nc.Dataset(rf"data_raw/chl/ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-20250401-fv6.0.nc")
        print("叶绿素数据元：")
        print(ncf.variables["chlor_a"])
        print("time数据元:")
        print(ncf.variables["time"])
        print("lat数据元:")
        print(ncf.variables["lat"])
        print("lon数据元:")
        print(ncf.variables["lon"])
        print("叶绿素示例值:")
        print(ncf.variables["chlor_a"][:].data[0, 0, 0])
        ncf.close()
    
    def dataProcess(self, skip_processing):
        """
        处理叶绿素数据，筛选中国近海区域，并保存为基准网格
        参数：skip_processing - 如果为True，则跳过数据处理，只返回基准网格（假设数据已处理）
        返回：基准网格信息（lat和lon数组）
        """
        # 保存一个基准网格信息，供其他数据重采样使用
        baseline_grid = None
        
        # 定义目标网格尺寸（600x480）
        target_lat_size = 600
        target_lon_size = 480
        
        # 1. 检查是否跳过处理
        if skip_processing:
            print("跳过叶绿素数据处理，使用示例网格...")
            # 创建纬度数组（从50°N到0°N，调整为目标尺寸）
            lat_values = np.linspace(50, 0, target_lat_size)
            # 创建经度数组（从100°E到140°E，调整为目标尺寸）
            lon_values = np.linspace(100, 140, target_lon_size)
            
            baseline_grid = {
                'lat': lat_values,
                'lon': lon_values
            }
            print(f"已创建示例基准网格：lat维度={len(baseline_grid['lat'])}, lon维度={len(baseline_grid['lon'])}")
            return baseline_grid
        
        # 确保输出目录存在
        output_dir = "data/Chl"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 2. 加载陆地掩码
        land_mask = None
        try:
            mask_data = np.load('dataset/land_mask.npz')
            land_mask = mask_data['island']  # land_mask是一个布尔数组，True表示陆地
            print(f"成功加载陆地掩码，形状: {land_mask.shape}")
        except Exception as e:
            print(f"加载陆地掩码失败: {e}，将只对有效数据进行插值")
        
        # 定义中国近海范围
        lat_range = slice(50, 0)  # 50°N到0°N，由于数据是递减的，所以用50到0
        lon_range = slice(100, 140)  # 中国近海经度范围（100-140°E）
        
        # 处理4月和5月的数据
        for m in range(4,6):
            for d in range(1,32):
                # 跳过4月31日（不存在）
                if(m == 4 and d == 31):
                    continue
                
                # 构建文件名
                date_str = f"2025{m:02d}{d:02d}"
                file_path = rf"data_raw/chl/ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-{date_str}-fv6.0.nc"
                output_path = rf"{output_dir}/chl_{date_str}.nc"
                
                try:
                    # 使用xarray读取数据
                    ds = xr.open_dataset(file_path)
                    
                    # 筛选中国近海范围
                    ds_region = ds.sel(lat=lat_range, lon=lon_range)
                    
                    # 处理填充值（将9.96921e+36替换为NaN）
                    ds_region = ds_region.where(ds_region['chlor_a'] != 9.96921e+36, np.nan)
                    
                    # 创建目标网格
                    if baseline_grid is None:
                        # 计算目标网格的经纬度范围（与原始数据相同的范围）
                        try:
                            min_lat_val = float(ds_region.lat.min().values)
                            max_lat_val = float(ds_region.lat.max().values)
                            min_lon_val = float(ds_region.lon.min().values)
                            max_lon_val = float(ds_region.lon.max().values)
                            
                            # 创建新的网格坐标
                            target_lat = np.linspace(max_lat_val, min_lat_val, target_lat_size)
                            target_lon = np.linspace(min_lon_val, max_lon_val, target_lon_size)
                            
                            baseline_grid = {
                                'lat': target_lat,
                                'lon': target_lon
                            }
                            print(f"已创建目标基准网格：lat维度={len(baseline_grid['lat'])}, lon维度={len(baseline_grid['lon'])}")
                        except Exception as e:
                            print(f"创建目标网格时出错: {e}")
                            # 使用固定范围的备用方案
                            target_lat = np.linspace(50, 0, target_lat_size)
                            target_lon = np.linspace(100, 140, target_lon_size)
                            
                            baseline_grid = {
                                'lat': target_lat,
                                'lon': target_lon
                            }
                    
                    # 先进行数据重采样到目标网格（600x480）
                    try:
                        # 确保目标网格的经纬度数组是正确的类型和形状
                        lat_coord = baseline_grid['lat'].astype(float)
                        lon_coord = baseline_grid['lon'].astype(float)
                        
                        # 进行数据重采样
                        ds_resampled = ds_region.interp(
                            lat=lat_coord,
                            lon=lon_coord,
                            method='linear'
                        )
                        print(f"数据重采样完成，日期：{date_str}")
                    except Exception as e:
                        print(f"数据重采样时出错: {e}")
                        # 创建空的数据集作为备用
                        ds_resampled = xr.Dataset(
                            data_vars={
                                'chlor_a': (['lat', 'lon'], np.full((target_lat_size, target_lon_size), np.nan))
                            },
                            coords={
                                'lat': baseline_grid['lat'],
                                'lon': baseline_grid['lon']
                            }
                        )
                    
                    # 获取重采样后的chlor_a数据数组
                    chl_data = ds_resampled['chlor_a'].values
                    
                    # 记录原始NaN值数量
                    original_nan_count = np.isnan(chl_data).sum()
                    
                    # 检查陆地掩码是否可用且网格匹配
                    mask_is_valid = False
                    if land_mask is not None:
                        # 检查陆地掩码形状是否与目标网格匹配
                        if land_mask.shape == (target_lat_size, target_lon_size):
                            mask_is_valid = True
                            print(f"陆地掩码网格匹配，使用陆地掩码进行插值")
                        else:
                            print(f"陆地掩码形状不匹配（掩码: {land_mask.shape}, 目标: {(target_lat_size, target_lon_size)}），不使用掩码进行插值")
                    
                    # 3. 处理三维数据
                    if len(chl_data.shape) == 3:
                        # 获取目标网格坐标
                        lon_grid, lat_grid = np.meshgrid(ds_resampled.lon.values, ds_resampled.lat.values)
                        
                        # 对每个时间点进行处理
                        for t in range(chl_data.shape[0]):
                            chl_slice = chl_data[t, :, :]
                            
                            if mask_is_valid:
                                # 有有效的陆地掩码：只对海洋区域（非陆地）的NaN值进行插值
                                print(f"使用有效陆地掩码，只对海洋区域NaN值进行插值")
                                
                                # 创建海洋区域的有效数据点掩码（非陆地且非NaN）
                                ocean_valid_mask = ~land_mask & ~np.isnan(chl_slice)
                                ocean_valid_points_count = np.sum(ocean_valid_mask)
                                
                                if ocean_valid_points_count > 0:
                                    # 获取有效数据点的坐标和值
                                    valid_points = np.column_stack((lon_grid[ocean_valid_mask], lat_grid[ocean_valid_mask]))
                                    valid_values = chl_slice[ocean_valid_mask]
                                    
                                    # 获取需要插值的点（海洋区域但为NaN）
                                    ocean_nan_mask = ~land_mask & np.isnan(chl_slice)
                                    ocean_nan_points_count = np.sum(ocean_nan_mask)
                                    
                                    if ocean_nan_points_count > 0:
                                        interpolate_points = np.column_stack((lon_grid[ocean_nan_mask], lat_grid[ocean_nan_mask]))
                                        
                                        try:
                                            # 进行双线性插值
                                            interpolated_values = griddata(valid_points, valid_values, interpolate_points, method='linear')
                                            
                                            # 将插值结果填充到数据中
                                            chl_slice[ocean_nan_mask] = interpolated_values
                                            
                                            # 确保陆地区域仍然是NaN
                                            chl_slice[land_mask] = np.nan
                                            
                                        except Exception as e:
                                            print(f"执行双线性插值时出错: {e}")
                            else:
                                # 没有有效陆地掩码：直接对所有有效数据点进行插值
                                # 但确保不进行全局插值（只对有有效数据的区域进行局部插值）
                                print(f"无有效陆地掩码，对有效数据点进行局部插值")
                                
                                # 获取有效数据点
                                valid_mask = ~np.isnan(chl_slice)
                                valid_points_count = np.sum(valid_mask)
                                
                                if valid_points_count > 0:
                                    # 获取有效数据点的坐标和值
                                    valid_points = np.column_stack((lon_grid[valid_mask], lat_grid[valid_mask]))
                                    valid_values = chl_slice[valid_mask]
                                    
                                    # 获取需要插值的点（NaN区域）
                                    nan_mask = np.isnan(chl_slice)
                                    nan_points_count = np.sum(nan_mask)
                                    
                                    if nan_points_count > 0:
                                        interpolate_points = np.column_stack((lon_grid[nan_mask], lat_grid[nan_mask]))
                                        
                                        try:
                                            # 进行双线性插值
                                            interpolated_values = griddata(valid_points, valid_values, interpolate_points, method='linear')
                                            
                                            # 将插值结果填充到数据中
                                            chl_slice[nan_mask] = interpolated_values
                                            
                                        except Exception as e:
                                            print(f"执行双线性插值时出错: {e}")
                            
                            # 更新数据
                            chl_data[t, :, :] = chl_slice
                        
                        # 更新数据集
                        ds_resampled['chlor_a'].values = chl_data
                        
                        # 计算插值后的NaN值数量
                        after_interpolate_nan_count = np.isnan(chl_data).sum()
                        nan_reduction = original_nan_count - after_interpolate_nan_count
                        print(f"插值前NaN值数量: {original_nan_count}")
                        print(f"插值后NaN值数量: {after_interpolate_nan_count}")
                        print(f"NaN值减少数量: {nan_reduction}")
                    
                    # 保存重采样后的数据
                    ds_resampled.to_netcdf(output_path)
                    
                    # 关闭数据集
                    ds.close()
                    ds_resampled.close()
                    
                    print(f"已处理叶绿素数据: {date_str}")
                except Exception as e:
                    print(f"处理叶绿素数据{date_str}时出错: {e}")
        
        print("叶绿素数据预处理完成")
        return baseline_grid

class Cur:
    """
    处理海流数据的类
    数据特征：已为中国近海范围，每个文件包含多天数据，需要按天提取并重采样到叶绿素网格
    """
    def showStruct(self):
        ncf=nc.Dataset(rf"data_raw/cur/04-01  04-11/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_1756889683665.nc")
        print(ncf.variables.keys())
        print("time数据元：")
        print(ncf.variables["time"])
        print("depth数据元：")
        print(ncf.variables["depth"])
        print("经度数据元：")
        print(ncf.variables["longitude"])
        print("纬度数据元：")
        print(ncf.variables["latitude"])
        print("uo数据元：")
        print(ncf.variables["uo"])

        ncf.close()
    
    def dataProcess(self, baseline_grid=None):
        """
        处理海流数据，按天提取并重采样到叶绿素网格
        参数：baseline_grid - 叶绿素的基准网格信息
        """
        # 确保输出目录存在
        output_dir = "data/OC"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 加载陆地掩码
        land_mask = None
        try:
            mask_data = np.load('dataset/land_mask.npz')
            land_mask = mask_data['island']  # land_mask是一个布尔数组，True表示陆地
            print(f"成功加载陆地掩码，形状: {land_mask.shape}")
        except Exception as e:
            print(f"加载陆地掩码失败: {e}，将只对有效数据进行插值")
        
        # 获取所有海流数据文件
        cur_dirs = glob.glob(r"data_raw/cur/*")
        
        for dir_path in cur_dirs:
            try:
                # 获取目录下的所有nc文件
                nc_files = glob.glob(os.path.join(dir_path, "*.nc"))
                
                for file in nc_files:
                    try:
                        # 使用xarray读取数据
                        ds = xr.open_dataset(file)
                        
                        # 遍历文件中的每一天数据
                        for t in range(len(ds.time)):
                            try:
                                # 提取单日数据，只保留第一层水面数据
                                ds_day = ds.isel(time=t, depth=0)
                                
                                # 获取日期字符串（从文件名中提取或从time变量计算）
                                time_value = pd.to_datetime(ds_day.time.values)
                                date_str = time_value.strftime("%Y%m%d")
                                output_path = rf"{output_dir}/cur_{date_str}.nc"
                                
                                # 检查是否有基准网格，如果有则先进行重采样
                                if baseline_grid is not None:
                                    try:
                                        print(f"准备重采样海流数据 - 目标网格尺寸: {len(baseline_grid['lat'])}x{len(baseline_grid['lon'])}")
                                        # 确保目标网格的经纬度数组是正确的类型和形状
                                        lat_coord = baseline_grid['lat'].astype(float)
                                        lon_coord = baseline_grid['lon'].astype(float)
                                        
                                        # 使用双线性插值重采样到叶绿素数据的网格
                                        ds_resampled = ds_day.interp(
                                            latitude=lat_coord,
                                            longitude=lon_coord,
                                            method='linear'
                                        )
                                        print(f"海流数据重采样完成，日期：{date_str}")
                                        
                                        # 重命名坐标以匹配叶绿素数据
                                        ds_resampled = ds_resampled.rename({'latitude': 'lat', 'longitude': 'lon'})
                                        
                                    except Exception as e:
                                        print(f"重采样海流数据时出错: {e}")
                                        # 创建空的数据集作为备用
                                        ds_resampled = xr.Dataset(
                                            data_vars={
                                                'uo': (['lat', 'lon'], np.full((len(baseline_grid['lat']), len(baseline_grid['lon'])), np.nan)),
                                                'vo': (['lat', 'lon'], np.full((len(baseline_grid['lat']), len(baseline_grid['lon'])), np.nan))
                                            },
                                            coords={
                                                'lat': baseline_grid['lat'],
                                                'lon': baseline_grid['lon']
                                            }
                                        )
                                else:
                                    # 没有基准网格时直接使用原始数据
                                    ds_resampled = ds_day.rename({'latitude': 'lat', 'longitude': 'lon'})
                                
                                # 获取重采样后的海流数据
                                uo_data = ds_resampled['uo'].values
                                vo_data = ds_resampled['vo'].values
                                
                                # 获取目标网格坐标
                                lon_grid, lat_grid = np.meshgrid(ds_resampled.lon.values, ds_resampled.lat.values)
                                
                                # 检查陆地掩码是否有效（与目标网格形状匹配）
                                mask_is_valid = False
                                if land_mask is not None and land_mask.shape == uo_data.shape:
                                    mask_is_valid = True
                                    print(f"陆地掩码网格匹配，使用陆地掩码进行插值")
                                else:
                                    print(f"陆地掩码形状不匹配或不存在，不使用掩码进行插值")
                                
                                # 对海流数据进行NaN值插值处理
                                if mask_is_valid:
                                    # 记录原始NaN值数量
                                    original_nan_count_uo = np.isnan(uo_data).sum()
                                    original_nan_count_vo = np.isnan(vo_data).sum()
                                    
                                    # 处理uo数据
                                    # 创建海洋区域的有效数据点掩码（非陆地且非NaN）
                                    ocean_valid_mask_uo = ~land_mask & ~np.isnan(uo_data)
                                    ocean_valid_points_count_uo = np.sum(ocean_valid_mask_uo)
                                    
                                    if ocean_valid_points_count_uo > 0:
                                        # 获取有效数据点的坐标和值
                                        valid_points_uo = np.column_stack((lon_grid[ocean_valid_mask_uo], lat_grid[ocean_valid_mask_uo]))
                                        valid_values_uo = uo_data[ocean_valid_mask_uo]
                                        
                                        # 获取需要插值的点（海洋区域但为NaN）
                                        ocean_nan_mask_uo = ~land_mask & np.isnan(uo_data)
                                        ocean_nan_points_count_uo = np.sum(ocean_nan_mask_uo)
                                        
                                        if ocean_nan_points_count_uo > 0:
                                            interpolate_points_uo = np.column_stack((lon_grid[ocean_nan_mask_uo], lat_grid[ocean_nan_mask_uo]))
                                            
                                            try:
                                                # 进行双线性插值
                                                interpolated_values_uo = griddata(valid_points_uo, valid_values_uo, interpolate_points_uo, method='linear')
                                                
                                                # 将插值结果填充到数据中
                                                uo_data[ocean_nan_mask_uo] = interpolated_values_uo
                                                
                                                # 确保陆地区域仍然是NaN
                                                uo_data[land_mask] = np.nan
                                                
                                            except Exception as e:
                                                print(f"执行uo数据双线性插值时出错: {e}")
                                    
                                    # 处理vo数据
                                    # 创建海洋区域的有效数据点掩码（非陆地且非NaN）
                                    ocean_valid_mask_vo = ~land_mask & ~np.isnan(vo_data)
                                    ocean_valid_points_count_vo = np.sum(ocean_valid_mask_vo)
                                    
                                    if ocean_valid_points_count_vo > 0:
                                        # 获取有效数据点的坐标和值
                                        valid_points_vo = np.column_stack((lon_grid[ocean_valid_mask_vo], lat_grid[ocean_valid_mask_vo]))
                                        valid_values_vo = vo_data[ocean_valid_mask_vo]
                                        
                                        # 获取需要插值的点（海洋区域但为NaN）
                                        ocean_nan_mask_vo = ~land_mask & np.isnan(vo_data)
                                        ocean_nan_points_count_vo = np.sum(ocean_nan_mask_vo)
                                        
                                        if ocean_nan_points_count_vo > 0:
                                            interpolate_points_vo = np.column_stack((lon_grid[ocean_nan_mask_vo], lat_grid[ocean_nan_mask_vo]))
                                            
                                            try:
                                                # 进行双线性插值
                                                interpolated_values_vo = griddata(valid_points_vo, valid_values_vo, interpolate_points_vo, method='linear')
                                                
                                                # 将插值结果填充到数据中
                                                vo_data[ocean_nan_mask_vo] = interpolated_values_vo
                                                
                                                # 确保陆地区域仍然是NaN
                                                vo_data[land_mask] = np.nan
                                                
                                            except Exception as e:
                                                print(f"执行vo数据双线性插值时出错: {e}")
                                    
                                    # 更新数据集
                                    ds_resampled['uo'].values = uo_data
                                    ds_resampled['vo'].values = vo_data
                                    
                                    # 计算插值后的NaN值数量
                                    after_interpolate_nan_count_uo = np.isnan(uo_data).sum()
                                    after_interpolate_nan_count_vo = np.isnan(vo_data).sum()
                                    
                                    print(f"uo数据：插值前NaN值数量: {original_nan_count_uo}，插值后NaN值数量: {after_interpolate_nan_count_uo}")
                                    print(f"vo数据：插值前NaN值数量: {original_nan_count_vo}，插值后NaN值数量: {after_interpolate_nan_count_vo}")
                                
                                # 保存处理后的数据
                                ds_resampled.to_netcdf(output_path)
                                print(f"已处理海流数据: {date_str}")
                            except Exception as e:
                                print(f"处理海流数据时间步{t}时出错: {e}")
                        
                        ds.close()
                    except Exception as e:
                        print(f"处理海流文件{file}时出错: {e}")
            except Exception as e:
                print(f"访问目录{dir_path}时出错: {e}")
        
        print("海流数据预处理完成")
class Salt:
    """
    处理营养盐数据的类
    数据特征：已为中国近海范围，每个文件包含多天数据，需要按天提取并重采样到叶绿素网格
    """
    def showStruct(self):
        ncf=nc.Dataset(rf"data_raw/salt/04-01  04-30/cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m_1756886158123.nc")
        print(ncf.variables.keys())
        print("time数据元：")
        print(ncf.variables["time"])
        print("depth数据元：")
        print(ncf.variables["depth"])
        print("latitude数据元：")
        print(ncf.variables["latitude"])
        print("longitude数据元：")
        print(ncf.variables["longitude"])
        print("fe数据元：")
        print(ncf.variables["fe"])
        print("no3数据元：")
        print(ncf.variables["no3"])
    
    def dataProcess(self, baseline_grid):
        """
        处理营养盐数据，按天提取并重采样到叶绿素网格
        参数：baseline_grid - 叶绿素的基准网格信息
        """
        # 确保输出目录存在
        output_dir = "data/salt"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 加载陆地掩码
        land_mask = None
        try:
            mask_data = np.load('dataset/land_mask.npz')
            land_mask = mask_data['island']  # land_mask是一个布尔数组，True表示陆地
            print(f"成功加载陆地掩码，形状: {land_mask.shape}")
        except Exception as e:
            print(f"加载陆地掩码失败: {e}，将只对有效数据进行插值")
        
        # 获取所有营养盐数据文件
        salt_dirs = glob.glob(r"data_raw/salt/*")
        
        for dir_path in salt_dirs:
            try:
                # 获取目录下的所有nc文件
                nc_files = glob.glob(os.path.join(dir_path, "*.nc"))
                
                for file in nc_files:
                    try:
                        # 使用xarray读取数据
                        ds = xr.open_dataset(file)
                        
                        # 遍历文件中的每一天数据
                        for t in range(len(ds.time)):
                            try:
                                # 提取单日数据，只保留第一层水面数据
                                ds_day = ds.isel(time=t, depth=0)
                                
                                # 获取日期字符串
                                time_value = pd.to_datetime(ds_day.time.values)
                                date_str = time_value.strftime("%Y%m%d")
                                output_path = rf"{output_dir}/salt_{date_str}.nc"
                                
                                # 检查是否有基准网格，如果有则先进行重采样
                                if baseline_grid is not None:
                                    try:
                                        # 检查单日数据是否包含有效值
                                        if ds_day.notnull().any():
                                            print(f"准备重采样营养盐数据 - 目标网格尺寸: {len(baseline_grid['lat'])}x{len(baseline_grid['lon'])}")
                                            # 确保目标网格的经纬度数组是正确的类型和形状
                                            lat_coord = baseline_grid['lat'].astype(float)
                                            lon_coord = baseline_grid['lon'].astype(float)
                                             
                                            # 使用双线性插值重采样到叶绿素数据的网格
                                            ds_resampled = ds_day.interp(
                                                latitude=lat_coord,
                                                longitude=lon_coord,
                                                method='linear'
                                            )
                                            print(f"营养盐数据重采样完成，日期：{date_str}")
                                             
                                            # 重命名坐标以匹配叶绿素数据
                                            ds_resampled = ds_resampled.rename({'latitude': 'lat', 'longitude': 'lon'})
                                            
                                        else:
                                            # 创建空的数据集作为备用
                                            data_vars = {}
                                            for var in ds_day.data_vars:
                                                if var not in ['lat', 'lon', 'time', 'depth', 'latitude', 'longitude']:
                                                    data_vars[var] = (['lat', 'lon'], np.full((len(baseline_grid['lat']), len(baseline_grid['lon'])), np.nan))
                                            
                                            ds_resampled = xr.Dataset(
                                                data_vars=data_vars,
                                                coords={
                                                    'lat': baseline_grid['lat'],
                                                    'lon': baseline_grid['lon']
                                                }
                                            )
                                            print(f"营养盐数据全为NaN，创建空数据集")
                                             
                                    except Exception as e:
                                        print(f"重采样营养盐数据时出错: {e}")
                                        # 创建空的数据集作为备用
                                        data_vars = {}
                                        for var in ds_day.data_vars:
                                            if var not in ['lat', 'lon', 'time', 'depth', 'latitude', 'longitude']:
                                                data_vars[var] = (['lat', 'lon'], np.full((len(baseline_grid['lat']), len(baseline_grid['lon'])), np.nan))
                                        
                                        ds_resampled = xr.Dataset(
                                            data_vars=data_vars,
                                            coords={
                                                'lat': baseline_grid['lat'],
                                                'lon': baseline_grid['lon']
                                            }
                                        )
                                else:
                                    # 没有基准网格时直接使用原始数据
                                    ds_resampled = ds_day.rename({'latitude': 'lat', 'longitude': 'lon'})
                                
                                # 获取目标网格坐标
                                lon_grid, lat_grid = np.meshgrid(ds_resampled.lon.values, ds_resampled.lat.values)
                                
                                # 获取营养盐变量（排除坐标变量）
                                salt_vars = [var for var in ds_resampled.data_vars if var not in ['lat', 'lon', 'time', 'depth']]
                                
                                # 检查陆地掩码是否有效（与目标网格形状匹配）
                                mask_is_valid = False
                                if land_mask is not None and land_mask.shape == (len(ds_resampled.lat), len(ds_resampled.lon)):
                                    mask_is_valid = True
                                    print(f"陆地掩码网格匹配，使用陆地掩码进行插值")
                                else:
                                    print(f"陆地掩码形状不匹配或不存在，不使用掩码进行插值")
                                
                                # 对每个营养盐变量进行NaN值插值处理（只对海洋区域）
                                if mask_is_valid:
                                    for var in salt_vars:
                                        if var in ds_resampled:
                                            var_data = ds_resampled[var].values
                                             
                                            # 记录原始NaN值数量
                                            original_nan_count = np.isnan(var_data).sum()
                                             
                                            # 对海洋区域（非陆地）的NaN值进行插值
                                            print(f"对营养盐变量 {var} 进行海洋区域NaN值插值")
                                             
                                            # 创建海洋区域的有效数据点掩码（非陆地且非NaN）
                                            ocean_valid_mask = ~land_mask & ~np.isnan(var_data)
                                            ocean_valid_points_count = np.sum(ocean_valid_mask)
                                             
                                            if ocean_valid_points_count > 0:
                                                # 获取有效数据点的坐标和值
                                                valid_points = np.column_stack((lon_grid[ocean_valid_mask], lat_grid[ocean_valid_mask]))
                                                valid_values = var_data[ocean_valid_mask]
                                                 
                                                # 获取需要插值的点（海洋区域但为NaN）
                                                ocean_nan_mask = ~land_mask & np.isnan(var_data)
                                                ocean_nan_points_count = np.sum(ocean_nan_mask)
                                                 
                                                if ocean_nan_points_count > 0:
                                                    interpolate_points = np.column_stack((lon_grid[ocean_nan_mask], lat_grid[ocean_nan_mask]))
                                                     
                                                    try:
                                                        # 进行双线性插值
                                                        interpolated_values = griddata(valid_points, valid_values, interpolate_points, method='linear')
                                                         
                                                        # 将插值结果填充到数据中
                                                        var_data[ocean_nan_mask] = interpolated_values
                                                         
                                                        # 确保陆地区域仍然是NaN
                                                        var_data[land_mask] = np.nan
                                                         
                                                        # 更新数据集
                                                        ds_resampled[var].values = var_data
                                                         
                                                    except Exception as e:
                                                        print(f"执行营养盐变量 {var} 双线性插值时出错: {e}")
                                             
                                            # 计算插值后的NaN值数量
                                            after_interpolate_nan_count = np.isnan(var_data).sum()
                                            print(f"营养盐变量 {var}：插值前NaN值数量: {original_nan_count}，插值后NaN值数量: {after_interpolate_nan_count}")
                                
                                # 保存处理后的数据
                                ds_resampled.to_netcdf(output_path)
                                print(f"已处理营养盐数据: {date_str}")
                            except Exception as e:
                                print(f"处理营养盐数据时间步{t}时出错: {e}")
                        
                        ds.close()
                    except Exception as e:
                        print(f"处理营养盐文件{file}时出错: {e}")
            except Exception as e:
                print(f"访问目录{dir_path}时出错: {e}")
        
        print("营养盐数据预处理完成")

# 添加主函数，用于运行所有预处理
def main():
    print("开始数据预处理...")
    
    # 控制是否跳过叶绿素数据处理
    skip_chl_processing = False  # 设为True可跳过叶绿素处理
    
    # 处理叶绿素数据，获取基准网格
    print("\n处理叶绿素数据：")
    chl = Chl()
    baseline_grid = chl.dataProcess(skip_processing=skip_chl_processing)
    
    # 处理海流数据
    print("\n处理海流数据：")
    cur = Cur()
    cur.dataProcess(baseline_grid)
    
    # 处理营养盐数据
    print("\n处理营养盐数据：")
    salt = Salt()
    salt.dataProcess(baseline_grid)
    
    print("\n所有数据预处理完成！")

# 如果作为主程序运行，则执行main函数
if __name__ == "__main__":
    main()