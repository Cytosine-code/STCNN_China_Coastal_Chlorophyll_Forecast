import xarray as xr
import numpy as np
import os

# 调试脚本：检查原始数据和处理后的数据
def check_data_comparison():
    # 定义中国近海范围（可以根据需要调整）
    coastal_lon_range = slice(100, 140)  # 中国近海经度范围
    coastal_lat_range = slice(50, 0)    # 中国近海纬度范围（注意可能是递减的）
    # 原始数据和处理后数据的路径
    chlProcessed = r"data/Chl/chl_20250401.nc"
    oceanCurProcessed = r"data/OC/cur_20250401.nc"
    saltProcessed = r"data/salt/salt_20250401.nc"
    
    # 假设原始数据路径
    chlRawDir = r"data_raw/chl"
    curRawDir = r"data_raw/cur/04-01  04-11"
    saltRawDir = r"data_raw/salt"
    
    # 查找青岛附近海域范围（120~121°E，35~36°N）
    target_lon_range = slice(120, 121)
    target_lat_range = slice(36, 35)  # 注意原始数据纬度可能是递减的，所以用36到35
    
    # 找到原始数据文件
    def find_first_nc_file(directory):
        if not os.path.exists(directory):
            print(f"警告：目录不存在: {directory}")
            return None
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.nc'):
                    return os.path.join(root, file)
        return None
    
    chlRaw = find_first_nc_file(chlRawDir)
    curRaw = find_first_nc_file(curRawDir)
    saltRaw = find_first_nc_file(saltRawDir)
    
    # 通用函数：打印数据维度和变量
    def print_data_info(ds, data_name):
        print(f"\n{data_name}数据集维度:")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")
        print(f"\n{data_name}数据集变量:")
        for var in ds.data_vars:
            print(f"  {var}")
    
    # 计算指定区域内nan值比例的函数
    def calculate_nan_ratio(ds, lon_range, lat_range, var_name, data_type="数据"):
        try:
            # 获取经纬度坐标
            lon_vals = ds.lon.values if 'lon' in ds.coords else ds.longitude.values
            lat_vals = ds.lat.values if 'lat' in ds.coords else ds.latitude.values
            
            # 找到区域索引
            lon_mask = (lon_vals >= lon_range.start) & (lon_vals <= lon_range.stop)
            lat_mask = (lat_vals >= min(lat_range.start, lat_range.stop)) & \
                      (lat_vals <= max(lat_range.start, lat_range.stop))
            lon_indices = np.where(lon_mask)[0]
            lat_indices = np.where(lat_mask)[0]
            
            if len(lon_indices) == 0 or len(lat_indices) == 0:
                print(f"在{data_type}中未找到指定区域")
                return None, None
            
            # 提取区域数据
            if var_name in ds:
                var_data = ds[var_name].isel(lat=lat_indices, lon=lon_indices)
                # 如果有时间维度，取第一个时间点
                if 'time' in var_data.dims:
                    var_data = var_data.isel(time=0)
                
                # 计算nan值比例
                data_array = var_data.values
                total_cells = data_array.size
                nan_cells = np.isnan(data_array).sum()
                nan_ratio = (nan_cells / total_cells) * 100
                
                print(f"\n{data_type}{var_name}在近海范围内的统计:")
                print(f"  区域大小: {len(lat_indices)} x {len(lon_indices)} = {total_cells} 个单元格")
                print(f"  NaN值单元格数: {nan_cells}")
                print(f"  NaN值比例: {nan_ratio:.2f}%")
                print(f"  有效数据比例: {100 - nan_ratio:.2f}%")
                
                return nan_ratio, total_cells
            else:
                print(f"变量 {var_name} 不存在于{data_type}中")
                return None, None
        except Exception as e:
            print(f"计算{data_type}nan值比例时出错: {e}")
            return None, None
    
    # 通用函数：在青岛附近海域抽样
    def sample_region(ds, lon_range, lat_range, var_names, data_type="处理后"):
        try:
            # 找到特定坐标范围的索引
            # 处理经度索引
            lon_vals = ds.lon.values if 'lon' in ds.coords else ds.longitude.values
            lon_mask = (lon_vals >= lon_range.start) & (lon_vals <= lon_range.stop)
            lon_indices = np.where(lon_mask)[0]
            
            # 处理纬度索引
            lat_vals = ds.lat.values if 'lat' in ds.coords else ds.latitude.values
            lat_mask = (lat_vals >= min(lat_range.start, lat_range.stop)) & \
                       (lat_vals <= max(lat_range.start, lat_range.stop))
            lat_indices = np.where(lat_mask)[0]
            
            if len(lon_indices) == 0 or len(lat_indices) == 0:
                # 找到最接近中心点的索引
                center_lon = 120.5  # 目标区域中心点
                center_lat = 35.5   # 目标区域中心点
                lon_idx = np.abs(lon_vals - center_lon).argmin()
                lat_idx = np.abs(lat_vals - center_lat).argmin()
                
                # 打印找到的索引位置
                print(f"在{data_type}数据中未找到完整区域，使用最接近中心点的单个点:")
                print(f"  中心点附近的索引: lat_idx={lat_idx}, lon_idx={lon_idx}")
                print(f"  实际经纬度: {lat_vals[lat_idx]}, {lon_vals[lon_idx]}")
                
                # 获取该点的值
                for var_name in var_names:
                    if var_name in ds:
                        try:
                            # 如果有时间维度，取第一个时间点
                            if 'time' in ds.dims:
                                var_value = ds[var_name].isel(lat=lat_idx, lon=lon_idx, time=0).values
                            else:
                                var_value = ds[var_name].isel(lat=lat_idx, lon=lon_idx).values
                            print(f"  {data_type}{var_name}值: {var_value}")
                        except Exception as e:
                            print(f"  获取{var_name}值时出错: {e}")
            else:
                # 如果成功找到区域，获取区域数据
                region_data = ds.isel(lat=lat_indices, lon=lon_indices)
                
                # 打印区域信息
                print(f"在{data_type}数据中找到的区域:")
                print(f"  纬度范围: {min(lat_vals[lat_indices])} 到 {max(lat_vals[lat_indices])}")
                print(f"  经度范围: {min(lon_vals[lon_indices])} 到 {max(lon_vals[lon_indices])}")
                
                # 获取区域中心的值
                center_lat_idx = len(lat_indices) // 2
                center_lon_idx = len(lon_indices) // 2
                
                for var_name in var_names:
                    if var_name in region_data:
                        try:
                            # 如果有时间维度，取第一个时间点
                            if 'time' in region_data.dims:
                                var_value = region_data[var_name].isel(lat=center_lat_idx, lon=center_lon_idx, time=0).values
                            else:
                                var_value = region_data[var_name].isel(lat=center_lat_idx, lon=center_lon_idx).values
                            print(f"  {data_type}{var_name}值（区域中心）: {var_value}")
                        except Exception as e:
                            print(f"  获取{var_name}值时出错: {e}")
        except Exception as e:
            print(f"  区域抽样出错: {e}")
    
    # 1. 检查叶绿素数据
    print("=== 检查叶绿素数据 ===")
    # 处理后数据
    try:
        ds_chl = xr.open_dataset(chlProcessed)
        print_data_info(ds_chl, "处理后叶绿素")
        # 抽样检查青岛附近海域
        chl_vars = ['chlor_a', 'chlorophyll'] if 'chlor_a' in ds_chl or 'chlorophyll' in ds_chl else list(ds_chl.data_vars)
        print("\n在青岛附近海域(120~121°E，35~36°N)抽样叶绿素数据:")
        sample_region(ds_chl, target_lon_range, target_lat_range, chl_vars, "处理后")
        
        # 计算近海范围内处理后数据的nan值比例
        print("\n=== 计算近海范围内处理后叶绿素数据的NaN值比例 ===")
        print(f"近海范围定义: {coastal_lon_range.start}~{coastal_lon_range.stop}°E, ")
        print(f"             {min(coastal_lat_range.start, coastal_lat_range.stop)}~{max(coastal_lat_range.start, coastal_lat_range.stop)}°N")
        processed_nan_ratio, _ = calculate_nan_ratio(ds_chl, coastal_lon_range, coastal_lat_range, 
                                                    chl_vars[0], "处理后叶绿素")
        
        # 原始数据
        if chlRaw:
            print(f"\n检查原始叶绿素数据: {os.path.basename(chlRaw)}")
            try:
                ds_chl_raw = xr.open_dataset(chlRaw)
                print_data_info(ds_chl_raw, "原始叶绿素")
                # 尝试在原始数据中找到相似的区域
                print("\n在青岛附近海域(120~121°E，35~36°N)抽样原始叶绿素数据:")
                # 可能需要调整坐标名称
                if 'lat' not in ds_chl_raw.coords and 'latitude' in ds_chl_raw.coords:
                    ds_chl_raw = ds_chl_raw.rename({'latitude': 'lat', 'longitude': 'lon'})
                sample_region(ds_chl_raw, target_lon_range, target_lat_range, chl_vars, "原始")
                ds_chl_raw.close()
                # 计算近海范围内原始数据的nan值比例
                print("\n=== 计算近海范围内原始叶绿素数据的NaN值比例 ===")
                original_nan_ratio, _ = calculate_nan_ratio(ds_chl_raw, coastal_lon_range, coastal_lat_range, 
                                                        chl_vars[0], "原始叶绿素")
                
                # 比较原始数据和处理后数据的nan值比例
                if processed_nan_ratio is not None and original_nan_ratio is not None:
                    improvement = original_nan_ratio - processed_nan_ratio
                    print(f"\n=== 原始数据与处理后数据比较 ===")
                    print(f"原始数据NaN值比例: {original_nan_ratio:.2f}%")
                    print(f"处理后数据NaN值比例: {processed_nan_ratio:.2f}%")
                    print(f"NaN值减少比例: {improvement:.2f}%")
                    
                ds_chl_raw.close()
            except Exception as e:
                print(f"读取原始叶绿素数据时出错: {e}")
        ds_chl.close()
    except Exception as e:
        print(f"读取处理后叶绿素数据时出错: {e}")
    
    # 2. 检查海流数据
    print("\n\n=== 检查海流数据 ===")
    # 处理后数据
    try:
        ds_cur = xr.open_dataset(oceanCurProcessed)
        print_data_info(ds_cur, "处理后海流")
        # 抽样检查青岛附近海域
        cur_vars = ['uo', 'vo'] if 'uo' in ds_cur and 'vo' in ds_cur else list(ds_cur.data_vars)
        print("\n在青岛附近海域(120~121°E，35~36°N)抽样海流数据:")
        sample_region(ds_cur, target_lon_range, target_lat_range, cur_vars, "处理后")
        
        # 原始数据
        if curRaw:
            print(f"\n检查原始海流数据: {os.path.basename(curRaw)}")
            try:
                ds_cur_raw = xr.open_dataset(curRaw)
                print_data_info(ds_cur_raw, "原始海流")
                # 尝试在原始数据中找到相似的区域
                print("\n在青岛附近海域(120~121°E，35~36°N)抽样原始海流数据:")
                # 可能需要调整坐标名称
                if 'lat' not in ds_cur_raw.coords and 'latitude' in ds_cur_raw.coords:
                    ds_cur_raw = ds_cur_raw.rename({'latitude': 'lat', 'longitude': 'lon'})
                # 对于原始数据，可能需要选择第一个时间点和第一个深度层
                if 'time' in ds_cur_raw.dims:
                    ds_cur_raw = ds_cur_raw.isel(time=0)
                if 'depth' in ds_cur_raw.dims:
                    ds_cur_raw = ds_cur_raw.isel(depth=0)
                sample_region(ds_cur_raw, target_lon_range, target_lat_range, cur_vars, "原始")
                ds_cur_raw.close()
            except Exception as e:
                print(f"读取原始海流数据时出错: {e}")
        else:
            print("未找到原始海流数据文件")
        
        ds_cur.close()
    except Exception as e:
        print(f"读取处理后海流数据时出错: {e}")
    
    # 3. 检查营养盐数据
    print("\n\n=== 检查营养盐数据 ===")
    # 处理后数据
    try:
        ds_salt = xr.open_dataset(saltProcessed)
        print_data_info(ds_salt, "处理后营养盐")
        # 抽样检查青岛附近海域，特别关注fe值
        salt_vars = ['fe'] + [var for var in list(ds_salt.data_vars) if var not in ['fe', 'lat', 'lon', 'time']]
        print("\n在青岛附近海域(120~121°E，35~36°N)抽样营养盐数据:")
        sample_region(ds_salt, target_lon_range, target_lat_range, salt_vars, "处理后")
        
        # 原始数据
        if saltRaw:
            print(f"\n检查原始营养盐数据: {os.path.basename(saltRaw)}")
            try:
                ds_salt_raw = xr.open_dataset(saltRaw)
                print_data_info(ds_salt_raw, "原始营养盐")
                # 尝试在原始数据中找到相似的区域
                print("\n在青岛附近海域(120~121°E，35~36°N)抽样原始营养盐数据:")
                # 可能需要调整坐标名称
                if 'lat' not in ds_salt_raw.coords and 'latitude' in ds_salt_raw.coords:
                    ds_salt_raw = ds_salt_raw.rename({'latitude': 'lat', 'longitude': 'lon'})
                # 对于原始数据，可能需要选择第一个时间点和第一个深度层
                if 'time' in ds_salt_raw.dims:
                    ds_salt_raw = ds_salt_raw.isel(time=0)
                if 'depth' in ds_salt_raw.dims:
                    ds_salt_raw = ds_salt_raw.isel(depth=0)
                sample_region(ds_salt_raw, target_lon_range, target_lat_range, salt_vars, "原始")
                ds_salt_raw.close()
            except Exception as e:
                print(f"读取原始营养盐数据时出错: {e}")
        else:
            print("未找到原始营养盐数据文件")
        
        ds_salt.close()
    except Exception as e:
        print(f"读取处理后营养盐数据时出错: {e}")

if __name__ == "__main__":
    check_data_comparison()