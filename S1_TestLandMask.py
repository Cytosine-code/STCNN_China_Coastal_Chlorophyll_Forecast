import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 测试陆地掩码的加载和显示
def test_land_mask():
    print("测试陆地掩码...")
    
    try:
        # 加载陆地掩码
        data = np.load('land_mask.npz')
        lon = data['lon']
        lat = data['lat']
        island = data['island']  # True表示陆地，False表示海洋
        
        print(f"陆地掩码信息:")
        print(f"经度范围: {lon.min()} 到 {lon.max()}")
        print(f"纬度范围: {lat.min()} 到 {lat.max()}")
        print(f"陆地像素数量: {np.sum(island)}")
        print(f"海洋像素数量: {np.sum(~island)}")
        print(f"陆地比例: {np.sum(island)/island.size*100:.2f}%")
        
        # 创建简单的可视化
        plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 添加海岸线作为参考
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # 显示陆地掩码（陆地为绿色，海洋为蓝色）
        cmap = plt.cm.Greens
        cmap.set_bad(color='lightblue', alpha=0.5)
        
        # 转换陆地掩码格式以适应显示
        display_mask = np.ma.masked_where(~island, island)
        
        im = ax.pcolormesh(lon, lat, display_mask, transform=ccrs.PlateCarree(), 
                          cmap=cmap, vmin=0, vmax=1)
        
        # 添加经纬度网格
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # 设置标题
        plt.title('陆地掩码可视化')
        
        # 添加颜色条说明
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_ticks([0.5])
        cbar.set_ticklabels(['陆地'])
        
        # 保存图像
        plt.savefig('visualization_results/land_mask_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("陆地掩码可视化已保存至: visualization_results/land_mask_visualization.png")
        
        # 测试与叶绿素数据的兼容性
        test_compatibility(lon, lat, island)
        
    except Exception as e:
        print(f"测试陆地掩码时出错: {str(e)}")

def test_compatibility(mask_lon, mask_lat, mask):
    """
    测试陆地掩码与叶绿素数据的兼容性
    """
    print("\n测试陆地掩码兼容性...")
    
    # 这里只是一个示例，实际使用时可以加载真实的叶绿素数据进行比较
    print(f"陆地掩码形状: {mask.shape}")
    print(f"经度点数: {len(mask_lon)}")
    print(f"纬度点数: {len(mask_lat)}")
    
    # 假设一个简单的使用场景
    print("\n陆地掩码使用示例:")
    print("# 在处理叶绿素数据时，可以这样使用陆地掩码:")
    print("# 1. 加载陆地掩码")
    print("# data = np.load('land_mask.npz')")
    print("# lon = data['lon']")
    print("# lat = data['lat']")
    print("# island_mask = data['island']")
    print("# ")
    print("# 2. 为叶绿素数据添加陆地特征")
    print("# chl_data_with_land = np.copy(chl_data)")
    print("# # 可以设置陆地值为特定值，或者在特征中使用island_mask作为额外特征")
    print("# ")
    print("# 3. 在建模时，可以使用island_mask作为输入特征之一")
    print("# features = np.concatenate([chl_features, island_mask[..., np.newaxis]], axis=-1)")

if __name__ == "__main__":
    test_land_mask()