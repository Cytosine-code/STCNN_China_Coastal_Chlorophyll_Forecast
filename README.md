# STCNN数值预测模型实践

# ——中国近海叶绿素变化预测

## Project Evalution

本项目成功实现了利用时空卷积神经网络(STCNN)预测中国近海叶绿素浓度变化的目标。项目完整涵盖了从数据预处理、特征工程到模型训练、评估和可视化的全流程，技术实现合理，功能完善。

<u>*个人认为是新手学习机器学习&数值预测&模型训练的最佳Demo，最小实现！*</u>

主要成就：
1. 建立了完整的数据处理流水线，解决了原始数据中的缺失值和时空对齐问题
2. 设计并实现了STCNN模型，有效捕捉海洋环境数据的时空特征
3. 开发了自定义损失函数，考虑陆地掩码影响，提高模型预测精度
4. 实现了全面的评估指标计算，包括RMSE、皮尔逊相关系数(R)和MAE
5. 生成了高质量的预测结果可视化，直观展示模型性能

## Project Structure

项目采用模块化设计，清晰分离数据处理、特征工程、模型训练和测试等功能，便于维护和扩展。

## Project Steps

### STEP 1

利用numpy,xarray,netCDF4进行数据读取及其预处理

1. `S1_DataProcess.py`

   **输入**：叶绿素，海流，营养盐最原始的卫星观测数据文件，陆地掩码。

   **逻辑**：对叶绿素先进行重采样至600*480网格，再根据相同分辨率的陆地掩码文件，对非陆地的nan值进行双线性局部插值，获取叶绿素值:64float。海流也重采样＋插值非陆地nan值，获取第一层水流u,v方向速度数据。营养盐数据也重采样＋插值非陆地nan值，获取第一层水流的fe,no,si,po4的值。

   **输出**：生成叶绿素，海流，营养盐的数据按天储存且时空对齐的数据文件在data下

   （纬度在原始卫星数据竟然是90\~0而不是0~90！）

2. `S1_DataCheck.py`

   **输入**：data/下的叶绿素，海流，营养盐数据文件

   **逻辑**：读取文件，计算叶绿素某些区域的nan值比例，输出信息。以确认数据都时空对齐。

   **输出**：三个数据的处理前处理后数据形状，并且抽样青岛附近海域的三个数据的前后对比。额外输出下叶绿素的nan值比例变化情况（因为叶绿素在项目前期nan值问题很严重）

3. `S1_CreateLandMask.py`

   **功能**：创建陆地掩码文件，用于标识陆地和海洋区域

5. `VisualizeData.py`

   **输入**：data/下的叶绿素，海流，营养盐数据文件

   **逻辑**：对三个数据文件进行逐天的数据可视化，以确定nan值正确处理以及数值预处理正确。

   **输出**：在visualize_result/文件下储存各个数据每天的数据可视化图。

### STEP 2

设计特征向量和标签向量，进行数据集生成。

1. `S2_FeatureEngineering.py`

   **输入**：叶绿素，海流，营养盐的数据文件

   **逻辑**：构建时空特征向量，包括历史叶绿素数据、海流数据和营养盐数据，形成训练样本

   **输出**：生成训练集(train.npz)、验证集(val.npz)和测试集(test.npz)，以及空间信息和陆地掩码文件

   （时间步的设计让我头痛死了，并且前期设计的1200*960原始叶绿素分辨率网格，使得特征工程极慢，接近3小时，并且伴随有严重的内存问题。最后使用低分辨率重采样＋分批数据加载保存解决。另外，特征工程必须保证无nan值，替换nan值为0并且在模型训练时提供掩码进行原始数据中nan值的忽略，否则模型的训练损失可能始终无效！）

2. `S2_CheckDataset.py`

   **功能**：检查特征向量和标签向量的完整性和质量。并且打印部分样本的特征向量和标签向量。

### STEP 3

模型训练与评估

1. `ModelTraining/STCNN_train.py`

   **功能**：实现STCNN模型的定义和训练过程
   
   **主要组件**：
   - 模型定义：三层3D卷积网络，用于捕捉时空特征
   - 自定义损失函数：考虑陆地掩码，忽略陆地区域的预测误差
   - 训练循环：支持GPU训练，快捷调整batch size,learning rate学习率,epochs训练轮次
   - 评估指标：计算RMSE、皮尔逊相关系数(R)和MAE
   - 可视化：训练和验证的损失曲线
   
   **输出**：训练好的模型权重文件(stcnn_model.pth)和训练日志

2. `ModelTraining/STCNN_test.py`

   **功能**：使用训练好的模型进行预测并进行评估，可视化预测结果与真实值的对比
   
   **主要组件**：
   - 模型加载：加载保存的模型权重
   - 预测生成：对测试集进行预测
   - 评估计算：计算测试集上的RMSE、R和MAE指标
   - 可视化：生成预测结果与真实值的对比图
   
   **输出**：预测结果可视化图表，保存在predictionVis目录

## Model Architecture

STCNN (Spatio-Temporal Convolutional Neural Network) 模型架构：

1. 输入层：接收8个特征通道（包括叶绿素历史数据、海流数据、营养盐数据等）
2. 3D卷积层1：32个输出通道，3×3×3卷积核，提取基本时空特征
3. ReLU激活函数
4. 3D卷积层2：64个输出通道，3×3×3卷积核，进一步提取深层特征
5. ReLU激活函数
6. 3D卷积层3：32个输出通道，3×3×3卷积核，融合特征
7. ReLU激活函数
8. 输出层：7个输出通道，1×1×1卷积核，预测未来7天的叶绿素浓度

## Evaluation Metrics

模型性能通过以下指标进行评估：

1. **均方根误差（RMSE）**：衡量预测值与真实值之间的平均误差
2. **皮尔逊相关系数（R）**：衡量预测值与真实值之间的线性相关性
3. **平均绝对误差（MAE）**：衡量预测值与真实值之间的绝对误差平均值


## Env Setup

项目依赖于以下主要库：

- Python 3.8
- PyTorch
- NumPy
- Xarray
- NetCDF4
- Matplotlib
- Cartopy

可通过以下命令创建conda环境（使用GPU）：
```bash
conda ML create -f env-GPU.yml
```

如果不想使用GPU进行训练则使用`env.yml`进行创建

## Usage

仓库并没有任何原始数据！如果需要复现项目请前往下面网址下载数据并独立进行数据预处理出特征工程想要的数据形状。

**1，全球数据：NASA MODIS/Aqua 叶绿素数据**

原始下载地址：

[https://climate.esa.int/en/projects/ocean-colour/](https://climate.esa.int/en/projects/ocean-colour/)

ftp网址：

[http://www.oceancolour.org](http://www.oceancolour.org)

登录账户信息：

FTP server: oceancolour.org

Username: oc-cci-data

Password: ELaiWai8ae

**2，中国近海数据：卫星叶绿素数据（区域：经度100-180，纬度0-60）**

**参考网址：**

[https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_BGC_001_028/services](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_BGC_001_028/services)

**3，近海区域海流信息（区域：经度100-180，纬度0-60）**

**参考网址：**

[https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/services](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/services)

1. 数据预处理：
   ```bash
   python S1_DataProcess.py
   python S1_DataCheck.py
   python VisualizeData.py
   ```

2. 特征工程：
   ```bash
   python S2_FeatureEngineering.py
   ```

3. 模型训练：
   ```bash
   cd ModelTraining
   python STCNN_train.py
   ```

4. 模型测试与可视化：
   ```bash
   python STCNN_test.py
   ```

## Future Improvements

1. 调整特征值(比如增加叶绿素相关的特征或者调整其权重,还有通过控制变量以选用特定的营养盐的值作为变量)

2. 生成置信区间(比如通过增加dropout层或者使用集成学习方法来生成置信区间)

3. 目前还是训练阶段，后续开发数据预测脚本，实现数据处理->特征工程->预测->预测结果＋评估一步到位

   ```txt
   netCDF结果：
   时间（time）
   经度（lon）
   纬度（lat）
   预测值（chlorophyll）
   不确定性字段（uncertainty）
   ```

4. 优化模型设计(等作者再深入学习下STCNN模型再说awa...)

## Last Words

感谢如今AI工具的强大以及模型的开源，使我一个新手小白也能敢于尝试自己的未知领域。豆包对项目进行的指导，解惑，还有AI IDE的强大补全功能，都为我提供了很大的帮助。模型的开源使得我即使没有太多机器学习的理论基础也能通过使用STCNN模型黑盒进行模型训练，数值预测。

在不断地遇到未知，学习未知、遇到困惑，解决困惑、遇到bug，解决bug的接近30天中，我很快乐。最激动的一刻莫过于，由于nan值问题，STCNN模型训练的损失始终无效，在修改了很多遍，重复了很多次的数据处理-特征工程-模型训练流程，在一个不被期待的夜晚，跑出了我人生第一个数值预测模型，那一刻，我感到一切的努力都是值得的，我所做的事情终于开花结果，很有意义。

我永远热爱深夜里那个对程序怀有无限期望的自己

2025/11/7 冬至

Cytosine

## Contact

1756877027@qq.com

希望有陌生人能对本项目提出宝贵的修改建议，希望我的项目能帮助到同样对机器学习感兴趣的人。