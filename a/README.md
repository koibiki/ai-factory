# 天池工业AI大赛-智能制造质量预测  思路代码

First文件夹中存放了初赛最开始使用的PCA思路代码。

    初赛处理文件：按照TOOL的不同将原始数据分割成若干数据集，针对每个数据集分别使用PCA降维，最后合并形成训练集，后期舍弃该方案。
    线下验证：通过删除全空字段、删除数值过大字段、删除取值单一字段处理后，使用GBDT + SelectFromModel生成训练集，最后使用xgb、lgb单模型等模型、以及构建Stacking进行验证。
    线上验证：使用xgb单模型、Stacking模型提交结果。
    遗传算法：本来考虑用遗传算法在使用GBDT构建出特征候选集的基础上再筛选特征，但是整体框架编写完后发现需要调节的细节太多，后续只用到适应性函数来辅助线下验证。
         遗传算法框架：
             1.整体种群设置10个个体，每个个体的基因长度为候选特征集长度，基因编码采用0 1二进制法，最后随机生成基因序列初始化种群。
             2.适应函数采用10折线性回归结果。
             3.选择算子采用轮盘赌算法，选择该轮进化中适应性最高的个体。
             4.交叉算子采用两端交叉法，随机选择父本和母本的一段基因序列进行交换。
             5.变异算子采用随机变异法，初始设定变异概率阈值，每个个体生成一个随机数，若随机数小于阈值，则对应个体的随机基因点进行取反变异。
         tips：
             实际使用时发现，轮盘赌算法在数轮进化后，所有个体最终的适应性值趋于一致，但是该值并不是最优结果，可能由于在每轮进化中最优个体没有保存下来，而产生的最优个体在轮盘赌中被洗牌。这里的选择算子可考虑换成锦标赛算法等。
             适应性函数中使用线性回归主要是考虑缩短时间，这里应该使用树形模型作为适应性评价模型，以保持和特征选择所使用的模型一致。
             交叉算子所使用的两端交叉，与两个端点的随机选择直接相关，端点距离太远可能导致重要特征被替换，距离太近导致无法实现交叉的效果，这个可以考虑更加灵活的交叉方法，比如固定长度单段交叉、固定长度多段交叉等等。
             变异算子主要是变异阈值的选取，可以通过增加种群个体数量，增加遗传轮次进行调整。
             关于遗传算法的几个关键算子，还有很多方法和技巧...在后续的比赛中争取再优化调整。

Second文件夹中存放复赛代码。

    复赛处理文件：按照以下步骤生成训练数据：
        1.删除全空字段
        2.转换obj字段，将TOOL字段数值化编码
        3.删除类似日期的较大值
        4.删除全相同字段
        5.四分位处理异常值
        6.再删除一次全相同字段
        7.删除小方差字段
        8.使用GBDT + SelectFromModel筛选特征
        
        最后结果由xgb单模型与Stacking模型融合而成。