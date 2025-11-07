# Implementation Plan

本实现计划将 Titanic 生存分析系统的设计转化为具体的编码任务。任务按照依赖关系排序，确保每个步骤都可以基于前面的工作进行。

## Task List

- [ ] 1. 项目结构初始化
  - 创建项目目录结构（data/, src/, models/, reports/, app/）
  - 创建 requirements.txt 文件，列出所有依赖包
  - 创建 README.md 文件，说明项目概述和运行方法
  - 创建 .gitignore 文件，排除数据和模型文件
  - _Requirements: 12.1_

- [ ] 2. 数据清洗与预处理模块
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1-2.7_

- [ ] 2.1 实现数据加载和基本信息展示
  - 编写 load_raw_data() 函数读取 CSV 文件
  - 编写函数显示数据集基本信息（行数、列数、数据类型）
  - 编写函数识别和报告缺失值
  - 编写函数生成数值型字段的统计摘要
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2.2 实现缺失值处理
  - 编写函数处理 Age 缺失值（按 Title 中位数填补）
  - 编写函数处理 Embarked 缺失值（众数填补）
  - 编写函数处理 Cabin 缺失值（提取首字母或标记 Unknown）
  - 编写函数处理 Fare 缺失值（中位数填补）
  - 添加异常值检测和处理逻辑
  - _Requirements: 2.2, 2.3, 2.4_

- [ ] 2.3 实现特征工程
  - 编写 extract_title() 函数从 Name 提取称谓
  - 编写函数将稀有称谓映射到常见类别
  - 编写函数创建 FamilySize 特征（SibSp + Parch + 1）
  - 编写函数创建 IsAlone 特征
  - 编写函数创建 AgeGroup 特征（分组：0-12, 13-30, 31-50, 50+）
  - 编写函数创建 FareBin 特征（四分位数分组）
  - 编写函数提取 CabinClass 特征
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 2.4 实现类别变量编码
  - 编写函数对 Sex 进行二值编码
  - 编写函数对 Embarked 进行 One-hot 编码
  - 删除或标记不使用的列（PassengerId, Ticket, Name）
  - _Requirements: 2.5, 2.6_

- [ ] 2.5 保存清洗后的数据
  - 编写函数保存清洗后的数据为 titanic_cleaned.csv
  - 编写函数保存特征工程元数据为 JSON 文件
  - 添加日志记录功能
  - _Requirements: 2.7, 3.7, 3.8_

- [ ] 3. 方向A - 决策树分类模型
  - _Requirements: 5.1-5.6, 6.1-6.9_

- [ ] 3.1 实现数据准备和划分
  - 编写函数加载清洗后的数据
  - 编写函数选择特征列和目标列
  - 编写函数划分训练集和测试集（80/20，stratify，random_state=42）
  - 保存训练/测试集索引
  - _Requirements: 5.1, 5.2_

- [ ] 3.2 实现决策树模型训练
  - 编写 train_decision_tree() 函数
  - 设置超参数（criterion='entropy', max_depth=5, min_samples_leaf=20, class_weight='balanced'）
  - 训练模型并保存为 decision_tree.pkl
  - _Requirements: 5.3, 5.4, 5.5_

- [ ] 3.3 实现决策树评估和规则提取
  - 编写函数计算评估指标（Accuracy, Precision, Recall, F1, AUC）
  - 编写函数生成混淆矩阵
  - 编写函数绘制 ROC 曲线
  - 编写 extract_decision_rules() 函数提取 IF-THEN 规则
  - 编写 get_decision_path() 函数获取单个样本的决策路径
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 3.4 实现规则解释和反事实分析
  - 编写函数将决策路径转换为结构化文字说明
  - 编写函数生成"如果...会怎样"的反事实建议
  - 编写函数计算特征重要性
  - _Requirements: 6.7, 6.8_

- [ ] 3.5 实现模型偏差检查
  - 编写函数计算不同性别的误报率和漏报率
  - 编写函数计算不同舱位的预测准确率
  - 生成公平性报告
  - _Requirements: 6.9, 11.1, 11.2_

- [ ] 4. 方向A - 随机森林模型
  - _Requirements: 5.1-5.6, 6.1-6.4_

- [ ] 4.1 实现随机森林模型训练
  - 编写 train_random_forest() 函数
  - 设置超参数（n_estimators=100, max_depth=7, min_samples_leaf=10, n_jobs=-1）
  - 训练模型并保存为 random_forest.pkl
  - _Requirements: 5.3, 5.5_

- [ ] 4.2 实现随机森林评估
  - 编写函数计算评估指标
  - 编写函数提取特征重要性
  - 生成混淆矩阵和 ROC 曲线
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 5. 方向A - 梯度提升模型
  - _Requirements: 5.1-5.6, 6.1-6.4_

- [ ] 5.1 实现梯度提升模型训练
  - 编写 train_gradient_boosting() 函数
  - 设置超参数（n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8）
  - 训练模型并保存为 gradient_boosting.pkl
  - _Requirements: 5.3, 5.5_

- [ ] 5.2 实现梯度提升评估
  - 编写函数计算评估指标
  - 编写函数提取特征重要性
  - 生成混淆矩阵和 ROC 曲线
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 6. 方向A - 模型对比
  - _Requirements: 6.1-6.4_

- [ ] 6.1 实现模型性能对比
  - 编写 compare_models() 函数
  - 生成模型对比表格（包含所有评估指标和训练时间）
  - 保存对比结果为 model_comparison.csv
  - _Requirements: 6.1, 6.3_


- [ ] 7. 方向B - 幸存者统计分析
  - _Requirements: 7.1-7.11_

- [ ] 7.1 实现幸存者数据筛选和基本统计
  - 编写 filter_survivors() 函数提取 Survived=1 的记录
  - 编写函数计算幸存者总人数和占比
  - 编写函数计算幸存者的平均年龄、中位数年龄
  - 编写函数计算幸存者的平均票价、中位数票价
  - _Requirements: 7.1, 7.2, 7.4, 7.6_

- [ ] 7.2 实现幸存者与非幸存者对比分析
  - 编写函数生成性别分布对比（柱状图）
  - 编写函数生成舱位分布对比（柱状图）
  - 编写函数生成年龄分布对比（直方图+箱型图）
  - 编写函数生成票价分布对比（箱型图）
  - 编写函数统计登船港口分布
  - 编写函数统计家庭规模和独行比例
  - _Requirements: 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

- [ ] 7.3 实现统计显著性检验
  - 编写函数对类别变量（Sex, Pclass, Embarked）进行卡方检验
  - 编写函数对连续变量（Age, Fare）进行 Mann-Whitney U 检验
  - 生成统计检验结果表格（包含 p 值）
  - _Requirements: 7.9, 7.10, 7.11_

- [ ] 8. 方向B - KMeans 聚类分析
  - _Requirements: 8.1-8.11_

- [ ] 8.1 实现聚类数据准备
  - 编写函数选择聚类特征（Age, Fare, Pclass, FamilySize）
  - 编写函数对特征进行标准化（StandardScaler）
  - 保存标准化器为 scaler.pkl
  - _Requirements: 8.1, 8.2_

- [ ] 8.2 实现最优聚类数确定
  - 编写函数计算不同 k 值（2-6）的轮廓系数
  - 编写函数绘制肘部曲线（Inertia vs k）
  - 编写函数绘制轮廓系数曲线
  - _Requirements: 8.3_

- [ ] 8.3 实现 KMeans 聚类训练
  - 编写 train_kmeans() 函数
  - 设置参数（n_clusters=3, init='k-means++', random_state=42）
  - 训练模型并保存为 kmeans.pkl
  - _Requirements: 8.4, 8.11_

- [ ] 8.4 实现 KMeans 聚类评估
  - 编写函数计算 Silhouette Score
  - 编写函数计算 Calinski-Harabasz Index
  - 编写函数计算 Davies-Bouldin Index
  - _Requirements: 8.5_

- [ ] 8.5 实现聚类画像生成
  - 编写 generate_cluster_profiles() 函数计算每个簇的统计特征
  - 编写函数计算每个簇的均值、中位数、众数
  - 编写 create_cluster_descriptions() 函数生成文字描述
  - 保存聚类画像为 cluster_profiles.csv
  - _Requirements: 8.6, 8.9, 8.10_

- [ ] 9. 方向B - 高斯混合模型聚类
  - _Requirements: 8.1-8.11_

- [ ] 9.1 实现 GMM 聚类训练
  - 编写 train_gmm() 函数
  - 设置参数（n_components=3, covariance_type='full', random_state=42）
  - 训练模型并保存为 gmm.pkl
  - _Requirements: 8.4, 8.11_

- [ ] 9.2 实现 GMM 聚类评估
  - 编写函数计算评估指标
  - 编写函数提取每个样本的簇概率
  - 生成聚类画像
  - _Requirements: 8.5, 8.6_

- [ ] 10. 方向B - 层次聚类分析
  - _Requirements: 8.1-8.11_

- [ ] 10.1 实现层次聚类训练
  - 编写 train_hierarchical() 函数
  - 设置参数（n_clusters=3, linkage='ward'）
  - 保存聚类标签为 hierarchical_labels.pkl
  - _Requirements: 8.4, 8.11_

- [ ] 10.2 实现层次聚类可视化
  - 编写函数绘制树状图（dendrogram）
  - 编写函数评估聚类质量
  - 生成聚类画像
  - _Requirements: 8.5, 8.6_

- [ ] 11. 方向B - 聚类方法对比
  - _Requirements: 8.5_

- [ ] 11.1 实现聚类方法对比
  - 编写 compare_clustering_methods() 函数
  - 生成聚类方法对比表格（包含所有评估指标）
  - 保存对比结果为 cluster_comparison.csv
  - _Requirements: 8.5_

- [ ] 12. 可视化和报告生成模块
  - _Requirements: 4.1-4.8, 10.1-10.9_

- [ ] 12.1 实现方向A可视化
  - 编写函数绘制模型准确率对比柱状图
  - 编写函数绘制所有模型的 ROC 曲线对比
  - 编写函数绘制多指标热力图
  - 编写函数绘制所有模型的混淆矩阵（子图）
  - 编写函数绘制决策树结构图
  - 编写函数绘制特征重要性对比图
  - 编写函数导出决策规则文本
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 12.2 实现方向B统计对比可视化
  - 编写函数绘制性别分布对比图
  - 编写函数绘制舱位分布对比图
  - 编写函数绘制年龄分布对比图
  - 编写函数绘制票价分布对比图
  - 编写函数绘制特征分布图（按生存情况分组）
  - 编写函数绘制特征相关性热力图
  - _Requirements: 4.1, 4.3_

- [ ] 12.3 实现方向B聚类可视化
  - 编写函数绘制聚类方法对比图
  - 编写函数绘制肘部曲线
  - 编写函数绘制聚类散点图（Age-Fare，使用 PCA 降维）
  - 编写函数绘制各簇特征雷达图
  - 编写函数绘制层次聚类树状图
  - 编写函数绘制聚类画像表格可视化
  - 编写函数绘制各簇人数分布饼图
  - _Requirements: 4.5_

- [ ] 12.4 整合所有可视化
  - 编写 generate_all_visualizations() 函数
  - 确保所有图表保存到 reports/figures/ 目录
  - 确保所有表格保存到 reports/tables/ 目录
  - 添加图表样式统一化
  - _Requirements: 4.8_

- [ ] 13. Streamlit 交互界面 - Tab1 生存预测
  - _Requirements: 9.1-9.10_

- [ ] 13.1 实现模型和数据加载
  - 编写 load_models() 函数加载所有训练好的模型
  - 编写 load_data() 函数加载清洗后的数据
  - 使用 @st.cache_resource 缓存模型
  - 使用 @st.cache_data 缓存数据
  - _Requirements: 9.10_

- [ ] 13.2 实现用户输入表单
  - 创建 Streamlit 输入组件（Sex, Age, Pclass, Fare, SibSp, Parch, Embarked）
  - 添加输入验证（Age: 0-100, Fare >= 0）
  - 编写 preprocess_input() 函数处理用户输入
  - _Requirements: 9.2_

- [ ] 13.3 实现预测和解释功能
  - 编写 predict_with_explanation() 函数
  - 实现生存概率计算
  - 实现决策路径提取
  - 编写 generate_counterfactuals() 函数生成反事实建议
  - _Requirements: 9.3, 9.4, 9.5, 9.6_

- [ ] 13.4 实现预测结果展示
  - 使用进度条可视化生存概率
  - 显示预测结果（幸存/遇难）
  - 显示决策路径（规则列表）
  - 显示改善建议
  - _Requirements: 9.4, 9.5, 9.6_

- [ ] 14. Streamlit 交互界面 - Tab2 幸存者画像
  - _Requirements: 9.7, 9.8, 9.9_

- [ ] 14.1 实现基本统计展示
  - 显示幸存者总人数和占比
  - 显示性别幸存率对比
  - 显示舱位幸存率对比
  - _Requirements: 9.7_

- [ ] 14.2 实现统计对比图表展示
  - 嵌入性别分布对比图
  - 嵌入舱位分布对比图
  - 嵌入年龄分布对比图
  - 嵌入票价分布对比图
  - _Requirements: 9.7_

- [ ] 14.3 实现聚类结果展示
  - 显示聚类散点图
  - 显示各簇特征雷达图
  - 显示聚类画像描述（文字）
  - 显示各簇统计特征表格
  - _Requirements: 9.8, 9.9_

- [ ] 15. Streamlit 交互界面 - Tab3 模型对比
  - _Requirements: 9.1-9.10_

- [ ] 15.1 实现分类模型对比展示
  - 显示模型性能对比表格
  - 嵌入 ROC 曲线对比图
  - 嵌入特征重要性对比图
  - 嵌入混淆矩阵对比图
  - _Requirements: 9.1-9.10_

- [ ] 15.2 实现聚类方法对比展示
  - 显示聚类方法对比表格
  - 嵌入聚类评估指标对比图
  - _Requirements: 9.1-9.10_

- [ ] 16. 研究报告生成
  - _Requirements: 10.1-10.9_

- [ ] 16.1 创建报告模板
  - 创建 Markdown 或 LaTeX 报告模板
  - 定义报告章节结构（摘要、引言、方法、结果、讨论、结论、附录）
  - _Requirements: 10.1_

- [ ] 16.2 编写报告内容生成函数
  - 编写函数生成摘要部分
  - 编写函数生成引言部分（数据背景、研究意义）
  - 编写函数生成方法部分（数据处理、特征工程、模型方法）
  - 编写函数生成实验结果部分（嵌入图表和表格）
  - 编写函数生成讨论部分（两个方向结果的相互印证、模型偏差分析）
  - 编写函数生成结论部分（关键发现、改进方向）
  - 编写函数生成附录部分（代码链接、完整可视化）
  - _Requirements: 10.2-10.8_

- [ ] 16.3 导出报告
  - 编写函数将报告导出为 PDF 格式
  - 编写函数将报告导出为 Word 格式（可选）
  - 保存报告为 reports/Titanic_Report.pdf
  - _Requirements: 10.9_

- [ ] 17. 模型卡和文档完善
  - _Requirements: 12.7, 11.4, 11.5, 11.6, 11.7_

- [ ] 17.1 创建模型卡
  - 编写 model_card.md 文件
  - 记录模型版本、训练日期
  - 记录特征列表和特征工程方法
  - 记录超参数设置
  - 记录性能指标
  - 记录模型局限性和伦理考量
  - _Requirements: 12.7, 11.4, 11.5, 11.6_

- [ ] 17.2 完善 README 文档
  - 更新项目概述
  - 添加安装依赖说明
  - 添加运行步骤说明
  - 添加文件结构说明
  - 添加结果示例截图
  - _Requirements: 12.6_

- [ ] 18. 测试和验证
  - _Requirements: 所有需求_

- [ ] 18.1 单元测试
  - 编写数据处理函数的单元测试
  - 编写特征工程函数的单元测试
  - 编写模型训练和预测函数的单元测试
  - 编写聚类函数的单元测试
  - 确保测试覆盖率 > 80%
  - _Requirements: 所有需求_

- [ ] 18.2 集成测试
  - 测试端到端数据流（从原始数据到预测结果）
  - 测试模型训练到保存的完整流程
  - 测试 Streamlit 界面的交互功能
  - _Requirements: 所有需求_

- [ ] 18.3 性能验证
  - 验证模型性能达到目标（AUC >= 0.80）
  - 验证训练时间 < 5 分钟
  - 验证预测延迟 < 100ms
  - 验证内存使用 < 2GB
  - _Requirements: 所有需求_

- [ ] 18.4 交叉验证
  - 实现 5-fold 交叉验证
  - 测试不同随机种子的稳定性
  - 生成交叉验证报告
  - _Requirements: 所有需求_

- [ ] 19. 最终整合和优化
  - _Requirements: 所有需求_

- [ ] 19.1 代码优化
  - 重构重复代码
  - 添加类型注解
  - 优化性能瓶颈
  - 统一代码风格（PEP 8）
  - _Requirements: 12.2_

- [ ] 19.2 文档完善
  - 为所有函数添加 docstring
  - 更新所有文档
  - 添加使用示例
  - _Requirements: 12.6_

- [ ] 19.3 最终验收
  - 运行完整流程确保无错误
  - 检查所有输出文件是否生成
  - 验证 Streamlit 界面所有功能正常
  - 验证报告内容完整准确
  - _Requirements: 所有需求_
