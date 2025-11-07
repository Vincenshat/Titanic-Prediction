# Requirements Document

## Introduction

本项目旨在基于经典的 Titanic 数据集进行全面的生存分析研究。项目包含两个核心研究方向：

**方向A（预测）**：构建可解释的决策树模型预测乘客生存概率，并提供基于规则的决策路径解释和改善建议。

**方向B（画像）**：深入分析真实幸存者（Survived=1）的群体特征，通过统计分析和聚类识别典型幸存者类型，并生成可读的人群画像描述。

项目采用轻量级建模策略（决策树 + KMeans），确保模型可解释性和可追溯性。最终输出包括完整的研究报告、可视化结果和交互式 Demo，符合数据科学项目的完整流程标准。

## Success Criteria

- 模型性能：AUC ≥ 0.80，准确率 ≥ 75%
- 可解释性：决策树规则清晰，聚类簇有明确特征区分
- 完整性：包含数据清洗、特征工程、建模、评估、可视化、报告全流程
- 可复现性：固定随机种子，保存模型文件和训练集划分
- 交互性：提供 Streamlit Demo 支持实时预测和画像展示

## Requirements

### Requirement 1: 数据导入与理解

**User Story:** 作为数据分析师，我希望能够导入 Titanic 数据集并理解各字段含义，以便为后续分析奠定基础。

#### Acceptance Criteria

1. WHEN 系统启动时 THEN 系统 SHALL 成功读取 Titanic-Dataset.csv 文件
2. WHEN 数据加载完成后 THEN 系统 SHALL 显示数据集的基本信息（行数、列数、数据类型）
3. WHEN 用户查看数据概览时 THEN 系统 SHALL 展示前几行样本数据
4. WHEN 系统分析数据质量时 THEN 系统 SHALL 识别并报告每列的缺失值数量和比例
5. WHEN 系统生成数据描述时 THEN 系统 SHALL 输出数值型字段的统计摘要（均值、标准差、分位数）

### Requirement 2: 数据清洗与预处理

**User Story:** 作为数据工程师，我希望对原始数据进行清洗和预处理，以便获得高质量的分析数据集。

#### Acceptance Criteria

1. WHEN 处理无用列时 THEN 系统 SHALL 删除或标记 PassengerId、Ticket 等对预测无意义的列
2. WHEN 处理 Age 缺失值时 THEN 系统 SHALL 根据称谓（Title）的中位数进行填补
3. WHEN 处理 Embarked 缺失值时 THEN 系统 SHALL 使用众数填补
4. WHEN 处理 Cabin 缺失值时 THEN 系统 SHALL 提取首字母作为舱位区域特征或标记为 "Unknown"
5. WHEN 编码 Sex 字段时 THEN 系统 SHALL 将其转换为二值变量（male=0, female=1）
6. WHEN 编码 Embarked 字段时 THEN 系统 SHALL 使用 One-hot 编码生成 C、Q、S 三个二值列
7. WHEN 数据清洗完成后 THEN 系统 SHALL 输出清洗后的数据集并保存为 CSV 文件

### Requirement 3: 特征工程

**User Story:** 作为机器学习工程师，我希望从原始数据中提取和构造有意义的特征，以便提升模型预测性能和可解释性。

#### Acceptance Criteria

1. WHEN 提取称谓特征时 THEN 系统 SHALL 从 Name 字段中解析出 Mr、Mrs、Miss、Master 等称谓
2. WHEN 处理稀有称谓时 THEN 系统 SHALL 使用映射规则将稀有称谓（如 Don、Countess）归类到常见类别
3. WHEN 计算家庭规模时 THEN 系统 SHALL 创建 FamilySize 特征（FamilySize = SibSp + Parch + 1）
4. WHEN 判断是否独行时 THEN 系统 SHALL 创建 IsAlone 二值特征（FamilySize == 1 时为 1）
5. WHEN 分组票价时 THEN 系统 SHALL 根据四分位数创建 FareBin 分类特征（Q1-Q4）
6. WHEN 分组年龄时 THEN 系统 SHALL 创建 AgeGroup 特征（0-12, 13-30, 31-50, 50+）
7. WHEN 特征工程完成后 THEN 系统 SHALL 输出扩充后的特征表并保存为 CSV 文件
8. WHEN 保存特征时 THEN 系统 SHALL 记录特征工程的转换逻辑和参数（如分位数阈值）

### Requirement 4: 探索性数据分析（EDA）

**User Story:** 作为数据科学家，我希望通过可视化和统计分析探索数据模式，以便发现影响生存率的关键因素。

#### Acceptance Criteria

1. WHEN 分析性别影响时 THEN 系统 SHALL 绘制性别与生存率的对比柱状图
2. WHEN 分析舱位影响时 THEN 系统 SHALL 绘制不同舱位等级的生存率对比图
3. WHEN 分析年龄分布时 THEN 系统 SHALL 绘制年龄直方图并叠加生存情况
4. WHEN 分析登船港口影响时 THEN 系统 SHALL 绘制不同登船港口的生存率对比图
5. WHEN 分析票价影响时 THEN 系统 SHALL 绘制票价分布的箱型图并按生存情况分组
6. WHEN 计算生存率统计时 THEN 系统 SHALL 输出总体生存人数、生存比例
7. WHEN 进行分组统计时 THEN 系统 SHALL 计算按性别、舱位、称谓、是否独行分组的平均生存率
8. WHEN 生成 EDA 报告时 THEN 系统 SHALL 保存所有可视化图表为图片文件

### Requirement 5: 方向A - 生存预测模型构建

**User Story:** 作为机器学习工程师，我希望构建可解释的决策树模型预测乘客生存概率，以便实现自动化且透明的生存预测功能。

#### Acceptance Criteria

1. WHEN 准备建模数据时 THEN 系统 SHALL 将数据集划分为训练集（80%）和测试集（20%），使用 stratify 保持类别平衡
2. WHEN 选择特征时 THEN 系统 SHALL 使用 Pclass、Sex、Age、SibSp、Parch、Fare、Embarked 及派生特征作为输入
3. WHEN 训练决策树模型时 THEN 系统 SHALL 使用 DecisionTreeClassifier，参数设置为 criterion='entropy', max_depth=4-5, min_samples_leaf=20, class_weight='balanced'
4. WHEN 设置随机种子时 THEN 系统 SHALL 使用固定的 random_state=42 确保可复现性
5. WHEN 模型训练完成后 THEN 系统 SHALL 保存训练好的模型为 tree_model.pkl 文件
6. WHEN 保存模型时 THEN 系统 SHALL 同时保存特征标准化器（如使用）和训练集索引

### Requirement 6: 方向A - 模型评估与解释

**User Story:** 作为数据科学家，我希望全面评估模型性能并提供清晰的解释，以便验证模型可靠性并让用户理解预测逻辑。

#### Acceptance Criteria

1. WHEN 评估模型准确性时 THEN 系统 SHALL 计算并输出 Accuracy、Precision、Recall、F1-score 指标
2. WHEN 生成混淆矩阵时 THEN 系统 SHALL 可视化混淆矩阵热力图
3. WHEN 评估模型区分能力时 THEN 系统 SHALL 绘制 ROC 曲线并计算 AUC 值（目标 AUC ≥ 0.80）
4. WHEN 分析特征重要性时 THEN 系统 SHALL 输出并可视化基于信息增益的特征重要性排序
5. WHEN 导出决策规则时 THEN 系统 SHALL 从决策树提取 IF-THEN 规则路径（如 "Sex=female & Pclass≤2 → Survive"）
6. WHEN 用户输入新样本时 THEN 系统 SHALL 输出该样本的生存概率预测值和决策路径
7. WHEN 生成规则解释时 THEN 系统 SHALL 将决策路径转换为结构化的文字说明
8. WHEN 提供改善建议时 THEN 系统 SHALL 基于决策树规则生成"如果...会怎样"的反事实分析
9. WHEN 检查模型偏差时 THEN 系统 SHALL 计算不同性别和舱位的误报率和漏报率

### Requirement 7: 方向B - 幸存者特征统计分析

**User Story:** 作为数据分析师，我希望深入分析幸存者群体的特征分布并与非幸存者对比，以便理解幸存者的典型画像和显著差异。

#### Acceptance Criteria

1. WHEN 筛选幸存者数据时 THEN 系统 SHALL 提取所有 Survived=1 的记录
2. WHEN 统计基本信息时 THEN 系统 SHALL 计算幸存者的总人数和占比
3. WHEN 分析性别分布时 THEN 系统 SHALL 绘制幸存者 vs 非幸存者的性别对比柱状图
4. WHEN 分析年龄分布时 THEN 系统 SHALL 绘制幸存者年龄直方图并计算平均年龄和中位数
5. WHEN 分析舱位分布时 THEN 系统 SHALL 绘制幸存者 vs 非幸存者的舱位等级对比图
6. WHEN 分析票价分布时 THEN 系统 SHALL 绘制幸存者 vs 非幸存者的票价箱型图
7. WHEN 分析登船港口时 THEN 系统 SHALL 统计幸存者在各登船港口的分布比例
8. WHEN 分析家庭规模时 THEN 系统 SHALL 统计幸存者的平均家庭规模和独行比例
9. WHEN 进行显著性检验时 THEN 系统 SHALL 对类别变量（Sex, Pclass, Embarked）使用卡方检验（χ²）
10. WHEN 进行显著性检验时 THEN 系统 SHALL 对连续变量（Age, Fare）使用 Mann-Whitney U 检验
11. WHEN 输出统计结果时 THEN 系统 SHALL 生成对比表格，包含均值、中位数、p值等统计量

### Requirement 8: 方向B - 幸存者聚类分析

**User Story:** 作为数据科学家，我希望对幸存者进行聚类分析并生成可读的群体画像，以便识别和理解不同类型的幸存者群体特征。

#### Acceptance Criteria

1. WHEN 准备聚类数据时 THEN 系统 SHALL 选择 Age、Fare、Pclass、FamilySize 等特征作为聚类输入
2. WHEN 标准化数据时 THEN 系统 SHALL 对聚类特征进行标准化处理（StandardScaler）
3. WHEN 确定聚类数时 THEN 系统 SHALL 使用轮廓系数（Silhouette Score）评估最优聚类数（K=2~6）
4. WHEN 执行聚类时 THEN 系统 SHALL 使用 KMeans(n_clusters=3, random_state=42) 对幸存者进行分组
5. WHEN 评估聚类质量时 THEN 系统 SHALL 计算 Silhouette Score 和 Calinski-Harabasz 指数
6. WHEN 分析聚类结果时 THEN 系统 SHALL 输出每个聚类的统计特征（均值、中位数、众数）
7. WHEN 可视化聚类时 THEN 系统 SHALL 绘制 Age-Fare 散点图，用颜色区分不同聚类
8. WHEN 生成聚类画像时 THEN 系统 SHALL 为每个聚类生成雷达图展示多维特征
9. WHEN 生成画像描述时 THEN 系统 SHALL 基于统计特征使用规则模板生成群体描述
10. WHEN 输出画像时 THEN 系统 SHALL 生成类似"簇A：年轻、票价高、头等舱、家庭同行→'富裕家庭与青年'"的描述
11. WHEN 保存聚类模型时 THEN 系统 SHALL 保存 kmeans.pkl 和 scaler.pkl 文件

### Requirement 9: Streamlit 交互界面

**User Story:** 作为最终用户，我希望通过友好的 Web 界面输入乘客信息并获得预测结果和幸存者画像，以便直观地探索和使用分析成果。

#### Acceptance Criteria

1. WHEN 用户访问界面时 THEN 系统 SHALL 提供两个 Tab 页签：Tab1-生存预测、Tab2-幸存者画像
2. WHEN 用户在 Tab1 输入信息时 THEN 系统 SHALL 提供表单字段（Sex, Age, Pclass, Fare, SibSp, Parch, Embarked）
3. WHEN 用户提交预测请求时 THEN 系统 SHALL 实时返回生存概率和二分类结果
4. WHEN 显示预测结果时 THEN 系统 SHALL 使用进度条可视化生存概率
5. WHEN 展示决策路径时 THEN 系统 SHALL 显示该样本经过的决策树规则
6. WHEN 生成解释时 THEN 系统 SHALL 显示基于规则的文字说明和改善建议
7. WHEN 用户访问 Tab2 时 THEN 系统 SHALL 显示幸存者统计图表（性别、舱位、年龄分布）
8. WHEN 展示聚类结果时 THEN 系统 SHALL 显示聚类散点图和每个簇的特征雷达图
9. WHEN 显示聚类画像时 THEN 系统 SHALL 展示每个簇的特征描述和统计摘要
10. WHEN 界面加载时 THEN 系统 SHALL 自动加载预训练的模型文件（tree_model.pkl, kmeans.pkl）

### Requirement 10: 研究报告生成

**User Story:** 作为项目负责人，我希望生成完整的研究报告文档，以便展示项目成果和研究发现。

#### Acceptance Criteria

1. WHEN 生成报告时 THEN 系统 SHALL 包含以下章节：摘要、引言、方法、实验结果、讨论、结论、附录
2. WHEN 撰写摘要时 THEN 系统 SHALL 简述项目目标、方法和主要发现
3. WHEN 撰写引言时 THEN 系统 SHALL 介绍 Titanic 数据背景和研究意义
4. WHEN 撰写方法时 THEN 系统 SHALL 详细描述数据处理、特征工程、建模和聚类方法
5. WHEN 展示实验结果时 THEN 系统 SHALL 包含模型性能对比表、特征重要性图、聚类结果图
6. WHEN 撰写讨论时 THEN 系统 SHALL 分析两个方向的结果如何相互印证
7. WHEN 撰写结论时 THEN 系统 SHALL 总结关键发现（如性别、舱位、年龄是关键生存因子）
8. WHEN 添加附录时 THEN 系统 SHALL 包含代码链接、完整可视化结果和数据表
9. WHEN 报告完成后 THEN 系统 SHALL 输出 PDF 或 Word 格式的文档

### Requirement 11: 模型伦理与偏差检查

**User Story:** 作为负责任的数据科学家，我希望识别和报告模型中的潜在偏差，以便确保研究的伦理性和透明度。

#### Acceptance Criteria

1. WHEN 评估模型公平性时 THEN 系统 SHALL 计算不同性别的误报率（False Positive Rate）和漏报率（False Negative Rate）
2. WHEN 评估模型公平性时 THEN 系统 SHALL 计算不同舱位等级的预测准确率差异
3. WHEN 生成公平性报告时 THEN 系统 SHALL 可视化不同群体的性能指标对比
4. WHEN 撰写报告时 THEN 系统 SHALL 在讨论部分注明历史和社会结构因素对数据的影响
5. WHEN 说明模型局限时 THEN 系统 SHALL 明确指出模型仅用于教学示例，不应用于实际决策
6. WHEN 分析结果时 THEN 系统 SHALL 讨论"女性和儿童优先"等历史救援政策对数据的影响
7. WHEN 报告偏差时 THEN 系统 SHALL 提供改进建议（如重采样、公平性约束等）

### Requirement 12: 项目结构与代码组织

**User Story:** 作为开发者，我希望项目具有清晰的目录结构和模块化代码，以便维护、扩展和复现。

#### Acceptance Criteria

1. WHEN 创建项目结构时 THEN 系统 SHALL 按以下结构组织文件：
   ```
   titanic-survival-analysis/
   ├── data/
   │   ├── Titanic-Dataset.csv (原始数据)
   │   └── titanic_cleaned.csv (处理后数据)
   ├── src/
   │   ├── 00_data_prep.py
   │   ├── 10_tree_classifier.py
   │   ├── 20_survivor_clustering.py
   │   └── 30_reports.py
   ├── models/
   │   ├── tree_model.pkl
   │   ├── kmeans.pkl
   │   └── scaler.pkl
   ├── reports/
   │   ├── figures/ (图表)
   │   ├── tables/ (统计表格)
   │   └── Titanic_Report.pdf
   ├── app/
   │   └── streamlit_app.py
   ├── requirements.txt
   ├── README.md
   └── model_card.md
   ```
2. WHEN 组织源代码时 THEN 系统 SHALL 确保每个模块职责单一且可独立运行
3. WHEN 保存中间结果时 THEN 系统 SHALL 将清洗后数据保存为 data/titanic_cleaned.csv
4. WHEN 保存模型时 THEN 系统 SHALL 将模型保存为 models/ 目录下的 .pkl 文件
5. WHEN 管理依赖时 THEN 系统 SHALL 提供 requirements.txt 文件，包含所有必需的 Python 包及版本号
6. WHEN 提供文档时 THEN 系统 SHALL 包含 README.md 说明项目目标、运行方法、文件结构、依赖安装
7. WHEN 记录实验时 THEN 系统 SHALL 创建 model_card.md 记录模型版本、特征、超参数、性能指标、训练日期
