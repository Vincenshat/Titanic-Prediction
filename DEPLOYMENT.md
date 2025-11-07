# 部署说明 / Deployment Guide

## 版本兼容性问题解决 / Version Compatibility Fix

### 问题描述 / Problem Description

如果遇到以下错误：
```
Error loading models: Can't get attribute '__pyx_unpickle_CyHalfBinomialLoss' 
on <module 'sklearn._loss._loss' from '...'>
```

这是因为 scikit-learn 版本不兼容导致的。不同版本的 scikit-learn 内部实现不同，导致无法加载旧版本训练的模型。

### 解决方案 / Solution

#### 方法 1：重新训练模型（推荐）/ Retrain Models (Recommended)

1. **确保使用正确的 scikit-learn 版本**
   ```bash
   pip install scikit-learn==1.3.2
   ```

2. **重新训练所有模型**
   ```bash
   python src/10_tree_classifier.py
   python src/20_survivor_clustering.py
   ```

3. **提交更新后的模型文件到 GitHub**
   ```bash
   git add models/
   git commit -m "Update models with scikit-learn 1.3.2"
   git push
   ```

#### 方法 2：使用固定版本（已更新）/ Use Fixed Version (Updated)

项目已经更新了 `requirements.txt`，固定了 scikit-learn 版本为 1.3.2。

在 Streamlit Cloud 部署时：
1. 确保 `requirements.txt` 包含 `scikit-learn==1.3.2`
2. Streamlit Cloud 会自动安装正确的版本
3. 如果模型仍然无法加载，请使用方法 1 重新训练

### 当前配置 / Current Configuration

- **scikit-learn**: 1.3.2 (固定版本)
- **pandas**: >=1.3.0,<2.0.0
- **numpy**: >=1.21.0,<2.0.0
- **streamlit**: >=1.10.0

### 应用改进 / Application Improvements

应用已经更新，现在可以：
- ✅ 在某个模型加载失败时继续运行
- ✅ 显示详细的错误信息
- ✅ 只使用可用的模型进行预测
- ✅ 提供清晰的错误提示和解决建议

### 部署到 Streamlit Cloud / Deploy to Streamlit Cloud

1. **推送代码到 GitHub**
   ```bash
   git add .
   git commit -m "Fix scikit-learn version compatibility"
   git push
   ```

2. **在 Streamlit Cloud 中**
   - 连接到你的 GitHub 仓库
   - 选择主分支
   - Streamlit Cloud 会自动检测 `requirements.txt` 并安装依赖

3. **如果仍有问题**
   - 检查 Streamlit Cloud 的日志
   - 确保所有模型文件都已提交到 GitHub
   - 如果模型文件太大，考虑使用 Git LFS

### 注意事项 / Notes

- ⚠️ 模型文件（.pkl）必须与训练时使用的 scikit-learn 版本兼容
- ⚠️ 如果本地和云端使用不同版本的 scikit-learn，必须重新训练模型
- ✅ 建议在 `.gitignore` 中排除模型文件，但部署时需要包含它们
- ✅ 或者使用 Git LFS 管理大型模型文件

