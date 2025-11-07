# 部署说明 / Deployment Guide

## 版本兼容性问题解决 / Version Compatibility Fix

### 问题描述 / Problem Description

如果遇到以下错误：
```
Error loading models: Can't get attribute '__pyx_unpickle_CyHalfBinomialLoss' 
on <module 'sklearn._loss._loss' from '...'>
```

这通常是由以下原因导致的：
1. **Python 版本不兼容**：Streamlit Cloud 默认使用 Python 3.13，而 scikit-learn 1.3.2 可能不完全支持
2. **scikit-learn 版本不兼容**：不同版本的 scikit-learn 内部实现不同，导致无法加载旧版本训练的模型

### 解决方案 / Solution

#### ✅ 方法 1：指定 Python 版本（推荐，最简单）/ Specify Python Version (Recommended, Easiest)

**问题根源**：Streamlit Cloud 默认使用 Python 3.13，而 scikit-learn 1.3.2 不完全支持。

**解决方案**：创建 `runtime.txt` 文件指定 Python 3.11（已创建）

项目已经创建了 `runtime.txt` 文件，指定使用 Python 3.11.9：
```
3.11.9
```

**优点**：
- ✅ 不需要重新训练模型
- ✅ Python 3.11 是稳定版本，与 scikit-learn 1.3.2 完全兼容
- ✅ 最简单快捷的解决方案

**操作步骤**：
1. 确保 `runtime.txt` 文件已提交到 GitHub
2. 在 Streamlit Cloud 重新部署应用
3. Streamlit Cloud 会自动使用 Python 3.11

#### 方法 2：升级 scikit-learn（需要重新训练）/ Upgrade scikit-learn (Requires Retraining)

如果方法 1 不行，可以升级到支持 Python 3.13 的 scikit-learn 版本：

1. **更新 requirements.txt**
   ```txt
   scikit-learn==1.4.2
   ```

2. **重新训练所有模型**
   ```bash
   pip install scikit-learn==1.4.2
   python src/10_tree_classifier.py
   python src/20_survivor_clustering.py
   ```

3. **提交更新后的文件**
   ```bash
   git add requirements.txt models/
   git commit -m "Upgrade to scikit-learn 1.4.2 for Python 3.13 support"
   git push
   ```

#### 方法 3：使用固定版本（已更新）/ Use Fixed Version (Updated)

项目已经更新了 `requirements.txt`，固定了 scikit-learn 版本为 1.3.2。

在 Streamlit Cloud 部署时：
1. 确保 `requirements.txt` 包含 `scikit-learn==1.3.2`
2. 确保 `runtime.txt` 指定 Python 3.11
3. Streamlit Cloud 会自动安装正确的版本

### 当前配置 / Current Configuration

- **Python**: 3.11.9 (通过 runtime.txt 指定)
- **scikit-learn**: 1.3.2 (固定版本)
- **pandas**: >=1.3.0,<2.0.0
- **numpy**: >=1.21.0,<2.0.0
- **streamlit**: >=1.10.0
- **joblib**: >=1.2.0 (用于模型序列化)

### 应用改进 / Application Improvements

应用已经更新，现在可以：
- ✅ 在某个模型加载失败时继续运行
- ✅ 显示详细的错误信息
- ✅ 只使用可用的模型进行预测
- ✅ 提供清晰的错误提示和解决建议

### 部署到 Streamlit Cloud / Deploy to Streamlit Cloud

1. **推送代码到 GitHub**
   ```bash
   git add runtime.txt requirements.txt app/streamlit_app.py DEPLOYMENT.md
   git commit -m "Fix Python version compatibility for Streamlit Cloud"
   git push
   ```

2. **在 Streamlit Cloud 中**
   - 连接到你的 GitHub 仓库
   - 选择主分支
   - Streamlit Cloud 会自动：
     - 检测 `runtime.txt` 并使用 Python 3.11
     - 检测 `requirements.txt` 并安装依赖

3. **验证部署**
   - 检查 Streamlit Cloud 的日志，确认使用的是 Python 3.11
   - 确认所有依赖都正确安装
   - 测试模型加载是否正常

4. **如果仍有问题**
   - 检查 Streamlit Cloud 的日志
   - 确保所有模型文件都已提交到 GitHub
   - 如果模型文件太大，考虑使用 Git LFS
   - 尝试方法 2：升级 scikit-learn 并重新训练模型

### 注意事项 / Notes

- ⚠️ 模型文件（.pkl）必须与训练时使用的 scikit-learn 版本兼容
- ⚠️ 如果本地和云端使用不同版本的 scikit-learn，必须重新训练模型
- ✅ 建议在 `.gitignore` 中排除模型文件，但部署时需要包含它们
- ✅ 或者使用 Git LFS 管理大型模型文件

