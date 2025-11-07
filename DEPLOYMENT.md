# 部署说明 / Deployment Guide

## Python 3.13 配置 / Python 3.13 Configuration

### 当前配置 / Current Configuration

项目已配置为使用 **Python 3.13**，所有依赖包版本已更新以支持 Python 3.13：

- **Python**: 3.13.0 (通过 runtime.txt 指定)
- **pandas**: >=2.2.3 (支持 Python 3.13)
- **scikit-learn**: >=1.6.0 (支持 Python 3.13)
- **numpy**: >=1.26.0 (支持 Python 3.13)
- **matplotlib**: >=3.8.0
- **seaborn**: >=0.13.0
- **streamlit**: >=1.40.0
- **scipy**: >=1.13.0
- **joblib**: >=1.4.0

### 重要提示 / Important Notes

⚠️ **模型需要重新训练**：由于升级到 Python 3.13 和 scikit-learn>=1.6.0，所有模型文件必须使用新版本重新训练。

### 部署步骤 / Deployment Steps

#### 1. 本地重新训练模型 / Retrain Models Locally

在部署之前，需要在本地使用新的依赖版本重新训练所有模型：

```bash
# 安装兼容 Python 3.13 的依赖
pip install -r requirements.txt

# 重新训练分类模型
python src/10_tree_classifier.py

# 重新训练聚类模型
python src/20_survivor_clustering.py
```

#### 2. 提交更新的文件 / Commit Updated Files

```bash
# 添加所有更新的文件
git add runtime.txt requirements.txt app/streamlit_app.py models/

# 提交更改
git commit -m "Update to Python 3.13 with compatible dependencies"

# 推送到 GitHub
git push
```

#### 3. Streamlit Cloud 部署 / Streamlit Cloud Deployment

Streamlit Cloud 会自动：
- 检测 `runtime.txt` 并使用 Python 3.13.0
- 检测 `requirements.txt` 并安装兼容版本的依赖
- 加载重新训练的模型文件

### 版本兼容性问题解决 / Version Compatibility Fix

如果遇到以下错误：
```
Error loading models: Can't get attribute '__pyx_unpickle_CyHalfBinomialLoss' 
on <module 'sklearn._loss._loss' from '...'>
```

**原因**：模型文件是用旧版本的 scikit-learn (<1.6.0) 训练的，与新版本不兼容。

**解决方案**：按照上述步骤重新训练模型。

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

