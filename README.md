# ICU预测系统 v6.0

## 项目简介

这是一个基于机器学习的ICU预测系统，用于预测患者在48小时内是否需要转入ICU的风险评估。该系统基于真实医疗数据训练，为临床决策提供支持。

## 🎯 主要功能

- **高精度预测模型**：使用CatBoost、LightGBM等先进算法
- **可视化风险分析**：提供详细的风险评估和解释
- **多模型对比**：支持不同模型的预测结果比较
- **文献搜索**：集成PubMed文献搜索功能
- **用户友好界面**：基于Streamlit的直观Web界面

## 🏥 应用场景

- 早期预警系统
- 医疗决策支持
- 风险评估工具
- 临床研究辅助

## 📊 数据特点

- **数据类型**：高度不平衡医疗数据集
- **阳性样本比例**：< 1%
- **预测任务**：二分类问题
- **时间窗口**：48小时内ICU转入风险

## 🚀 快速开始

### 在线访问

访问我们的在线版本：[ICU预测系统](https://your-app-url.streamlit.app)

### 本地运行

1. 克隆项目
```bash
git clone https://github.com/wenxuehuan/ICU_V6.git
cd ICU_V6
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
streamlit run app.py
```

## 🔧 技术栈

- **机器学习**：CatBoost, LightGBM, XGBoost
- **数据处理**：Pandas, NumPy, Scikit-learn
- **可视化**：Matplotlib, Seaborn, Plotly
- **Web框架**：Streamlit
- **模型解释**：SHAP, LIME

## 📁 项目结构

```
ICU_V6/
├── app.py                  # 主应用入口
├── apps/                   # Streamlit应用
├── data/                   # 数据存储
├── notebooks/              # Jupyter notebooks
├── src/                    # 源代码
├── results/                # 实验结果
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 🎯 模型性能

- **AUROC**: > 0.85
- **AUPRC**: > 0.30
- **Sensitivity**: > 0.80
- **Specificity**: > 0.85

## ⚠️ 重要声明

- 本系统仅供医疗决策参考，不能替代专业医生诊断
- 所有预测结果都应结合临床实际情况综合判断
- 如有疑问，请咨询专业医疗人员

## 📄 许可证

本项目仅供学术研究使用。

## 👥 贡献者

- wenxuehuan@gmail.com

## 📞 联系我们

如有问题或建议，请联系：wenxuehuan@gmail.com 