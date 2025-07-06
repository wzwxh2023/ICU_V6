"""
ICU预测系统 - 主应用入口
Medical ICU Prediction System - Main Application Entry Point
"""

import streamlit as st
import sys
import os

# 添加apps目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'apps'))

def main():
    st.set_page_config(
        page_title="ICU预测系统",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 ICU预测系统")
    st.markdown("---")
    
    # 侧边栏选择应用
    st.sidebar.title("选择预测模型")
    
    app_choice = st.sidebar.selectbox(
        "请选择要使用的模型：",
        [
            "CatBoost模型 (推荐)",
            "LightGBM模型",
            "最佳模型对比",
            "PubMed文献搜索"
        ]
    )
    
    # 根据选择加载不同的应用
    if app_choice == "CatBoost模型 (推荐)":
        try:
            from icu_catboost_smotetomek_005_app import main as catboost_app
            catboost_app()
        except ImportError as e:
            st.error(f"无法加载CatBoost应用: {e}")
            st.info("请确保所有依赖都已正确安装")
    
    elif app_choice == "LightGBM模型":
        try:
            from icu_lgbm_adasyn_030_app import main as lgbm_app
            lgbm_app()
        except ImportError as e:
            st.error(f"无法加载LightGBM应用: {e}")
            st.info("请确保所有依赖都已正确安装")
    
    elif app_choice == "最佳模型对比":
        try:
            from icu_best_model_app import main as best_app
            best_app()
        except ImportError as e:
            st.error(f"无法加载最佳模型应用: {e}")
            st.info("请确保所有依赖都已正确安装")
    
    elif app_choice == "PubMed文献搜索":
        try:
            from pubmed import main as pubmed_app
            pubmed_app()
        except ImportError as e:
            st.error(f"无法加载PubMed应用: {e}")
            st.info("请确保所有依赖都已正确安装")
    
    # 添加项目信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 项目信息")
    st.sidebar.info(
        """
        **ICU预测系统 v6.0**
        
        本系统用于预测患者在48小时内
        是否需要转入ICU的风险评估。
        
        - 🎯 高精度预测模型
        - 📊 可视化风险分析
        - 🔬 基于真实医疗数据
        - 🏥 临床决策支持
        """
    )
    
    # 添加使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 如何使用本系统：
        
        1. **选择模型**：在左侧选择要使用的预测模型
        2. **输入数据**：根据界面提示输入患者相关信息
        3. **获取预测**：点击预测按钮获取ICU转入风险评估
        4. **解释结果**：查看详细的风险分析和建议
        
        ### 模型说明：
        - **CatBoost模型**：推荐使用，具有最佳的预测性能
        - **LightGBM模型**：轻量级模型，运行速度快
        - **最佳模型对比**：可以比较不同模型的预测结果
        - **PubMed搜索**：查找相关医学文献
        
        ### 注意事项：
        - 本系统仅供医疗决策参考，不能替代专业医生诊断
        - 所有预测结果都应结合临床实际情况综合判断
        - 如有疑问，请咨询专业医疗人员
        """)

if __name__ == "__main__":
    main() 