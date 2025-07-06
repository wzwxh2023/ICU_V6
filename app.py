import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import requests
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 自定义预处理类定义 (from notebooks/preprocessing/02_preprocess.py) ---

class FeatureGeneratorWithNames(BaseEstimator, TransformerMixin):
    """支持特征名追踪的特征生成器"""
    
    def __init__(self):
        self.input_features_ = []
        self.output_features_ = []
        self.generated_features_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.input_features_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame):
        X_ = X.copy()
        initial_features = set(X_.columns)
        
        # 1. Shock Index (依赖: sbp, pulse)
        if all(c in X_.columns for c in ['sbp', 'pulse']):
            X_['shock_index'] = np.where(
                (X_['sbp'] > 0) & (X_['pulse'] > 0), 
                X_['pulse'] / X_['sbp'], 
                np.nan
            )
            X_['shock_flag'] = ((X_['shock_index'] > 0.9) & (X_['shock_index'].notna())).astype(int)

        # 2. SBP category (依赖: sbp)
        if 'sbp' in X_.columns:
            X_['sbp_cat'] = pd.cut(
                X_['sbp'],
                bins=[0, 90, 120, 140, 200, np.inf],
                labels=['hypo','normal','elev','htn1','htn2']
            ).astype(str)

        # 3. Temperature category (依赖: tempreture)
        if 'tempreture' in X_.columns:
            X_['temp_cat'] = pd.cut(
                X_['tempreture'],
                bins=[0, 35, 36.5, 37.5, 39, np.inf],
                labels=['hypo','normal','low_fever','fever','hyper']
            ).astype(str)

        # 4. Respiratory & Cardio flag (依赖: res, pulse)
        if all(c in X_.columns for c in ['res', 'pulse']):
            X_['resp_cardio_flag'] = (
                (X_['res'] > 30) & (X_['pulse'] > 120) & 
                (X_['res'].notna()) & (X_['pulse'].notna())
            ).astype(int)

        # 5. Vital abnormal count (依赖: sbp, tempreture, pulse, res)
        vital_cols = ['sbp', 'tempreture', 'pulse', 'res']
        if all(c in X_.columns for c in vital_cols):
            conditions = [
                ((X_['sbp'] < 90) | (X_['sbp'] > 180)) & (X_['sbp'].notna()),
                ((X_['tempreture'] < 36) | (X_['tempreture'] > 38.5)) & (X_['tempreture'].notna()),
                ((X_['pulse'] > 120) | (X_['pulse'] < 50)) & (X_['pulse'].notna()),
                ((X_['res'] > 30) | (X_['res'] < 10)) & (X_['res'].notna())
            ]
            X_['vital_abn_cnt'] = sum(cond.astype(int) for cond in conditions)

        # 6. MEWS high flag (依赖: mews_total)
        if 'mews_total' in X_.columns:
            X_['mews_high'] = ((X_['mews_total'] >= 5) & (X_['mews_total'].notna())).astype(int)

        # 7. 交互项: age * shock_index (依赖: age, shock_index)
        if all(c in X_.columns for c in ['age', 'shock_index']):
            X_['age_shock'] = X_['age'] * X_['shock_index'].fillna(0)

        # 8. 交互项: age * bmi (依赖: age, bmi)
        if all(c in X_.columns for c in ['age', 'bmi']):
            X_['age_bmi'] = X_['age'] * X_['bmi'].fillna(0)

        # 9. Critical value features
        crit_cols = ['exam_critical_flag', 'lab_critical_flag']
        if all(c in X_.columns for c in crit_cols):
            X_['any_critical_flag'] = (
                (X_['exam_critical_flag'] == 1) | (X_['lab_critical_flag'] == 1)
            ).astype(int)
            X_['critical_count'] = X_['exam_critical_flag'] + X_['lab_critical_flag']
            
            if 'age' in X_.columns and 'any_critical_flag' in X_.columns:
                X_['age_critical_interact'] = X_['age'] * X_['any_critical_flag']
            
            if 'mews_total' in X_.columns and 'any_critical_flag' in X_.columns:
                X_['mews_critical_interact'] = X_['mews_total'].fillna(0) * X_['any_critical_flag']

        # 10. 分箱特征
        if 'age' in X_.columns:
            X_['age_bucket'] = pd.cut(
                X_['age'], 
                bins=[0,40,60,120], 
                labels=['young','middle','old']
            ).astype(str)
        
        if 'res' in X_.columns:
            X_['res_bin'] = pd.cut(
                X_['res'], 
                bins=[0,12,20,30,60], 
                labels=['low','normal','tachy','extreme']
            ).astype(str)
        
        if 'bmi' in X_.columns:
            X_['bmi_level'] = pd.cut(
                X_['bmi'], 
                bins=[0,18.5,24,28,32,100],
                labels=['under','normal','over','obese','morbid']
            ).astype(str)

        # 追踪生成的特征
        final_features = set(X_.columns)
        self.generated_features_ = list(final_features - initial_features)
        self.output_features_ = list(X_.columns)
        
        return X_
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_features_)


class NamedAdaptivePCA(BaseEstimator, TransformerMixin):
    """支持特征名的自适应PCA"""
    
    def __init__(self, total_dims=80, variance_threshold=0.85, min_dims=3,
                 group_variance=None, group_max_dims=None, scale_features=True):
        self.total_dims = total_dims
        self.variance_threshold = variance_threshold
        self.group_variance = group_variance or {}
        self.group_max_dims = group_max_dims or {}
        self.min_dims = min_dims
        self.scale_features = scale_features
        self.dims_allocation_ = {}
        self.pcas_ = {}
        self.scalers_ = {}
        self.feature_names_out_ = []
        self.emb_col_mapping_ = {}
    
    def _dynamic_split_embeddings(self, X):
        """动态分组embedding"""
        if hasattr(X, 'columns'):
            columns = X.columns
            X_array = X.values
        else:
            if hasattr(self, 'emb_col_mapping_') and self.emb_col_mapping_:
                emb_groups = {}
                for prefix, col_indices in self.emb_col_mapping_.items():
                    start_idx = min(col_indices)
                    end_idx = max(col_indices) + 1
                    emb_groups[prefix] = X[:, start_idx:end_idx]
                return emb_groups
            else:
                return self._infer_embedding_groups(X)

        emb_groups = {}
        for prefix in ['diag', 'hist', 'exam', 'lab']:
            emb_cols = [i for i, col in enumerate(columns) if col.startswith(f'{prefix}_emb_')]
            if emb_cols:
                start_idx = min(emb_cols)
                end_idx = max(emb_cols) + 1
                emb_groups[prefix] = X_array[:, start_idx:end_idx]
                self.emb_col_mapping_[prefix] = emb_cols
        
        return emb_groups
    
    def _infer_embedding_groups(self, X):
        """从数据维度推断embedding分组"""
        emb_groups = {}
        total_cols = X.shape[1]
        
        if total_cols >= 400:
            group_size = total_cols // 4
            emb_groups['diag'] = X[:, :group_size]
            emb_groups['hist'] = X[:, group_size:2*group_size]
            emb_groups['exam'] = X[:, 2*group_size:3*group_size]
            emb_groups['lab'] = X[:, 3*group_size:]
        else:
            emb_groups['combined'] = X
        
        return emb_groups
    
    def fit(self, X, y=None):
        emb_groups = self._dynamic_split_embeddings(X)
        
        if not emb_groups:
            self.feature_names_out_ = []
            return self
        
        for name, emb_data in emb_groups.items():
            if emb_data.shape[1] == 0 or emb_data.shape[0] <= 1:
                continue
            
            if self.scale_features:
                self.scalers_[name] = StandardScaler(with_mean=False)
                emb_data_scaled = self.scalers_[name].fit_transform(emb_data)
            else:
                emb_data_scaled = emb_data
            
            var_thr = self.group_variance.get(name, self.variance_threshold)
            # 确保max_components不超过样本数-1和特征数
            max_components = min(emb_data_scaled.shape[1], emb_data_scaled.shape[0] - 1, 100)
            
            # 如果max_components <= 0，跳过这个组
            if max_components <= 0:
                continue
            
            pca_temp = PCA(n_components=max_components)
            pca_temp.fit(emb_data_scaled)
            
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            needed_dims = np.argmax(cumsum >= var_thr) + 1
            
            max_cap = self.group_max_dims.get(name, max_components)
            dims_final = int(min(max_cap, max(self.min_dims, needed_dims)))
            
            self.dims_allocation_[name] = dims_final
        
        self._normalize_allocation()
        
        self.feature_names_out_ = []
        for name, emb_data in emb_groups.items():
            if name in self.dims_allocation_:
                if self.scale_features and name in self.scalers_:
                    emb_data_scaled = self.scalers_[name].transform(emb_data)
                else:
                    emb_data_scaled = emb_data
                
                n_components = self.dims_allocation_[name]
                self.pcas_[name] = PCA(n_components=n_components, random_state=42)
                self.pcas_[name].fit(emb_data_scaled)
                
                for i in range(n_components):
                    var_explained = self.pcas_[name].explained_variance_ratio_[i]
                    feature_name = f'{name}_pca_{i:02d}_var{var_explained:.3f}'
                    self.feature_names_out_.append(feature_name)
        
        return self
    
    def transform(self, X):
        emb_groups = self._dynamic_split_embeddings(X)
        
        # 特殊处理：如果是单样本预测且PCA未训练，返回降维后的特征
        if X.shape[0] == 1 and not self.pcas_:
            # 对于单样本预测，我们需要创建一个简化的特征表示
            transformed_parts = []
            for name, emb_data in emb_groups.items():
                if emb_data.shape[1] > 0:
                    # 简单降维：取前N个特征
                    target_dims = min(20, emb_data.shape[1])  # 每组最多20维
                    simplified = emb_data[:, :target_dims]
                    transformed_parts.append(simplified)
            
            if transformed_parts:
                result = np.hstack(transformed_parts)
                return result
            else:
                return np.empty((X.shape[0], 0))
        
        transformed_parts = []
        for name, emb_data in emb_groups.items():
            if name in self.pcas_:
                if self.scale_features and name in self.scalers_:
                    emb_data_scaled = self.scalers_[name].transform(emb_data)
                else:
                    emb_data_scaled = emb_data
                
                transformed = self.pcas_[name].transform(emb_data_scaled)
                transformed_parts.append(transformed)
        
        if transformed_parts:
            result = np.hstack(transformed_parts)
            return result
        else:
            return np.empty((X.shape[0], 0))
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
    
    def _normalize_allocation(self):
        if not self.dims_allocation_:
            return
            
        total_needed = sum(self.dims_allocation_.values())
        
        if total_needed != self.total_dims:
            scale_factor = self.total_dims / total_needed
            for name in self.dims_allocation_:
                self.dims_allocation_[name] = max(
                    self.min_dims, 
                    int(self.dims_allocation_[name] * scale_factor)
                )


class NamedCombinedPreprocessor(BaseEstimator, TransformerMixin):
    """支持特征名的组合预处理器"""
    
    def __init__(self, emb_processor):
        self.emb_processor = emb_processor
        self.non_emb_processor = None
        self.emb_cols = []
        self.num_cols = []
        self.cat_cols = []
        self.feature_names_out_ = []
    
    def fit(self, X, y=None):
        self.emb_cols = [c for c in X.columns if c.startswith(('diag_emb_','hist_emb_','exam_emb_','lab_emb_'))]
        
        obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_categorical = ['exam_critical_flag', 'lab_critical_flag', 'o2', 'mews_aware']
        
        all_cat_cols = list(set(obj_cols + numeric_categorical))
        self.cat_cols = [c for c in all_cat_cols if c in X.columns]
        self.cat_cols.sort()
        
        self.num_cols = [c for c in X.columns 
                        if c not in self.emb_cols + self.cat_cols]
        self.num_cols.sort()
        
        transformers = []
        
        if self.num_cols:
            transformers.append(('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), self.num_cols))
        
        if self.cat_cols:
            transformers.append(('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), self.cat_cols))
        
        self.non_emb_processor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        X_non_emb = X.drop(columns=self.emb_cols)
        X_emb = X[self.emb_cols]
        
        if not X_non_emb.empty:
            self.non_emb_processor.fit(X_non_emb, y)
        
        if not X_emb.empty:
            self.emb_processor.fit(X_emb, y)
        
        self._generate_feature_names()
        
        return self
    
    def _generate_feature_names(self):
        """生成完整的特征名列表"""
        self.feature_names_out_ = []
        
        self.feature_names_out_.extend(self.num_cols)
        
        if self.cat_cols:
            cat_pipeline = self.non_emb_processor.named_transformers_['cat']
            if hasattr(cat_pipeline.named_steps['ohe'], 'get_feature_names_out'):
                cat_feature_names = cat_pipeline.named_steps['ohe'].get_feature_names_out(self.cat_cols)
                self.feature_names_out_.extend(cat_feature_names)
            else:
                for col in self.cat_cols:
                    self.feature_names_out_.append(f'{col}_encoded')
        
        if hasattr(self.emb_processor, 'feature_names_out_'):
            self.feature_names_out_.extend(self.emb_processor.feature_names_out_)
    
    def transform(self, X):
        X_non_emb = X.drop(columns=self.emb_cols)
        X_emb = X[self.emb_cols]
        
        transformed_parts = []
        
        if not X_non_emb.empty and hasattr(self, 'non_emb_processor'):
            X_non_emb_transformed = self.non_emb_processor.transform(X_non_emb)
            transformed_parts.append(X_non_emb_transformed)
        
        if not X_emb.empty:
            X_emb_transformed = self.emb_processor.transform(X_emb)
            if X_emb_transformed.shape[1] > 0:
                transformed_parts.append(X_emb_transformed)
        
        if transformed_parts:
            result = np.hstack(transformed_parts)
            return result
        else:
            return np.empty((X.shape[0], 0))
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

# --- Page Configuration ---
st.set_page_config(page_title="ICU Risk Prediction - CatBoost SMOTETomek 0.05", layout="wide")

# --- Configuration ---
DEFAULT_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/embeddings"
DEFAULT_SILICONFLOW_API_KEY = "YOUR_SILICONFLOW_API_KEY_HERE"
DEFAULT_BGE_MODEL_NAME = "BAAI/bge-m3"


# --- Model and File Paths ---
# 直接使用相对路径，适用于Streamlit Cloud部署
MODEL_PATH = 'results/feature_optimized_training_20250705_222545/best_model.pkl'
PIPELINE_PATH = 'processed_data/preprocess_pipeline.pkl'
FEATURES_PATH = 'processed_data/final_features_list.txt'

# 最佳模型信息（基于catboost_smotetomek_0.05模型）
MODEL_INFO = {
    'name': 'CatBoost with SMOTETomek (0.05) - Feature Optimized Best Model',
    'combination': 'catboost_smotetomek_0.05',
    'model_type': 'CatBoost',
    'sampler': 'SMOTETomek',
    'sampling_ratio': 0.05,
    'selection_basis': 'Validation AUPRC (Best Performance)',
    'stage1_cv_auprc': 0.2685,  # 第一阶段CV AUPRC
    'stage2_cv_auprc': 0.2863,  # 第二阶段CV AUPRC
    'valid_auprc': 0.3132,      # 验证集AUPRC (最高)
    'test_auprc': 0.3175,       # 测试集AUPRC
    'test_auc': 0.9230,         # 测试集AUC
    'test_recall': 0.3309,      # 测试集召回率
    'test_precision': 0.4894,   # 测试集精确率
    'test_f1': 0.3948,          # 测试集F1
    'optimal_threshold': 0.2142, # 最优阈值
    'generalization_score': 1.109,  # 泛化能力得分 (test_auprc/stage1_cv_auprc)
    'note': 'Best model after removing admission_unit feature - CatBoost excels at handling categorical features and imbalanced data'
}

# 阈值设置（基于最佳模型的阈值）
MODEL_THRESHOLD_OPTIMAL = MODEL_INFO['optimal_threshold']    # 0.2142 (最优阈值)
MODEL_THRESHOLD_MEDICAL = 0.18     # 医疗平衡阈值 (更保守)
MODEL_THRESHOLD_SENSITIVE = 0.03   # 高敏感性阈值 (更早预警) - 优化后的敏感性优先阈值
MODEL_THRESHOLD_SPECIFIC = 0.25    # 高特异性阈值 (减少假阳性)

# 更新模型信息的阈值
MODEL_INFO.update({
    'threshold_optimal': MODEL_THRESHOLD_OPTIMAL,
    'threshold_medical': MODEL_THRESHOLD_MEDICAL,
    'threshold_sensitive': MODEL_THRESHOLD_SENSITIVE,
    'threshold_specific': MODEL_THRESHOLD_SPECIFIC
})

# --- 资源加载函数 ---
@st.cache_resource
def load_resources():
    """Load model and preprocessing pipeline"""
    try:
        # 加载最佳模型
        if os.path.exists(MODEL_PATH):
            model_dict = joblib.load(MODEL_PATH)
            # 模型文件是字典格式，实际模型在pipeline键中
            if isinstance(model_dict, dict) and 'pipeline' in model_dict:
                model = model_dict['pipeline']
                st.success("✅ Model loaded successfully")
            else:
                model = model_dict  # 直接是模型对象
                st.success("✅ Model loaded successfully")
        else:
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None, None, None
        
        # 尝试加载预处理pipeline（直接使用，不重新创建）
        if os.path.exists(PIPELINE_PATH):
            try:
                # 直接加载预训练的pipeline
                preprocessing_pipeline = joblib.load(PIPELINE_PATH)
                st.success("✅ Preprocessing pipeline loaded successfully")
                return model, preprocessing_pipeline, None
            except Exception as e:
                st.error(f"❌ Failed to load preprocessing pipeline: {str(e)}")
                st.error("Cannot load pretrained pipeline, please check file integrity")
                return None, None, None
        else:
            st.error(f"❌ Pipeline file not found: {PIPELINE_PATH}")
            return None, None, None
            
        return model, pipeline, feature_names
        
    except Exception as e:
        st.error(f"❌ Resource loading failed: {str(e)}")
        return None, None, None

# --- Embedding API 函数 ---
def get_bge_embedding(text: str, api_key: str, api_url: str, model_name: str) -> list[float]:
    """Get BGE embedding"""
    if not text.strip():
        return [0.0] * 1024  # 返回零向量
    
    # 检查API配置
    if not api_key or api_key == "YOUR_SILICONFLOW_API_KEY_HERE":
        st.error("❌ SiliconFlow API Key not configured! Please enter a valid API Key in the sidebar")
        return [0.0] * 1024
    
    if not api_url or not model_name:
        st.error("❌ API URL or model name not configured!")
        return [0.0] * 1024
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "input": [text],
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
        elif response.status_code == 401:
            st.error("❌ API authentication failed! Please check if the API Key is correct")
        elif response.status_code == 429:
            st.error("❌ API rate limit exceeded! Please try again later")
        else:
            st.error(f"❌ API call failed! Status code: {response.status_code}")
        return [0.0] * 1024
    except requests.exceptions.Timeout:
        st.error("❌ API call timeout! Please check network connection")
        return [0.0] * 1024
    except Exception as e:
        st.error(f"❌ API call exception: {str(e)}")
        return [0.0] * 1024

# --- 特征处理函数 ---
def process_features_with_pipeline(data_input: dict, text_inputs: dict, sf_api_key: str, sf_api_url: str, sf_model_name: str, preprocessing_pipeline):
    """Process features using pretrained pipeline - fully matching training process"""
    try:
        st.info("🔧 Processing features...")
        
        # 步骤1: 创建完全匹配训练数据格式的DataFrame
        # 基础特征（与训练时完全一致）
        base_features = {
            'age': data_input.get('age', 50),
            'height': data_input.get('height', 170),
            'weight': data_input.get('weight', 65),
            'bmi': data_input.get('bmi', 25), 
            'pulse': data_input.get('pulse', 80),
            'tempreture': data_input.get('tempreture', 37),
            'sbp': data_input.get('sbp', 120),
            'res': data_input.get('res', 16),
            'mews_total': data_input.get('mews_total', 1),
            'gender': data_input.get('gender', 0),
            'admission_unit': data_input.get('admission_unit', 0),
            'surgey': data_input.get('surgey', 0),
            'intervention': data_input.get('intervention', 0),
            'exam_critical_flag': data_input.get('exam_critical_flag', 0),
            'lab_critical_flag': data_input.get('lab_critical_flag', 0),
            'o2': data_input.get('o2', 0),
            'mews_aware': data_input.get('mews_aware', 0)
        }
        
        # 步骤2: 获取文本嵌入（4组，每组1024维）
        st.info("🔄 Getting text embeddings...")
        
        text_mappings = {
            "admission_diagnosis": "diag",
            "history": "hist", 
            "exam_critical_value": "exam",
            "lab_critical_value": "lab"
        }
        
        embedding_features = {}
        zero_vector_count = 0
        
        for text_key, emb_prefix in text_mappings.items():
            try:
                text_content = text_inputs.get(text_key, "").strip()
                if not text_content or text_content.lower() in ['', 'none', 'n/a', 'na']:
                    # 提供有意义的默认文本
                    default_texts = {
                        "admission_diagnosis": "General medical condition under observation",
                        "history": "No significant past medical history reported", 
                        "exam_critical_value": "No critical physical examination findings",
                        "lab_critical_value": "No critical laboratory abnormalities detected"
                    }
                    text_content = default_texts[text_key]
                
                # 获取1024维嵌入向量
                embedding = get_bge_embedding(text_content, sf_api_key, sf_api_url, sf_model_name)
                if embedding and len(embedding) >= 1024 and not all(x == 0.0 for x in embedding[:10]):  # 检查前10个值是否都为0
                    # 使用与训练时完全相同的命名格式
                    for i in range(1024):
                        embedding_features[f'{emb_prefix}_emb_{i}'] = embedding[i]
                    st.success(f"✅ Successfully obtained {text_key} embedding: {len(embedding)} dims")
                else:
                    st.warning(f"⚠️ {text_key} embedding failed, using zero vector (will affect prediction accuracy)")
                    zero_vector_count += 1
                    for i in range(1024):
                        embedding_features[f'{emb_prefix}_emb_{i}'] = 0.0
                        
            except Exception as e:
                st.error(f"❌ Processing {text_key} failed: {e}")
                zero_vector_count += 1
                for i in range(1024):
                    embedding_features[f'{emb_prefix}_emb_{i}'] = 0.0
        
        # 显示embedding状态总结
        if zero_vector_count > 0:
            st.error(f"🚨 Warning: {zero_vector_count}/4 text embeddings used zero vector, this will significantly affect prediction accuracy!")
            st.error("Please configure correct SiliconFlow API Key to get accurate prediction results.")
        else:
            st.success("✅ All text embeddings obtained successfully!")
        
        # 步骤3: 合并所有特征为完整的DataFrame
        all_features = {**base_features, **embedding_features}
        input_df = pd.DataFrame([all_features])
        
        # 验证特征数量
        expected_emb_cols = 4 * 1024  # 4 groups of embedding, each 1024 dims
        actual_emb_cols = len([col for col in input_df.columns if '_emb_' in col])
        st.info(f"📊 Feature validation: {len(base_features)} base + {actual_emb_cols} embedding = {len(all_features)} total")
        
        # 步骤4: 应用预处理pipeline
        # Pipeline执行顺序: FeatureGenerator -> CombinedPreprocessor (non_emb + emb processing)
        st.info("🔄 Applying preprocessing pipeline...")
        processed_features = preprocessing_pipeline.transform(input_df)
        
        # 确保返回numpy数组用于模型预测
        if hasattr(processed_features, 'toarray'):
            processed_features = processed_features.toarray()
        
        st.success(f"✅ Preprocessing complete: {processed_features.shape}")
        return processed_features
        
    except Exception as e:
        st.error(f"❌ Preprocessing pipeline error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# --- Clinical Risk Assessment Function ---
def assess_clinical_risk(data_payload):
    """Clinical risk assessment rules"""
    risk_factors = []
    risk_score = 0
    
    # Basic vital signs risk
    if data_payload.get('sbp', 120) < 90:
        risk_factors.append("Hypotension (SBP < 90)")
        risk_score += 3
    elif data_payload.get('sbp', 120) > 180:
        risk_factors.append("Hypertensive crisis (SBP > 180)")
        risk_score += 2
    
    if data_payload.get('pulse', 80) > 120:
        risk_factors.append("Tachycardia (HR > 120)")
        risk_score += 2
    elif data_payload.get('pulse', 80) < 50:
        risk_factors.append("Bradycardia (HR < 50)")
        risk_score += 2
    
    if data_payload.get('tempreture', 37) > 38.5:
        risk_factors.append("Fever (Temp > 38.5°C)")
        risk_score += 1
    elif data_payload.get('tempreture', 37) < 36:
        risk_factors.append("Hypothermia (< 36°C)")
        risk_score += 2
    
    if data_payload.get('res', 16) > 30:
        risk_factors.append("Tachypnea (RR > 30)")
        risk_score += 2
    elif data_payload.get('res', 16) < 10:
        risk_factors.append("Respiratory depression (RR < 10)")
        risk_score += 3
    
    # Age risk
    age = data_payload.get('age', 50)
    if age > 80:
        risk_factors.append("Advanced age (> 80 years)")
        risk_score += 2
    elif age > 65:
        risk_factors.append("Elderly (> 65 years)")
        risk_score += 1
    
    # MEWS score risk
    mews = data_payload.get('mews_total', 0)
    if mews >= 5:
        risk_factors.append(f"High MEWS score ({mews} points)")
        risk_score += 3
    elif mews >= 3:
        risk_factors.append(f"Moderate MEWS score ({mews} points)")
        risk_score += 1
    
    # Surgery risk
    if data_payload.get('surgey', 0):
        risk_factors.append("Recent surgery")
        risk_score += 1
    
    # Calculate shock index
    shock_index = data_payload.get('pulse', 80) / max(data_payload.get('sbp', 120), 1)
    if shock_index > 0.9:
        risk_factors.append(f"Abnormal shock index ({shock_index:.2f})")
        risk_score += 3
    
    # Risk level
    if risk_score >= 8:
        risk_level = "🔴 Very High Risk"
        recommendation = "Immediate ICU assessment, continuous monitoring"
    elif risk_score >= 5:
        risk_level = "🟡 High Risk"
        recommendation = "Enhanced monitoring, consider ICU assessment"
    elif risk_score >= 3:
        risk_level = "🟠 Moderate Risk"
        recommendation = "Close observation, regular assessment"
    else:
        risk_level = "🟢 Low Risk"
        recommendation = "Routine monitoring"
    
    return {
        'risk_factors': risk_factors,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'recommendation': recommendation
    }

# --- Prediction Function ---
def make_prediction(processed_features, data_payload=None, threshold_mode="optimal"):
    """Make prediction using the best model"""
    model, _, _ = load_resources()
    
    if model is None or processed_features is None:
        return None
    
    try:
        # 获取预测概率
        pred_proba = model.predict_proba(processed_features)[0, 1]
        
        # 根据阈值模式选择阈值
        thresholds = {
            "optimal": MODEL_THRESHOLD_OPTIMAL,
            "medical": MODEL_THRESHOLD_MEDICAL, 
            "sensitive": MODEL_THRESHOLD_SENSITIVE,
            "specific": MODEL_THRESHOLD_SPECIFIC
        }
        
        threshold = thresholds.get(threshold_mode, MODEL_THRESHOLD_OPTIMAL)
        prediction = 1 if pred_proba >= threshold else 0
        
        # Risk level
        if pred_proba >= 0.3:
            risk_level = "🔴 Very High Risk"
        elif pred_proba >= 0.2:
            risk_level = "🟡 High Risk"
        elif pred_proba >= 0.1:
            risk_level = "🟠 Moderate Risk"
        else:
            risk_level = "🟢 Low Risk"
        
        # Clinical assessment
        clinical_assessment = None
        if data_payload:
            clinical_assessment = assess_clinical_risk(data_payload)
        
        return {
            'prediction': prediction,
            'probability': pred_proba,
            'threshold': threshold,
            'risk_level': risk_level,
            'clinical_assessment': clinical_assessment,
            'model_info': MODEL_INFO
        }
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

# --- Main Interface ---
def main():
    st.title("🏥 ICU Risk Prediction System")
    st.subheader("Powered by CatBoost + SMOTETomek (0.03)")
    
    # 侧边栏 - 简化配置
    with st.sidebar:
        # 阈值设置
        with st.expander("🎚️ Threshold Settings"):
            threshold_mode = st.selectbox(
                "Select prediction threshold mode",
                ["sensitive", "optimal", "medical", "specific"],
                format_func=lambda x: {
                    "optimal": f"Optimal ({MODEL_THRESHOLD_OPTIMAL:.2f})",
                    "medical": f"Medical Balance ({MODEL_THRESHOLD_MEDICAL:.2f})",
                    "sensitive": f"High Sensitivity ({MODEL_THRESHOLD_SENSITIVE:.2f})",
                    "specific": f"High Specificity ({MODEL_THRESHOLD_SPECIFIC:.2f})"
                }[x]
            )
        
        # API配置
        with st.expander("🔧 API Configuration (Required)", expanded=True):
            st.warning("⚠️ Text embedding requires SiliconFlow API configuration, otherwise prediction accuracy will be significantly reduced!")
            sf_api_key = st.text_input("SiliconFlow API Key", 
                                     value=DEFAULT_SILICONFLOW_API_KEY,
                                     type="password",
                                     help="Please enter a valid SiliconFlow API Key")
            sf_api_url = st.text_input("API URL", 
                                     value=DEFAULT_SILICONFLOW_API_URL)
            sf_model_name = st.text_input("Model Name", 
                                        value=DEFAULT_BGE_MODEL_NAME)
            
            # API状态检查
            if sf_api_key == DEFAULT_SILICONFLOW_API_KEY:
                st.error("🚨 API Key not configured! Zero vectors will be used instead of text embeddings")
            else:
                st.success("✅ API Key configured")
    
    # 主要输入区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Basic Information")
        
        # 基础信息输入 - 移除admission_unit以避免过拟合
        # admission_unit = st.selectbox("入院科室", [0, 1, 2], 
        #                             format_func=lambda x: ["科室A", "科室B", "科室C"][x])
        admission_unit = 0  # 设置默认值，不显示给用户
        age = st.number_input("Age", min_value=0, max_value=120, value=65)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: ["Female", "Male"][x])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        
        # 计算BMI
        bmi = weight / ((height/100) ** 2)
        st.info(f"BMI: {bmi:.1f}")
        
        # 生理指标
        st.subheader("🫀 Vital Signs")
        sbp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
        pulse = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
        tempreture = st.number_input("Temperature (°C)", min_value=33.0, max_value=45.0, value=37.0, step=0.1)
        res = st.number_input("Respiratory Rate (/min)", min_value=5, max_value=60, value=16)
        
        # MEWS相关
        st.subheader("🏥 Medical Assessment")
        mews_total = st.number_input("MEWS Total Score", min_value=0, max_value=20, value=2)
        mews_aware = st.selectbox("Consciousness Level", [0, 1, 2, 3], 
                                format_func=lambda x: ["Alert", "Drowsy", "Coma", "Other"][x])
        
        # 其他信息
        surgey = st.checkbox("Recent Surgery")
        intervention = st.checkbox("Intervention Treatment")
        o2 = st.selectbox("Oxygen Therapy", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    
    with col2:
        st.header("📝 Clinical Text")
        
        # 文本输入
        admission_diagnosis = st.text_area("Admission Diagnosis", 
                                         placeholder="Enter admission diagnosis...",
                                         height=100)
        
        history = st.text_area("Medical History", 
                             placeholder="Enter medical history...",
                             height=100)
        
        exam_critical_value = st.text_area("Critical Examination Values", 
                                         placeholder="Enter critical examination values...",
                                         height=80)
        
        lab_critical_value = st.text_area("Critical Laboratory Values", 
                                        placeholder="Enter critical laboratory values...",
                                        height=80)
        
        # 异常标记
        st.subheader("⚠️ Critical Flags")
        exam_critical_flag = st.checkbox("Examination Critical Flag")
        lab_critical_flag = st.checkbox("Laboratory Critical Flag")
    
    # 预测按钮
    if st.button("🔮 Predict ICU Risk", type="primary", use_container_width=True):
        # 准备数据
        data_input = {
            'admission_unit': admission_unit,
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'sbp': sbp,
            'pulse': pulse,
            'tempreture': tempreture,
            'res': res,
            'mews_total': mews_total,
            'mews_aware': mews_aware,
            'surgey': int(surgey),
            'intervention': int(intervention),
            'o2': o2,
            'exam_critical_flag': int(exam_critical_flag),
            'lab_critical_flag': int(lab_critical_flag)
        }
        
        text_inputs = {
            'admission_diagnosis': admission_diagnosis,
            'history': history,
            'exam_critical_value': exam_critical_value,
            'lab_critical_value': lab_critical_value
        }
        
        # 加载资源
        model, pipeline, _ = load_resources()
        
        if model and pipeline:
            # 处理特征
            processed_features = process_features_with_pipeline(
                data_input, text_inputs, 
                sf_api_key, sf_api_url, sf_model_name, 
                pipeline
            )
            
            if processed_features is not None:
                # 进行预测
                result = make_prediction(processed_features, data_input, threshold_mode)
                
                if result:
                    # 显示预测结果
                    st.header("🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ICU Admission Probability", f"{result['probability']:.1%}")
                    
                    with col2:
                        st.metric("Threshold Used", f"{result['threshold']:.2f}")
                    
                    with col3:
                        prediction_text = "ICU Required" if result['prediction'] else "No ICU Required"
                        st.metric("Prediction Result", prediction_text)
                    
                    # 风险等级
                    st.write(f"**Risk Level**: {result['risk_level']}")
                    
                    # 临床评估
                    if result['clinical_assessment']:
                        clinical = result['clinical_assessment']
                        
                        st.subheader("👨‍⚕️ Clinical Risk Assessment")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Risk Level**: {clinical['risk_level']}")
                            st.write(f"**Risk Score**: {clinical['risk_score']}")
                        
                        with col2:
                            st.write(f"**Recommendation**: {clinical['recommendation']}")
                        
                        if clinical['risk_factors']:
                            st.write("**Identified Risk Factors**:")
                            for factor in clinical['risk_factors']:
                                st.write(f"• {factor}")

if __name__ == "__main__":
    main()
