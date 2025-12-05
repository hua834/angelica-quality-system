# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import platform
# 引入分类器 (Classifier) 用于鉴别，回归器 (Regressor) 用于评分
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request

# ==================== 0. 页面配置 ====================
st.set_page_config(
    page_title="当归酒制品智能鉴别与评价系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS 美化界面
st.markdown("""
    <style>
    h1 {color: #2C3E50; font-family: "Microsoft YaHei";}
    .result-card {
        background-color: #f0f8ff;
        border-left: 5px solid #2980B9;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-text {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 全局字体修复 (防止云端乱码) ====================
@st.cache_resource
def configure_chinese_font():
    """自动检测并加载中文字体"""
    system_fonts = fm.findSystemFonts()
    found_font = None
    # 优先找系统自带字体
    for font_path in system_fonts:
        if 'SimHei' in font_path or 'Microsoft YaHei' in font_path or 'PingFang' in font_path:
            found_font = font_path
            break
            
    # 如果没找到（比如在云端Linux），则下载 SimHei
    if not found_font:
        font_filename = "SimHei.ttf"
        if not os.path.exists(font_filename):
            url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
            try:
                with st.spinner("正在配置云端字体环境..."):
                    urllib.request.urlretrieve(url, font_filename)
                found_font = font_filename
            except:
                pass
        else:
            found_font = font_filename

    # 注册字体
    if found_font:
        fm.fontManager.addfont(found_font)
        font_prop = fm.FontProperties(fname=found_font)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        return True
    return False

configure_chinese_font()

# ==================== 2. 定义指标 (关键配置) ====================
# 理化指标 (7项)
CHEM_COLS = [
    '多糖含量(mg/g)', '阿魏酸含量(%)', '总灰分(%)', '酸不溶性灰分(%)', 
    '挥发油含量(mL/g)', '水分(%)', '浸出物含量(%)'
]
# 鉴别目标 (5类)
TARGET_TYPES = ['生当归', '酒炙当归', '酒洗当归', '酒炒当归', '酒浸当归']

# 指标方向 (用于辅助评分: True越大越好, False越小越好)
CHEM_DIRS = [True, True, False, False, True, False, True]

# 【已修正】只保留 8 个传感器
SENSOR_COLS = [f'传感器{i}' for i in range(1, 9)]

# ==================== 3. 核心算法引擎 ====================
class IntelligentSystem:
    def __init__(self, df):
        self.raw_df = df.copy() # 复制一份，避免修改原数据
        # 提取数据矩阵
        self.chem_data = df[CHEM_COLS].values
        self.labels = df['样品类型'].values
        
        # 检查传感器数据是否存在
        if SENSOR_COLS[0] in df.columns:
            self.sensor_data = df[SENSOR_COLS].values
        else:
            self.sensor_data = None

        # 1. 训练 TOPSIS 评分标准 (作为辅助参考)
        self.topsis_scores = self._calc_topsis_score()
        
        # 【关键修复】：将计算出的综合得分写入 dataframe，供后续查询
        self.raw_df['综合得分'] = self.topsis_scores
        
        # 2. 训练 Q-Marker 筛选器
        self._train_qmarker_selector()
        
        # 3. 训练 鉴别分类器 (核心)
        self._train_classifiers()

    def _calc_topsis_score(self):
        """计算历史数据的 TOPSIS 得分"""
        X = self.chem_data
        # 极差归一化 + 熵权法
        min_v, max_v = X.min(axis=0), X.max(axis=0)
        norm = (X - min_v) / (max_v - min_v + 1e-10)
        p = norm / norm.sum(axis=0)
        k = 1 / np.log(len(X))
        entropy = -k * np.sum(p * np.log(p + 1e-10), axis=0)
        weights = (1 - entropy) / (1 - entropy).sum()
        
        self.weights = weights # 保存权重
        self.min_vals = min_v
        self.max_vals = max_v
        
        # 向量归一化
        self.norm_scale = np.sqrt((X**2).sum(axis=0))
        z_weighted = (X / self.norm_scale) * weights
        
        # 正负理想解
        pos = np.array([z_weighted[:, i].max() if CHEM_DIRS[i] else z_weighted[:, i].min() for i in range(7)])
        neg = np.array([z_weighted[:, i].min() if CHEM_DIRS[i] else z_weighted[:, i].max() for i in range(7)])
        
        d_pos = np.sqrt(((z_weighted - pos)**2).sum(axis=1))
        d_neg = np.sqrt(((z_weighted - neg)**2).sum(axis=1))
        return d_neg / (d_pos + d_neg + 1e-10)

    def _train_qmarker_selector(self):
        """利用随机森林筛选对'分类'最重要的指标"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.chem_data, self.labels)
        
        importances = rf.feature_importances_
        # 选出前3名
        indices = np.argsort(importances)[::-1][:3]
        self.q_markers = [CHEM_COLS[i] for i in indices]
        
        # 训练 Q-Marker 专用分类器
        self.q_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.q_clf.fit(self.raw_df[self.q_markers], self.labels)

    def _train_classifiers(self):
        """训练全指标和电子鼻的分类模型"""
        # 全指标分类器
        self.full_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.full_clf.fit(self.chem_data, self.labels)
        
        # 电子鼻分类器
        if self.sensor_data is not None:
            self.enose_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.enose_clf.fit(self.sensor_data, self.labels)
            
        # 辅助回归器 (用于给新样品估算一个分数)
        self.score_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.score_reg.fit(self.chem_data, self.topsis_scores)

    # --- 预测功能 ---
    def predict(self, model, input_vals, feat_names):
        # 1. 鉴别类型
        pred_type = model.predict([input_vals])[0]
        # 2. 置信度
        probs = model.predict_proba([input_vals])[0]
        confidence = np.max(probs)
        
        # 3. 估算评分 (仅供参考)
        if len(input_vals) == 7: # 如果是全指标
            est_score = self.score_reg.predict([input_vals])[0]
        else:
            # 如果是部分指标，取该类型的平均分作为参考
            # 这里的 self.raw_df 已经包含了 '综合得分' 列，不会再报错了
            est_score = self.raw_df[self.raw_df['样品类型']==pred_type]['综合得分'].mean()
            
        # 4. 计算特征相对水平 (用于画图)
        # 对比该类型的平均值
        type_means = self.raw_df[self.raw_df['样品类型']==pred_type][feat_names].mean()
        # 偏差百分比
        deviations = (np.array(input_vals) - type_means) / type_means
        
        return pred_type, confidence, est_score, deviations, probs

# ==================== 4. 辅助函数 ====================
def plot_probs(classes, probs):
    """绘制概率分布条形图"""
    fig, ax = plt.subplots(figsize=(6, 3))
    # 颜色：最高概率用深蓝，其他用灰
    colors = ['#2980B9' if p == max(probs) else '#BDC3C7' for p in probs]
    ax.barh(classes, probs, color=colors)
    ax.set_xlim(0, 1.1)
    ax.set_title("AI 判别置信度 (Probability)")
    
    # 加上数字标签
    for i, v in enumerate(probs):
        ax.text(v + 0.02, i, f"{v:.1%}", va='center', fontsize=9)
        
    plt.tight_layout()
    return fig

# ==================== 5. 主程序 ====================

st.title("🔬 当归酒制品智能鉴别与评价系统")
st.markdown("Intelligent Identification System | **Classification & Evaluation**")

# 1. 加载数据
file_path = 'standard_data.xlsx'
if not os.path.exists(file_path):
    st.error("❌ 未找到数据文件 'standard_data.xlsx'！")
    st.info("请先运行数据生成脚本，或者上传您的 Excel 文件。")
    st.stop()

try:
    df = pd.read_excel(file_path)
    # 简单校验
    if '样品类型' not in df.columns:
        st.error("❌ 数据缺少【样品类型】列，无法训练鉴别模型。请检查 Excel 表头。")
        st.stop()
except Exception as e:
    st.error(f"读取数据失败: {e}")
    st.stop()

# 2. 训练模型 (缓存加速)
if 'model' not in st.session_state:
    with st.spinner('正在训练 AI 鉴别模型 (Random Forest)...'):
        st.session_state.model = IntelligentSystem(df)
model = st.session_state.model

# 3. 侧边栏模式选择
st.sidebar.header("🕹️ 鉴别模式选择")
mode = st.sidebar.radio(
    "请选择输入数据类型：",
    ("🔍 全指标精准鉴别 (理化数据)", 
     "⚡ Q-Marker 快速筛查 (关键指标)", 
     "👃 电子鼻无损鉴别 (8个传感器)")
)
st.sidebar.markdown("---")
st.sidebar.info(f"📚 知识库已加载：{len(df)} 个样本")

# 4. 主界面交互
col_input, col_result = st.columns([1, 1.5])
inputs, result_type = [], None

# --- 模式 A: 全指标 ---
if "全指标" in mode:
    with col_input:
        st.subheader("📝 理化数据录入")
        with st.form("f1"):
            for col in CHEM_COLS:
                # 默认填入平均值方便演示
                avg_val = float(df[col].mean())
                inputs.append(st.number_input(col, value=avg_val, format="%.3f"))
            if st.form_submit_button("开始鉴别"):
                result_type, conf, score, devs, probs = model.predict(model.full_clf, inputs, CHEM_COLS)
                feat_names = CHEM_COLS

# --- 模式 B: Q-Marker ---
elif "Q-Marker" in mode:
    with col_input:
        st.subheader("⚡ 关键标志物录入")
        st.caption("AI 已筛选出区分度最高的 3 个指标：")
        with st.form("f2"):
            for col in model.q_markers:
                avg_val = float(df[col].mean())
                inputs.append(st.number_input(col, value=avg_val, format="%.3f"))
            if st.form_submit_button("快速鉴别"):
                result_type, conf, score, devs, probs = model.predict(model.q_clf, inputs, model.q_markers)
                feat_names = model.q_markers

# --- 模式 C: 电子鼻 ---
elif "电子鼻" in mode:
    with col_input:
        st.subheader("👃 传感器数据录入")
        with st.form("f3"):
            c1, c2 = st.columns(2)
            # 动态生成 8 个输入框
            for i, col in enumerate(SENSOR_COLS):
                avg_val = float(df[col].mean())
                with c1 if i < 4 else c2:
                    inputs.append(st.number_input(col, value=avg_val, format="%.2f"))
            if st.form_submit_button("AI 智能识别"):
                result_type, conf, score, devs, probs = model.predict(model.enose_clf, inputs, SENSOR_COLS)
                feat_names = SENSOR_COLS

# 5. 结果展示
with col_result:
    st.subheader("📊 鉴定报告")
    st.markdown("---")
    
    if result_type:
        # 卡片式结果
        st.markdown(f"""
        <div class="result-card">
            <div style="color:#666; font-size:14px;">系统判定该样品的炮制工艺为：</div>
            <div class="prediction-text">🌿 {result_type}</div>
            <div style="margin-top:10px;">
                <span>AI 置信度：</span>
                <span style="font-weight:bold; color:#2980B9">{conf*100:.1f}%</span>
                <div style="width:100%; background:#e0e0e0; height:8px; border-radius:4px; margin-top:2px;">
                    <div style="width:{conf*100}%; background:#2980B9; height:8px; border-radius:4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 概率分布", "📋 质量参考"])
        
        with tab1:
            st.caption("AI 对该样品属于各分类的可能性预判：")
            fig = plot_probs(model.full_clf.classes_, probs)
            st.pyplot(fig)
            
        with tab2:
            st.caption("基于理化指标的辅助质量评分 (TOPSIS)：")
            c1, c2 = st.columns([1, 2])
            c1.metric("综合质量指数", f"{score:.4f}")
            if score > 0.6:
                c1.success("质量评价：较优")
            else:
                c1.warning("质量评价：一般")
            
            c2.markdown("**特征指纹偏差图** (vs 该类平均值)")
            chart_df = pd.DataFrame({"特征": feat_names, "偏差度": devs})
            st.bar_chart(chart_df, x="特征", y="偏差度", color="#2980B9", height=200)
            
    else:
        st.info("👈 请在左侧输入数据并点击按钮，AI 将为您鉴别样品类型。")