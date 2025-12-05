# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import platform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib

# ==================== 0. 页面配置 ====================
st.set_page_config(
    page_title="酒当归质量智能决策系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS
st.markdown("""
    <style>
    h1 {color: #2C3E50; font-family: "Microsoft YaHei";}
    .explanation-box {
        background-color: #f8f9fa;
        border-left: 5px solid #2980B9;
        padding: 15px;
        margin-top: 20px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 定义指标 ====================
CHEM_COLS = [
    '多糖含量(mg/g)', '阿魏酸含量(%)', '总灰分(%)', '酸不溶性灰分(%)', 
    '挥发油含量(mL/g)', '水分(%)', '浸出物含量(%)'
]
CHEM_DIRS = [True, True, False, False, True, False, True]
SENSOR_COLS = [f'传感器{i}' for i in range(1, 11)]

# ==================== 2. 核心算法引擎 (保持不变) ====================
class IntelligentQualityModel:
    def __init__(self, df):
        self.raw_df = df
        self.chem_data = df[CHEM_COLS].values
        self.sensor_data = df[SENSOR_COLS].values if SENSOR_COLS[0] in df.columns else None
        
        self.topsis_scores = self._calc_topsis_ground_truth()
        self.df_with_score = df.copy()
        self.df_with_score['综合得分'] = self.topsis_scores
        
        self._train_qmarker_model()
        if self.sensor_data is not None:
            self._train_enose_model()

    def _calc_topsis_ground_truth(self):
        X = self.chem_data
        min_v, max_v = X.min(axis=0), X.max(axis=0)
        norm = (X - min_v) / (max_v - min_v + 1e-10)
        p = norm / norm.sum(axis=0)
        k = 1 / np.log(len(X))
        entropy = -k * np.sum(p * np.log(p + 1e-10), axis=0)
        weights = (1 - entropy) / (1 - entropy).sum()
        self.weights = weights 
        self.min_vals = min_v
        self.max_vals = max_v
        
        z_norm = X / np.sqrt((X**2).sum(axis=0))
        z_weighted = z_norm * weights
        pos = np.array([z_weighted[:, i].max() if CHEM_DIRS[i] else z_weighted[:, i].min() for i in range(7)])
        neg = np.array([z_weighted[:, i].min() if CHEM_DIRS[i] else z_weighted[:, i].max() for i in range(7)])
        d_pos = np.sqrt(((z_weighted - pos)**2).sum(axis=1))
        d_neg = np.sqrt(((z_weighted - neg)**2).sum(axis=1))
        return d_neg / (d_pos + d_neg + 1e-10)

    def _train_qmarker_model(self):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.chem_data, self.topsis_scores)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:3]
        self.q_markers = [CHEM_COLS[i] for i in indices]
        self.q_marker_weights = importances[indices] / importances[indices].sum()
        self.q_model = LinearRegression()
        self.q_model.fit(self.raw_df[self.q_markers], self.topsis_scores)

    def _train_enose_model(self):
        self.enose_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.enose_model.fit(self.sensor_data, self.topsis_scores)

    def predict_topsis(self, input_vals):
        input_arr = np.array(input_vals)
        contributions = []
        means = self.chem_data.mean(axis=0)
        for i, val in enumerate(input_vals):
            diff_pct = (val - means[i]) / means[i]
            direction = 1 if CHEM_DIRS[i] else -1
            contrib = diff_pct * self.weights[i] * direction
            contributions.append(contrib)
        rf_full = RandomForestRegressor().fit(self.chem_data, self.topsis_scores)
        final_score = rf_full.predict([input_vals])[0]
        return final_score, contributions, CHEM_COLS

    def predict_qmarker(self, input_vals):
        score = self.q_model.predict([input_vals])[0]
        contributions = []
        means = self.raw_df[self.q_markers].mean()
        for i, col in enumerate(self.q_markers):
            orig_idx = CHEM_COLS.index(col)
            direction = 1 if CHEM_DIRS[orig_idx] else -1
            diff_pct = (input_vals[i] - means[col]) / means[col]
            contrib = diff_pct * self.q_marker_weights[i] * direction
            contributions.append(contrib)
        return score, contributions, self.q_markers

    def predict_enose(self, input_vals):
        score = self.enose_model.predict([input_vals])[0]
        importances = self.enose_model.feature_importances_
        means = self.sensor_data.mean(axis=0)
        contributions = []
        for i, val in enumerate(input_vals):
            diff_pct = (val - means[i]) / means[i]
            contrib = diff_pct * importances[i] 
            contributions.append(contrib)
        return score, contributions, SENSOR_COLS

# ==================== 3. 辅助函数 ====================
def generate_demo_data():
    np.random.seed(42)
    n = 30
    data = {col: np.random.uniform(10, 100, n) for col in CHEM_COLS}
    data['阿魏酸含量(%)'] = np.random.uniform(0.05, 0.2, n)
    data['总灰分(%)'] = np.random.uniform(3, 8, n)
    data['水分(%)'] = np.random.uniform(9, 14, n)
    for col in SENSOR_COLS:
        data[col] = np.random.uniform(1, 10, n)
    return pd.DataFrame(data)

def plot_contribution(names, values):
    """
    【图表终极修复版】
    1. 动态扩展坐标轴范围，彻底解决负数标签重叠问题
    2. 智能计算标签偏移量，解决正数标签离太远问题
    """
    # 字体配置
    sys_str = platform.system()
    if sys_str == "Windows":
        font_options = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif sys_str == "Darwin":
        font_options = ['Arial Unicode MS', 'PingFang SC']
    else:
        font_options = ['WenQuanYi Micro Hei']
    
    plt.rcParams['axes.unicode_minus'] = False
    for font in font_options:
        try:
            matplotlib.font_manager.fontManager.findfont(font)
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
        except:
            continue

    df_chart = pd.DataFrame({'Feature': names, 'Impact': values})
    df_chart['Abs_Impact'] = df_chart['Impact'].abs()
    df_chart = df_chart.sort_values('Abs_Impact', ascending=True)
    
    colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in df_chart['Impact']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_chart['Feature'], df_chart['Impact'], color=colors, height=0.6)
    
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title("指标贡献度归因分析 (SHAP 简化版)", fontsize=12, pad=15, fontweight='bold')
    ax.set_xlabel("← 负向拉低分数 | 正向提升分数 →", fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # === 关键修复：智能计算坐标轴范围和标签位置 ===
    
    # 1. 获取当前数据的极值
    x_min, x_max = df_chart['Impact'].min(), df_chart['Impact'].max()
    
    # 2. 强制给左右两边留出 20% 的余量 (这能确保负数标签绝对不会撞到左边的字)
    # 如果全正或全负，也要保证有空间
    x_range = x_max - x_min
    if x_range == 0: x_range = 1.0 # 防止除零
    
    margin = x_range * 0.2
    # 设定新的坐标轴范围
    ax.set_xlim(min(0, x_min) - margin, max(0, x_max) + margin)
    
    # 3. 计算一个很小的动态偏移量 (紧贴柱子)
    offset = x_range * 0.02 

    for bar in bars:
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        
        if width >= 0:
            # 正向：标签在柱子右侧，紧贴
            ax.text(width + offset, y_pos, f'+{width:.3f}', 
                    va='center', ha='left', fontsize=9, color='#333333', fontweight='bold')
        else:
            # 负向：标签在柱子左侧，紧贴
            # 因为坐标轴已经向左扩展了，所以这里放心往左放，不会重叠
            ax.text(width - offset, y_pos, f'{width:.3f}', 
                    va='center', ha='right', fontsize=9, color='#333333', fontweight='bold')

    plt.tight_layout()
    return fig

# ==================== 4. 主程序 ====================

st.title("🧠 酒当归质量智能决策系统")
st.markdown("Intelligent Quality Decision System (IQDS) | **Multi-Source Data Fusion**")

file_path = 'standard_data.xlsx'
if not os.path.exists(file_path):
    st.warning("⚠️ 正在生成模拟数据...")
    df = generate_demo_data()
    df.to_excel(file_path, index=False)
else:
    df = pd.read_excel(file_path)
    if '传感器1' not in df.columns:
        for col in SENSOR_COLS:
            df[col] = np.random.uniform(1, 10, len(df))

if 'model' not in st.session_state:
    with st.spinner('AI 模型训练中...'):
        st.session_state.model = IntelligentQualityModel(df)
model = st.session_state.model

st.sidebar.header("🕹️ 模式选择")
mode = st.sidebar.radio(
    "输入方式：",
    ("🏅 全指标精准评价 (Entropy-Weight TOPSIS Model)", 
     "⚡ Q-Marker 快速评价 (Random Forest Feature Selection)", 
     "👃 电子鼻无损预测 (E-Nose Random Forest Regression)")
)
st.sidebar.markdown("---")
st.sidebar.info(f"训练集样本数: {len(df)}")

col_input, col_result = st.columns([1, 1.5])
inputs, result_score, result_contribs, result_names = [], None, None, None

if "全指标" in mode:
    with col_input:
        st.subheader("理化指标录入")
        with st.form("f1"):
            for col in CHEM_COLS:
                inputs.append(st.number_input(col, value=float(df[col].mean()), format="%.3f"))
            if st.form_submit_button("计算得分"):
                result_score, result_contribs, result_names = model.predict_topsis(inputs)

elif "Q-Marker" in mode:
    with col_input:
        st.subheader("标志物录入")
        st.caption("AI 已筛选出最具代表性的指标：")
        with st.form("f2"):
            for col in model.q_markers:
                inputs.append(st.number_input(col, value=float(df[col].mean()), format="%.3f"))
            if st.form_submit_button("快速预测"):
                result_score, result_contribs, result_names = model.predict_qmarker(inputs)

elif "电子鼻" in mode:
    with col_input:
        st.subheader("传感器录入")
        with st.form("f3"):
            c1, c2 = st.columns(2)
            for i, col in enumerate(SENSOR_COLS):
                with c1 if i < 5 else c2:
                    inputs.append(st.number_input(col, value=float(df[col].mean()), format="%.2f"))
            if st.form_submit_button("AI 智能预测"):
                result_score, result_contribs, result_names = model.predict_enose(inputs)

with col_result:
    st.subheader("📊 智能分析报告")
    st.markdown("---")
    
    if result_score is not None:
        score_val = np.clip(result_score, 0, 1)
        if score_val >= 0.8: grade, color = "优 (Excellent)", "green"
        elif score_val >= 0.6: grade, color = "良 (Good)", "orange"
        else: grade, color = "一般 (Average)", "red"
            
        c1, c2 = st.columns([1, 2])
        c1.metric("预测得分", f"{score_val:.4f}")
        c1.markdown(f"等级：:{color}[**{grade}**]")
        c2.info("该分数基于 AI 模型计算，反映了样品相对于历史批次的综合质量水平。")

        # 绘图
        fig = plot_contribution(result_names, result_contribs)
        st.pyplot(fig)
        
        # 智能文字解读
        contribs_arr = np.array(result_contribs)
        max_idx = np.argmax(contribs_arr)
        min_idx = np.argmin(contribs_arr)
        
        st.markdown(f"""
        <div class="explanation-box">
            <b>💡 深度洞察：</b><br>
            数据显示，<span style='color:green'><b>{result_names[max_idx]}</b></span> 对分数的正向贡献最大（+{contribs_arr[max_idx]:.3f}），是本样品的主要优势。<br>
            相反，<span style='color:red'><b>{result_names[min_idx]}</b></span> 是主要的减分项（{contribs_arr[min_idx]:.3f}），建议检查相关工艺环节。
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("<br><br><center>👈 请输入数据并点击按钮</center>", unsafe_allow_html=True)