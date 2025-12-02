# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os

# ==================== 0. 页面配置 ====================
st.set_page_config(
    page_title="酒当归质量智能评价系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义CSS：保留了您喜欢的清爽风格
st.markdown("""
    <style>
    /* 标题样式 */
    h1 {color: #2C3E50; font-family: "Microsoft YaHei";}
    
    /* 侧边栏背景 */
    [data-testid="stSidebar"] {background-color: #F8F9FA;}
    
    /* 结果卡片 */
    div[data-testid="metric-container"] {
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
    }
    
    /* 隐藏表格索引列 */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 定义指标与标准 ====================
FEATURE_COLS = [
    '多糖含量(mg/g)', '阿魏酸含量(%)', '总灰分(%)', '酸不溶性灰分(%)', 
    '挥发油含量(mL/g)', '水分(%)', '浸出物含量(%)'
]

# True=正向(越大越好), False=负向(越小越好)
FEATURE_DIRECTIONS = [True, True, False, False, True, False, True]

# ==================== 2. 核心算法模型 ====================
class QualityModel:
    def __init__(self, ref_df):
        """初始化模型"""
        # 保留 DataFrame 用于展示
        self.ref_df_display = ref_df[FEATURE_COLS].copy() 
        
        self.ref_data = ref_df[FEATURE_COLS].values
        self.stats_mean = ref_df[FEATURE_COLS].mean()
        
        # --- 算法逻辑 ---
        X = self.ref_data
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1e-10
        
        norm_entropy = (X - min_vals) / ranges
        p_matrix = norm_entropy / np.sum(norm_entropy, axis=0, keepdims=True)
        p_matrix[p_matrix < 1e-10] = 1e-10
        k = 1 / np.log(len(X))
        entropy = -k * np.sum(p_matrix * np.log(p_matrix), axis=0)
        diversity = 1 - entropy
        self.weights = diversity / np.sum(diversity)
        
        self.norm_scale = np.sqrt(np.sum(X**2, axis=0))
        self.norm_scale[self.norm_scale == 0] = 1e-10
        
        # 确定正负理想解
        weighted_X = (X / self.norm_scale) * self.weights
        self.pos_ideal = np.zeros(len(FEATURE_COLS))
        self.neg_ideal = np.zeros(len(FEATURE_COLS))
        
        for j in range(len(FEATURE_COLS)):
            if FEATURE_DIRECTIONS[j]:
                self.pos_ideal[j] = np.max(weighted_X[:, j])
                self.neg_ideal[j] = np.min(weighted_X[:, j])
            else:
                self.pos_ideal[j] = np.min(weighted_X[:, j])
                self.neg_ideal[j] = np.max(weighted_X[:, j])
                
        # 计算分级阈值
        ref_scores = self._calc_topsis_score(X)
        self.mu = np.mean(ref_scores)
        self.sigma = np.std(ref_scores)
        self.limit_high = min(self.mu + 0.8 * self.sigma, 0.98)
        self.limit_low = max(self.mu - 0.8 * self.sigma, 0.1)

    def _calc_topsis_score(self, data_matrix):
        norm = data_matrix / self.norm_scale
        weighted = norm * self.weights
        d_pos = np.sqrt(np.sum((weighted - self.pos_ideal)**2, axis=1))
        d_neg = np.sqrt(np.sum((weighted - self.neg_ideal)**2, axis=1))
        return d_neg / (d_pos + d_neg + 1e-10)

    def predict(self, input_values):
        X_new = np.array(input_values).reshape(1, -1)
        score = self._calc_topsis_score(X_new)[0]
        if score >= self.limit_high:
            grade = "一等品 (优)"
            css_color = "success"
        elif score >= self.limit_low:
            grade = "二等品 (良)"
            css_color = "warning"
        else:
            grade = "三等品 (合格)"
            css_color = "error"
        return score, grade, css_color

@st.cache_resource
def load_system():
    file_path = 'standard_data.xlsx'
    if not os.path.exists(file_path):
        return None, "未找到 standard_data.xlsx"
    try:
        df = pd.read_excel(file_path)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            return None, f"Excel 表头缺失: {missing}"
        model = QualityModel(df)
        return model, None
    except Exception as e:
        return None, f"读取数据出错: {str(e)}"

# ==================== 3. 界面主逻辑 ====================

# 侧边栏：输入区 (保留了您喜欢的范围提示)
with st.sidebar:
    st.header("📝 样品数据录入")
    st.caption("请依据检测报告录入数据：")
    st.markdown("---")
    
    input_form = st.form("sample_form")
    user_inputs = {}
    
    with input_form:
        st.subheader("🧪 化学成分")
        # 多糖
        user_inputs['多糖含量(mg/g)'] = st.number_input(
            "多糖含量 (mg/g)", min_value=0.0, max_value=300.0, format="%.2f",
            help="参考范围：40 ~ 150 mg/g"
        )
        # 阿魏酸
        user_inputs['阿魏酸含量(%)'] = st.number_input(
            "阿魏酸含量 (%)", min_value=0.0, max_value=5.0, step=0.001, format="%.3f",
            help="药典标准：不得少于 0.050%"
        )
        # 挥发油
        user_inputs['挥发油含量(mL/g)'] = st.number_input(
            "挥发油 (mL/g)", min_value=0.0, max_value=5.0, format="%.2f",
            help="参考范围：0.3 ~ 1.0 mL/g"
        )
        # 浸出物
        user_inputs['浸出物含量(%)'] = st.number_input(
            "浸出物 (%)", min_value=0.0, max_value=100.0, format="%.2f",
            help="药典标准：不得少于 45.0%"
        )

        st.subheader("⚖️ 物理性质")
        # 水分
        user_inputs['水分(%)'] = st.number_input(
            "水分 (%)", min_value=0.0, max_value=20.0, format="%.2f",
            help="药典标准：不得过 13.0%"
        )
        # 灰分
        user_inputs['总灰分(%)'] = st.number_input(
            "总灰分 (%)", min_value=0.0, max_value=20.0, format="%.2f",
            help="药典标准：不得过 7.0%"
        )
        # 酸不溶灰分
        user_inputs['酸不溶性灰分(%)'] = st.number_input(
            "酸不溶灰分 (%)", min_value=0.0, max_value=10.0, format="%.2f",
            help="药典标准：不得过 2.0%"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("开始评价", use_container_width=True)

# 主界面
model, error_msg = load_system()

if error_msg:
    st.error(error_msg)
    st.stop()

# 顶部标题
st.title("🔬 酒当归质量智能评价系统")
st.markdown("Quality Evaluation System for *Angelica sinensis* (Wine-processed)")
st.divider()

if not submitted:
    st.info("👈 请在左侧侧边栏输入数据，点击“开始评价”。")
    
    st.subheader("📊 历史批次标准库概况")
    # 【修复】这里去掉了 .style.background_gradient，解决了报错问题
    # 直接显示朴素的 DataFrame，或者使用 Streamlit 原生的高亮
    st.dataframe(
        model.ref_df_display.head(5), 
        use_container_width=True
    )
    st.caption(f"当前系统已加载 {len(model.ref_df_display)} 个标准批次数据作为评价基准。")

else:
    # 整理输入
    input_values = [user_inputs[col] for col in FEATURE_COLS]
    
    # 药典红线检查
    reject_reasons = []
    if user_inputs['水分(%)'] > 13.0: 
        reject_reasons.append(f"水分超标 (实测 {user_inputs['水分(%)']}% > 限度 13.0%)")
    if user_inputs['阿魏酸含量(%)'] < 0.050:
        reject_reasons.append(f"阿魏酸含量不足 (实测 {user_inputs['阿魏酸含量(%)']}% < 限度 0.050%)")
    if user_inputs['总灰分(%)'] > 7.0:
        reject_reasons.append(f"总灰分超标 (实测 {user_inputs['总灰分(%)']}% > 限度 7.0%)")

    # 显示结果
    if reject_reasons:
        st.error("❌ **该样品判定为：不合格**")
        for r in reject_reasons:
            st.markdown(f"- {r}")
    else:
        score, grade, color = model.predict(input_values)
        
        col_res1, col_res2 = st.columns([1, 1.5])
        
        with col_res1:
            st.subheader("评价结论")
            st.metric(label="综合质量得分", value=f"{score:.4f}")
            
            if color == "success":
                st.success(f"🎉 **{grade}**")
            elif color == "warning":
                st.warning(f"⚠️ **{grade}**")
            else:
                st.error(f"📉 **{grade}**")
                
        with col_res2:
            st.subheader("指标偏差分析")
            # 【还原】这里改回了最初版本的蓝色偏差图，简单直观
            safe_means = model.stats_mean.copy()
            safe_means[safe_means==0] = 1
            
            # 计算偏差百分比：(实测值 - 平均值) / 平均值
            pct_changes = (np.array(input_values) - model.stats_mean) / safe_means * 100
            
            chart_data = pd.DataFrame({
                "指标": FEATURE_COLS,
                "相对于平均值的偏差 (%)": pct_changes
            })
            
            st.bar_chart(
                chart_data, 
                x="指标", 
                y="相对于平均值的偏差 (%)",
                color="#2980B9", # 经典的科研蓝
                height=300
            )
            st.caption("注：0刻度线代表历史标准批次的平均水平。")