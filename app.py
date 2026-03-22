"""
Leptospirosis Risk Prediction Dashboard
MSC Research - CMM799
"""

## Import libraries
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import math
from pathlib import Path

## Load styles
def load_css(filepath):
    css = Path(filepath).read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("assets/styles.css")

## Page config
st.set_page_config(
    page_title="Leptospirosis Risk Prediction - Sri Lanka",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

## Model Paths
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
FIG_DIR    = os.path.join(BASE_DIR, "scripts", "03_Data_Modeling")



## Dual UI Helper
def sync_state(from_key, to_key):
    st.session_state[to_key] = st.session_state[from_key]

def dual_input(sb, label, min_val, max_val, default_val, step, key):
    if f"{key}_s" not in st.session_state:
        st.session_state[f"{key}_s"] = default_val
    if f"{key}_n" not in st.session_state:
        st.session_state[f"{key}_n"] = default_val
    
    sb.markdown(f"<div style='font-size:0.82rem; color:#8b949e; margin-bottom: 5px;'>{label}</div>", unsafe_allow_html=True)
    c1, c2 = sb.columns([3, 1])
    with c1:
        c1.slider(label, min_val, max_val, step=step, key=f"{key}_s", 
                  on_change=sync_state, args=(f"{key}_s", f"{key}_n"), label_visibility="collapsed")
    with c2:
        c2.number_input(label, min_val, max_val, step=step, key=f"{key}_n", 
                        on_change=sync_state, args=(f"{key}_n", f"{key}_s"), label_visibility="collapsed")
    return st.session_state[f"{key}_s"]



## Load Artifacts
@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    le       = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    drr      = pd.read_csv(os.path.join(MODEL_DIR, "district_risk_rate.csv"),
                           index_col=0, header=0)
    drr.index.name = "District"
    return model, le, drr

model, le, district_risk_rate = load_artifacts()
DISTRICTS = sorted(le.classes_.tolist())
THRESHOLD  = 1/3 ## Leaned to high risk
MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",
               5:"May",6:"June",7:"July",8:"August",
               9:"September",10:"October",11:"November",12:"December"}


st.markdown("""
<div class="hero-banner">
    <h1>🐀🌡️ Leptospirosis Risk Prediction Dashboard</h1>
</div>
""", unsafe_allow_html=True)

## Top Level Cards (Aggregated Counts)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown('<div class="metric-card"><div class="label">Districts Monitored</div><div class="value">25</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="metric-card"><div class="label">Final Model</div><div class="value" style="font-size:2rem">Random Forest</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="metric-card"><div class="label">Test AUC</div><div class="value">0.892</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown('<div class="metric-card"><div class="label">Decision Threshold</div><div class="value">1/3</div></div>', unsafe_allow_html=True)
with kpi5:
    st.markdown('<div class="metric-card"><div class="label">Features Used</div><div class="value">51</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

## Tabs
tab_pred, tab_perf, tab_shap, tab_about = st.tabs([
    "🎯  Risk Predictor",
    "⚡  Model Performance",
    "💡  Feature Insights",
    "📖  About the Research",
])


# ============================================================================
# TAB 1 - PREDICTION
# ============================================================================
with tab_pred:
    sb = st.sidebar
    sb.markdown("## ⚙️ Input Parameters")
    sb.markdown("---")

    #  District & Time 
    sb.markdown("### 📍 Location & Month")
    district = sb.selectbox("District", DISTRICTS, index=DISTRICTS.index("Colombo")) ##default to colombo
    month    = sb.selectbox("Month", list(MONTH_NAMES.keys()),
                             format_func=lambda m: MONTH_NAMES[m], index=0) # and Jan

    # Derived from district / month (uploaded a .csv with these values)
    dist_risk = float(district_risk_rate.loc[district].values[0]) if district in district_risk_rate.index else 0.48
    dist_enc  = int(le.transform([district])[0]) ## encoder
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    # Climate inputs
    sb.markdown("### 🌧️ Climate")
    precip   = dual_input(sb, "Precipitation (mm)",   0.0,  820.0, 156.0, 1.0, "c_precip")
    temp_min = dual_input(sb, "Min Temperature (°C)", 5.0,   28.0,  22.0, 0.1, "c_tmin")
    temp_max = dual_input(sb, "Max Temperature (°C)", 19.0,  40.0,  31.4, 0.1, "c_tmax")
    soil_moist = dual_input(sb, "Soil Moisture (m³/m³)", 0.0, 0.51, 0.33, 0.01, "c_soil")##(0 to 7 cm soil layer)

    mean_temp  = (temp_min + temp_max) / 2
    temp_range = temp_max - temp_min

    # Lag features
    sb.markdown("### 🕑 Lag / History")
    hr_lag1  = sb.select_slider("High-Risk last month?",   options=[0, 1], value=0)
    hr_lag2  = sb.select_slider("High-Risk 2 months ago?", options=[0, 1], value=0)
    hr_roll3 = dual_input(sb, "3-month High-Risk rate (0-1)", 0.0, 1.0, 0.33, 0.3333, "c_hr3")

    # Previous months precip / temp for lags (use same value as a simple default)
    sb.markdown("### 🕑 Lagged Climate (prev months)")
    precip_lag1 = dual_input(sb, "Precipitation lag-1 (mm)", 0.0, 820.0, precip, 1.0, "l_precip1")
    precip_lag2 = dual_input(sb, "Precipitation lag-2 (mm)", 0.0, 820.0, precip, 1.0, "l_precip2")
    precip_lag3 = dual_input(sb, "Precipitation lag-3 (mm)", 0.0, 820.0, precip, 1.0, "l_precip3")

    tmin_lag1 = dual_input(sb, "Temp Min lag-1 (°C)", 5.0, 28.0, temp_min, 0.1, "l_tmin1")
    tmin_lag2 = dual_input(sb, "Temp Min lag-2 (°C)", 5.0, 28.0, temp_min, 0.1, "l_tmin2")
    tmin_lag3 = dual_input(sb, "Temp Min lag-3 (°C)", 5.0, 28.0, temp_min, 0.1, "l_tmin3")

    tmax_lag1 = dual_input(sb, "Temp Max lag-1 (°C)", 19.0, 40.0, temp_max, 0.1, "l_tmax1")
    tmax_lag2 = dual_input(sb, "Temp Max lag-2 (°C)", 19.0, 40.0, temp_max, 0.1, "l_tmax2")
    tmax_lag3 = dual_input(sb, "Temp Max lag-3 (°C)", 19.0, 40.0, temp_max, 0.1, "l_tmax3")

    soil_lag1 = dual_input(sb, "Soil Moist lag-1", 0.0, 0.51, soil_moist, 0.01, "l_soil1")
    soil_lag2 = dual_input(sb, "Soil Moist lag-2", 0.0, 0.51, soil_moist, 0.01, "l_soil2")
    soil_lag3 = dual_input(sb, "Soil Moist lag-3", 0.0, 0.51, soil_moist, 0.01, "l_soil3")

    # Agricultural / Socio-economic
    sb.markdown("### 🌾 Agriculture & Population")
    mrice_area  = sb.number_input("Maha Rice Area (ha)",    0, 120000, 26544)
    mrice_yield = sb.number_input("Maha Rice Yield (kg/ha)",0,   7000,  3998)
    srice_area  = sb.number_input("Yala Rice Area (ha)",    0,  70000, 14912)
    srice_yield = sb.number_input("Yala Rice Yield (kg/ha)",0,   7000,  3699)
    population  = sb.number_input("Population",          80000,2400000, 823742)
    households  = sb.number_input("Households",          19000, 660000, 213132)

    pop_per_hh = population / max(households, 1)

    # BioClim 
    sb.markdown("### 🌍 Bio Climatic Variables")
    BIO2  = dual_input(sb, "BIO2  - Diurnal Range (°C)",       4.0, 20.0, 9.0,  0.1, "b_2")
    BIO3  = dual_input(sb, "BIO3  - Isothermality",            0.2,  0.7, 0.42, 0.01, "b_3")
    BIO4  = dual_input(sb, "BIO4  - Temp Seasonality",         0.0,400.0, 90.0, 1.0, "b_4")
    BIO13 = dual_input(sb, "BIO13 - Precip Wettest Month (mm)",0.0,500.0,230.0, 1.0, "b_13")
    BIO14 = dual_input(sb, "BIO14 - Precip Driest Month (mm)", 0.0, 60.0,  8.0, 0.5, "b_14")
    BIO15 = dual_input(sb, "BIO15 - Precip Seasonality",       0.0,160.0, 80.0, 1.0, "b_15")
    BIO16 = dual_input(sb, "BIO16 - Precip Wettest Qtr (mm)",  0.0,1200.0,570.0,5.0, "b_16")
    BIO17 = dual_input(sb, "BIO17 - Precip Driest Qtr (mm)",   0.0, 200.0, 30.0,1.0, "b_17")
    BIO18 = dual_input(sb, "BIO18 - Precip Warmest Qtr (mm)",  0.0, 700.0,350.0,5.0, "b_18")
    BIO19 = dual_input(sb, "BIO19 - Precip Coldest Qtr (mm)",  0.0, 600.0,140.0,5.0, "b_19")

    # Rolling precip / soil
    precip_roll3 = (precip + precip_lag1 + precip_lag2) / 3
    precip_roll6 = (precip + precip_lag1 + precip_lag2 + precip_lag3) / 4   # approx
    soil_roll3   = (soil_moist + soil_lag1 + soil_lag2) / 3
    soil_roll6   = (soil_moist + soil_lag1 + soil_lag2 + soil_lag3) / 4

    #  Composite features 
    mrice_x_precip = mrice_area * precip
    srice_x_precip = srice_area * precip
    precip_vs_norm = precip / 156.0          # normalised to training mean
    precip_anomaly = precip - 156.0
    waterlog_index = precip * soil_moist
    heat_moisture  = mean_temp * soil_moist

    #  Build input DataFrame 
    input_data = pd.DataFrame([{
        "Month":                         month,
        "Precipitation_mm":              precip,
        "Temp_Min_C":                    temp_min,
        "Temp_Max_C":                    temp_max,
        "Soil_Moisture_0_7cm":           soil_moist,
        "MRiceArea":                     mrice_area,
        "MRiceYield":                    mrice_yield,
        "SRiceArea":                     srice_area,
        "SRiceYield":                    srice_yield,
        "Population":                    population,
        "Households":                    households,
        "mean_temp":                     mean_temp,
        "BIO2":                          BIO2,
        "BIO3":                          BIO3,
        "BIO4":                          BIO4,
        "BIO13":                         BIO13,
        "BIO14":                         BIO14,
        "BIO15":                         BIO15,
        "BIO16":                         BIO16,
        "BIO17":                         BIO17,
        "BIO18":                         BIO18,
        "BIO19":                         BIO19,
        "Precipitation_mm_lag1":         precip_lag1,
        "Precipitation_mm_lag2":         precip_lag2,
        "Precipitation_mm_lag3":         precip_lag3,
        "Temp_Min_C_lag1":               tmin_lag1,
        "Temp_Min_C_lag2":               tmin_lag2,
        "Temp_Min_C_lag3":               tmin_lag3,
        "Temp_Max_C_lag1":               tmax_lag1,
        "Temp_Max_C_lag2":               tmax_lag2,
        "Temp_Max_C_lag3":               tmax_lag3,
        "Soil_Moisture_0_7cm_lag1":      soil_lag1,
        "Soil_Moisture_0_7cm_lag2":      soil_lag2,
        "Soil_Moisture_0_7cm_lag3":      soil_lag3,
        "Precipitation_mm_roll3_mean":   precip_roll3,
        "Precipitation_mm_roll6_mean":   precip_roll6,
        "Soil_Moisture_0_7cm_roll3_mean":soil_roll3,
        "Soil_Moisture_0_7cm_roll6_mean":soil_roll6,
        "Month_sin":                     month_sin,
        "Month_cos":                     month_cos,
        "MRice_x_Precip":                mrice_x_precip,
        "SRice_x_Precip":                srice_x_precip,
        "Temp_Range":                    temp_range,
        "Pop_per_Household":             pop_per_hh,
        "Precip_vs_norm":                precip_vs_norm,
        "Precip_anomaly":                precip_anomaly,
        "Waterlog_index":                waterlog_index,
        "Heat_Moisture":                 heat_moisture,
        "HR_lag1":                       hr_lag1,
        "HR_lag2":                       hr_lag2,
        "HR_roll3":                      hr_roll3,
        "District_enc":                  dist_enc,
        "District_risk_rate":            dist_risk,
    }])

    #  Predict 
    # Just predict() gives 50/50 chance prediction (cant use custom surveillance prob)
    # Use _proba ==>this gives the exact raw mathematical probability that the instance belongs to Class 0 and Class 1
    # Eg: the raw output array would look something like this: [[0.53, 0.47]] (This means: 53% chance of being Low Risk (0), 
    # and 47% chance of being High Risk (1)).
    prob  = model.predict_proba(input_data)[0, 1]
    label = "High Risk" if prob >= THRESHOLD else "Low Risk"
    pct   = prob * 100

    # Main panel
    col_res, col_context = st.columns([1, 1], gap="large")

    with col_res:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
        css_class = "risk-high" if label == "High Risk" else "risk-low"
        icon = "⚠️" if label == "High Risk" else "✅"
        st.markdown(f"""
        <div class="{css_class}">
            <h2>{icon} {label}</h2>
            <p><strong>{district}</strong> - {MONTH_NAMES[month]}</p>
            <p style="margin-top:12px; font-size:2.2rem; font-weight:700; color:#e6edf3">{pct:.1f}%</p>
            <p style="color:#8b949e; font-size:0.88rem">Estimated probability of High-Risk incidence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:12px;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e; margin-bottom:4px;">
                <span>0%</span><span>Threshold {THRESHOLD*100:.0f}%</span><span>100%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill" style="
                    width:{pct:.1f}%;
                    background:{'linear-gradient(90deg,#d1242f,#f85149)' if label=='High Risk' else 'linear-gradient(90deg,#238636,#3fb950)'};
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Threshold marker
        st.markdown(f"""
        <div style="position:relative; height:8px; margin-top:2px;">
            <div style="
                position:absolute; left:{THRESHOLD*100:.1f}%;
                width:2px; height:18px; background:#f0ad4e; top:-5px; border-radius:2px;
            "></div>
        </div>
        """, unsafe_allow_html=True)

    with col_context:
        st.markdown('<div class="section-header">Context</div>', unsafe_allow_html=True)

        # District historical risk
        hist_risk_pct = dist_risk * 100
        dr_color = "#f85149" if hist_risk_pct > 50 else "#3fb950"

        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px;">
            <p style="color:#8b949e; font-size:0.8rem; margin:0 0 4px; text-transform:uppercase; letter-spacing:.06em">Historical High-Risk Rate</p>
            <p style="font-size:1.8rem; font-weight:700; color:{dr_color}; margin:0">{hist_risk_pct:.1f}%</p>
            <p style="color:#8b949e; font-size:0.82rem; margin:4px 0 0">Long-run proportion of months that <strong>{district}</strong> was High-Risk (training data)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Key inputs summary table
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px;">
            <p style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; letter-spacing:.06em; margin:0 0 10px">Key Input Summary</p>
            <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                <tr><td style="color:#8b949e; padding:3px 0">Precipitation</td><td style="text-align:right; color:#e6edf3"><strong>{precip:.0f} mm</strong></td></tr>
                <tr><td style="color:#8b949e; padding:3px 0">Temperature (Min/Max)</td><td style="text-align:right; color:#e6edf3"><strong>{temp_min:.1f} / {temp_max:.1f} °C</strong></td></tr>
                <tr><td style="color:#8b949e; padding:3px 0">Soil Moisture</td><td style="text-align:right; color:#e6edf3"><strong>{soil_moist:.2f} m³/m³</strong></td></tr>
                <tr><td style="color:#8b949e; padding:3px 0">HR last month</td><td style="text-align:right; color:#e6edf3"><strong>{'Yes' if hr_lag1 else 'No'}</strong></td></tr>
                <tr><td style="color:#8b949e; padding:3px 0">3-month HR rate</td><td style="text-align:right; color:#e6edf3"><strong>{hr_roll3:.2f}</strong></td></tr>
                <tr><td style="color:#8b949e; padding:3px 0">Waterlog Index</td><td style="text-align:right; color:#e6edf3"><strong>{waterlog_index:.1f}</strong></td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    #  What does this mean
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)

    if label == "High Risk":
        st.warning(f"""
        **⚠️ High-Risk Alert** - The model predicts a **{pct:.1f}%** probability of elevated leptospirosis
        incidence for **{district}** in **{MONTH_NAMES[month]}**.

        A decision threshold of **{THRESHOLD:.0%}** is applied (cost-ratio FN:FP = 2:1), meaning the
        model prioritises sensitivity to avoid missing true outbreak months.

        **Recommended actions:**  Enhanced surveillance, pre-emptive public health messaging,
        increased rodent control activities, and targeted awareness campaigns in flood-prone areas.
        """)
    else:
        st.success(f"""
        **✅ Low-Risk Forecast** - The estimated probability ({pct:.1f}%) is below the
        **{THRESHOLD:.0%}** decision threshold.

        Routine surveillance is sufficient. Review again if precipitation or soil moisture increases
        significantly or if a neighbouring district reports a high-risk month.
        """)

    #  Raw input preview
    with st.expander("🔎 View raw feature vector sent to the model"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), width="stretch")


# ============================================================================
# TAB 2 - MODEL PERFORMANCE
# ============================================================================
with tab_perf:
    st.markdown('<div class="section-header">Model Selection - AUC Comparison</div>', unsafe_allow_html=True)

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        fig1 = os.path.join(FIG_DIR, "fig1_roc_pr.png")
        if os.path.exists(fig1):
            st.image(fig1, caption="Figure 1 - ROC & Precision-Recall curves for all candidate models",
                     width="stretch")
        else:
            st.info("fig1_roc_pr.png not found.")

    with img_col2:
        fig2 = os.path.join(FIG_DIR, "fig2_model_comparison.png")
        if os.path.exists(fig2):
            st.image(fig2, caption="Figure 2 - AUC comparison across selected five models",
                     width="stretch")
        else:
            st.info("fig2_model_comparison.png not found.")

    st.markdown('<div class="section-header">Statistical Significance - DeLong Test</div>', unsafe_allow_html=True)
    fig3 = os.path.join(FIG_DIR, "fig3_delong.png")
    if os.path.exists(fig3):
        st.image(fig3, caption="Figure 3 - DeLong pairwise AUC significance test",
                 width="stretch")
    else:
        st.info("fig3_delong.png not found.")

    st.markdown('<div class="section-header">Confusion Matrix & Metrics</div>', unsafe_allow_html=True)
    fig4 = os.path.join(FIG_DIR, "fig4_confusion.png")
    if os.path.exists(fig4):
        st.image(fig4, caption="Figure 4 - Confusion matrix and per-metric summary at threshold 0.25",
                 width="stretch")
    else:
        st.info("fig4_confusion.png not found.")

    st.markdown('<div class="section-header">District-Level Performance</div>', unsafe_allow_html=True)
    fig5 = os.path.join(FIG_DIR, "fig5_district.png")
    if os.path.exists(fig5):
        st.image(fig5, caption="Figure 5 - Accuracy and High-Risk recall by district",
                 width="stretch")
    else:
        st.info("fig5_district.png not found.")

    st.markdown('<div class="section-header">Generalisation - Learning Curves</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        fig7a = os.path.join(FIG_DIR, "fig7a_lc_all_folds.png")
        if os.path.exists(fig7a):
            st.image(fig7a, caption="Figure 7a - Learning curve (all folds)", width="stretch")
    with lc2:
        fig7b = os.path.join(FIG_DIR, "fig7b_lc_reliable_folds.png")
        if os.path.exists(fig7b):
            st.image(fig7b, caption="Figure 7b - Learning curve (reliable folds)", width="stretch")

    # Performance summary table
    st.markdown('<div class="section-header">Performance Metrics Summary Table</div>', unsafe_allow_html=True)
    perf_df = pd.DataFrame({
        "Metric"  : ["ROC-AUC", "Avg Precision (PR-AUC)", "Sensitivity (HR Recall)", "Specificity",
                     "PPV (HR Precision)", "F1 - High Risk", "Decision Threshold"],
        "Value"   : ["0.892", "0.894", "~0.83", "~0.80", "~0.78", "~0.80", "0.33 (FN:FP = 2:1)"],
        "Notes"   : ["Best across 5 candidate models", "High relevance for imbalanced classes",
                     "Achieved via cost-sensitive threshold", "Acceptable for surveillance context",
                     "Satisfactory for screening tool", "Balanced precision-recall at chosen threshold",
                     "Minimises missed high-risk months"],
    })
    st.dataframe(perf_df, width="stretch", hide_index=True)


# ============================================================================
# TAB 3 - FEATURE INSIGHTS (SHAP)
# ============================================================================
with tab_shap:
    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)
    sh1, sh2 = st.columns(2)

    with sh1:
        fig6a = os.path.join(FIG_DIR, "fig6a_shap_bar.png")
        if os.path.exists(fig6a):
            st.image(fig6a, caption="Figure 6a - Mean |SHAP| global importance (top 15 features)",
                     width="stretch")
        else:
            st.info("fig6a_shap_bar.png not found.")

    with sh2:
        fig6b = os.path.join(FIG_DIR, "fig6b_shap_beeswarm.png")
        if os.path.exists(fig6b):
            st.image(fig6b, caption="Figure 6b - SHAP beeswarm: direction & magnitude of feature impact",
                     width="stretch")
        else:
            st.info("fig6b_shap_beeswarm.png not found.")

    st.markdown('<div class="section-header">Top Feature Dependence Plots</div>', unsafe_allow_html=True)
    dep_figs = [f for f in os.listdir(FIG_DIR) if f.startswith("fig6c_shap_dep_")]
    if dep_figs:
        dep_cols = st.columns(min(3, len(dep_figs)))
        for i, fn in enumerate(sorted(dep_figs)[:3]):
            fpath = os.path.join(FIG_DIR, fn)
            feat_name = fn.replace("fig6c_shap_dep_","").replace("_"," ").replace(".png","")
            with dep_cols[i]:
                st.image(fpath, caption=f"SHAP Dependence: {feat_name}", width="stretch")
    else:
        st.info("No dependence plot images found.")

    st.markdown("""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-top:16px;">
        <p style="color:#8b949e; margin:0 0 10px; font-size:0.82rem; text-transform:uppercase; letter-spacing:.06em">Key Findings from SHAP Analysis</p>
        <ul style="color:#e6edf3; line-height:1.8;">
            <li><strong>HR_lag1</strong> - Whether last month was High-Risk is the single strongest predictor, highlighting the temporal autocorrelation of leptospirosis outbreaks.</li>
            <li><strong>District_risk_rate</strong> - Chronic endemic districts (e.g. Ratnapura, Kandy) have persistently elevated baseline risk.</li>
            <li><strong>HR_roll3</strong> - A 3-month rolling High-Risk rate amplifies predictive signal for sustained outbreak periods.</li>
            <li><strong>Precipitation features</strong> - High current and lagged rainfall strongly increases risk probability, consistent with the waterborne transmission pathway.</li>
            <li><strong>Soil Moisture</strong> - Elevated soil saturation creates environmental conditions favourable for <i>Leptospira</i> survival.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAB 4 - ABOUT
# ============================================================================
with tab_about:
    st.markdown("""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:28px; max-width:860px;">
        <h3 style="color:#58a6ff; margin-top:0">About This Project</h3>
        <p style="color:#c9d1d9; line-height:1.8;">
            This dashboard presents the outputs of a Master of Science research project focused on
            <strong>district-level predictive modelling of Leptospirosis incidence in Sri Lanka</strong>.
        </p>
        <h4 style="color:#8b949e; font-size:0.9rem; text-transform:uppercase; letter-spacing:.07em">Research Objectives</h4>
        <ul style="color:#c9d1d9; line-height:1.9;">
            <li>Build a monthly, district-level binary classifier to distinguish <em>High-Risk</em> from <em>Low-Risk</em> months.</li>
            <li>Incorporate climatic, temporal, agricultural, and epidemiological lag features.</li>
            <li>Evaluate multiple candidate classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) with time-series cross-validation.</li>
            <li>Select the best model using statistical testing (DeLong AUC comparison) and a cost-sensitive decision threshold.</li>
            <li>Explain model predictions using SHAP values to ensure interpretability for public health practitioners.</li>
        </ul>
        <h4 style="color:#8b949e; font-size:0.9rem; text-transform:uppercase; letter-spacing:.07em">Modelling Approach</h4>
        <ul style="color:#c9d1d9; line-height:1.9;">
            <li><strong>Split strategy</strong>: Time-series split (5 folds, 1-month gap) to prevent data leakage.</li>
            <li><strong>Hyperparameter tuning</strong>: Optuna TPE sampler (40 trials per model).</li>
            <li><strong>Threshold</strong>: 0.33 (Bayesian cost-ratio FN:FP = 2:1) to maximise sensitivity.</li>
            <li><strong>Final model</strong>: Random Forest - Selected based on evaluation metrics.</li>
        </ul>
        <h4 style="color:#8b949e; font-size:0.9rem; text-transform:uppercase; letter-spacing:.07em">Dataset</h4>
        <ul style="color:#c9d1d9; line-height:1.9;">
            <li>25 districts x monthly time steps.</li>
            <li>Train: 3,900 samples · Test: 1,200 samples.</li>
            <li>51 features: climate, soil, agriculture, population, BioClim, and engineered lag/rolling features.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sri Lanka map
    map_img = os.path.join(FIG_DIR, "sri_lanka_district_incidence_map.png")
    if os.path.exists(map_img):
        col_map, col_space = st.columns([1, 1])
        with col_map:
            st.image(map_img, caption="District-level Leptospirosis incidence map - Sri Lanka",
                     width="stretch")

# ─ Footer ─
st.markdown("""
<div class="footer">
    MSC Research · Leptospirosis Risk Prediction · Sri Lanka · 2026<br>
    Built with Streamlit · Random Forest · SHAP · Python
</div>
""", unsafe_allow_html=True)
