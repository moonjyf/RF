import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from lime.lime_tabular import LimeTabularExplainer

# 加载模型和测试数据
model = joblib.load('RF.pkl')
X_test = pd.read_csv('x_test.csv')

# 定义特征名
feature_names = X_test.columns.tolist()

st.title("糖尿病周围神经病变风险预测系统")

# 表单接口
with st.form("input_form"):
    st.subheader("请填写以下特征信息：")
    inputs = []
    for col in feature_names:
        if '无0有1' in col or '无0有1）' in col or '（0-2）' in col:
            unique_values = sorted(X_test[col].dropna().unique())
            try:
                default = int(np.median(unique_values))
            except:
                default = 0
            inputs.append(st.selectbox(col, options=unique_values, index=unique_values.index(default)))
        else:
            min_val = float(X_test[col].min())
            max_val = float(X_test[col].max())
            default_val = float(X_test[col].median())
            inputs.append(st.number_input(col, value=default_val, min_value=min_val, max_value=max_val))

    submitted = st.form_submit_button("提交预测")

if submitted:
    # 构造用户输入数据
    input_data = pd.DataFrame([inputs], columns=feature_names)
    st.subheader("模型输入特征如下：")
    st.dataframe(input_data)

    # 预测
    predicted_class = model.predict(input_data)[0]
    predicted_proba = model.predict_proba(input_data)[0]
    probability = predicted_proba[1] * 100

    st.write(f"**预测类别：** {predicted_class} (1: 有DPN, 0: 无DPN)")
    st.write(f"**预测概率：** {predicted_proba}")

    if predicted_class == 1:
        advice = (
            f"⚠️ **根据模型预测，您患有DPN的风险较高。**\n\n"
            f"预测概率为 **{probability:.1f}%**\n\n"
            "建议尽快展察医生，进行详细检查和应对。"
        )
    else:
        advice = (
            f"✅ **预测结果良好，目前无DPN风险**\n\n"
            f"预测概率仅为 **{probability:.1f}%**\n\n"
            "请继续保持良好生活习惯和随访。"
        )
    st.markdown(advice)

    # ===== SHAP Explanation =====
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(input_data)

    if isinstance(shap_values, list):
        shap_value_row = shap_values[1][0] if predicted_class == 1 else shap_values[0][0]
        base_value = explainer_shap.expected_value[1] if predicted_class == 1 else explainer_shap.expected_value[0]
    else:
        shap_value_row = shap_values[0]
        base_value = explainer_shap.expected_value

    plt.figure()
    shap.force_plot(
        base_value=base_value,
        shap_values=shap_value_row,
        features=input_data.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()

    st.image("shap_force_plot.png", caption="SHAP Force Plot Explanation")

    # ===== LIME Explanation =====
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=feature_names,
        class_names=['Not sick', 'Sick'],
        mode='classification'
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=input_data.iloc[0].values,
        predict_fn=model.predict_proba
    )

    lime_html = lime_exp.as_html(show_table=False)
    components.html(lime_html, height=800, scrolling=True)

