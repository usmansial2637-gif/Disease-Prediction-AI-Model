"""
app.py
------
Streamlit web application for the Disease Prediction project.

Lets a user pick a dataset + trained model, enter patient values through a
form, and get a live prediction with confidence, a suggested doctor,
general recommendations, and an optional SHAP "why this prediction"
breakdown. Every prediction is logged to a local SQLite database
(outputs handled in src/database.py) and browsable in the History tab.

Run with:
    streamlit run app.py
(Run `python main.py` at least once first so trained models exist in
outputs/models/)
"""

import json
import os
import pickle

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DATASETS
from src.explain import compute_shap_values, top_reasons
from src.database import log_prediction, fetch_history, clear_history
from src.pdf_report import build_pdf_report
from src.utils import safe_model_filename
from src.constants import DATASET_LABELS, DISEASE_LABELS, DOCTOR_MAP, RECOMMENDATIONS
from src.auth import verify_login, create_user, list_users, delete_user, ROLES
from src.chatbot import respond as chatbot_respond

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="centered")


def safe_name_to_label(fname_stub: str) -> str:
    return fname_stub.replace("_", " ")


# ============================================================
# LOGIN GATE (Level 9: Authentication + roles)
# ============================================================
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None
    st.session_state["auth_role"] = None

if st.session_state["auth_user"] is None:
    st.title("🩺 Disease Prediction System")
    st.caption("Please sign in to continue.")

    login_tab, register_tab = st.tabs(["Sign In", "Register (Patient)"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary")
        if submitted:
            role = verify_login(username, password)
            if role:
                st.session_state["auth_user"] = username
                st.session_state["auth_role"] = role
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Default admin account: `admin` / `admin123` (change this in a real deployment).")

    with register_tab:
        st.caption("Self-registration creates a **patient** account. Admin/doctor accounts "
                   "are created by an admin in the Users tab.")
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            reg_submitted = st.form_submit_button("Register")
        if reg_submitted:
            if not new_username or not new_password:
                st.error("Username and password are required.")
            else:
                ok, msg = create_user(new_username, new_password, "patient")
                if ok:
                    st.success(f"{msg} You can now sign in.")
                else:
                    st.error(msg)

    st.stop()

CURRENT_USER = st.session_state["auth_user"]
CURRENT_ROLE = st.session_state["auth_role"]

# ============================================================
# HEADER + NAV
# ============================================================
header_col, logout_col = st.columns([5, 1])
with header_col:
    st.title("🩺 Disease Prediction System")
    st.caption(f"Signed in as **{CURRENT_USER}** ({CURRENT_ROLE}) · "
               "ML demo — Logistic Regression · SVM · Random Forest · XGBoost · LightGBM · CatBoost")
with logout_col:
    st.write("")
    if st.button("Log out"):
        st.session_state["auth_user"] = None
        st.session_state["auth_role"] = None
        st.session_state.pop("prediction_result", None)
        st.rerun()

tab_names = ["🔮 Predict", "💬 Chatbot", "🗂️ History"]
if CURRENT_ROLE in ("admin", "doctor"):
    tab_names.append("📊 Dashboard")
if CURRENT_ROLE == "admin":
    tab_names.append("👤 Users")

tabs = st.tabs(tab_names)
tab_predict, tab_chatbot, tab_history = tabs[0], tabs[1], tabs[2]
tab_dashboard = tabs[3] if CURRENT_ROLE in ("admin", "doctor") else None
tab_users = tabs[-1] if CURRENT_ROLE == "admin" else None

# ============================================================
# PREDICT TAB
# ============================================================
with tab_predict:
    dataset_key = st.selectbox(
        "Choose a dataset", list(DATASET_LABELS.keys()), format_func=lambda k: DATASET_LABELS[k]
    )

    # --- Discover trained models + feature map for this dataset ---
    available_models = []
    if os.path.isdir(MODELS_DIR):
        for fname in sorted(os.listdir(MODELS_DIR)):
            if fname.startswith(dataset_key + "_") and fname.endswith(".pkl") \
                    and "scaler" not in fname:
                model_label = safe_name_to_label(fname[len(dataset_key) + 1: -4])
                available_models.append((model_label, fname))

    feature_map_path = os.path.join(MODELS_DIR, f"{dataset_key}_feature_map.json")

    if not available_models or not os.path.exists(feature_map_path):
        st.warning("No trained models found for this dataset yet. Run `python main.py` first "
                   "(it trains, tunes, and saves the models this app loads).")
        st.stop()

    with open(feature_map_path) as f:
        feature_map = json.load(f)
    all_features = feature_map["all_features"]
    selected_features = feature_map["selected_features"]
    selected_positions = [all_features.index(f) for f in selected_features]

    model_label = st.selectbox("Choose a model", [m[0] for m in available_models])
    model_file = dict(available_models)[model_label]

    with open(os.path.join(MODELS_DIR, model_file), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, f"{dataset_key}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    df, source_label = DATASETS[dataset_key]()
    st.caption(f"Source: {source_label}")
    st.caption(f"This model was trained on its top {len(selected_features)} of {len(all_features)} "
               f"features (selected via ANOVA F-score in Level 2).")

    st.subheader("Patient Input")
    st.write("Adjust the values below (defaults = dataset median):")

    user_values = {}
    cols = st.columns(2)
    for i, col_name in enumerate(all_features):
        col = cols[i % 2]
        series = df[col_name]
        default = float(series.median())
        if series.nunique() <= 3 and set(series.unique()).issubset({0, 1, 2, 3}):
            options = sorted(series.unique().tolist())
            idx = options.index(int(default)) if int(default) in options else 0
            user_values[col_name] = col.selectbox(col_name, options, index=idx)
        else:
            lo, hi = float(series.min()), float(series.max())
            user_values[col_name] = col.slider(col_name, lo, hi, default)

    show_explanation = st.checkbox("Show why (SHAP explanation)", value=False)
    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        input_df = pd.DataFrame([user_values])[all_features]
        input_scaled_full = scaler.transform(input_df)
        input_scaled = input_scaled_full[:, selected_positions]

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
        disease_name = DISEASE_LABELS.get(dataset_key, "Disease")
        predicted_label = disease_name if pred == 1 else "No Disease"

        reasons = None
        if show_explanation:
            with st.spinner("Computing SHAP explanation..."):
                X_train_scaled_full = scaler.transform(df[all_features])[:, selected_positions]
                shap_values = compute_shap_values(
                    model, X_train_scaled_full, input_scaled, selected_features,
                    max_background=50, max_eval=1
                )
                reasons = top_reasons(shap_values, 0)

        # --- Log to SQLite history (Level 5), tagged with the logged-in user (Level 9) ---
        log_prediction(
            dataset=dataset_key,
            model_name=model_label,
            feature_values=user_values,
            predicted_disease=predicted_label,
            prediction=int(pred),
            confidence=proba,
            username=CURRENT_USER,
        )

        # Stash everything needed to render the result + PDF, so it survives
        # the rerun triggered by clicking the PDF download button below.
        st.session_state["prediction_result"] = {
            "dataset_key": dataset_key,
            "model_label": model_label,
            "user_values": user_values,
            "pred": int(pred),
            "proba": float(proba) if proba is not None else None,
            "disease_name": disease_name,
            "reasons": reasons,
        }

    result = st.session_state.get("prediction_result")
    if result and result["dataset_key"] == dataset_key and result["model_label"] == model_label:
        pred = result["pred"]
        proba = result["proba"]
        disease_name = result["disease_name"]
        reasons = result["reasons"]

        st.subheader("Result")
        doctor = DOCTOR_MAP.get(dataset_key, "General Physician")
        recommendations = RECOMMENDATIONS.get(dataset_key, [])

        if pred == 1:
            conf_str = f" (confidence: {proba:.1%})" if proba is not None else ""
            st.error(f"⚠️ **{disease_name}** likely present{conf_str}")
            st.markdown(f"**Suggested Doctor:** {doctor}")
            st.markdown("**Recommendations:**")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            conf_str = f" (probability of disease: {proba:.1%})" if proba is not None else ""
            st.success(f"✅ **{disease_name}** unlikely{conf_str}")
            st.markdown("**Recommendations:**")
            st.markdown("- No immediate concern based on this input, but routine checkups are still worthwhile.")

        if reasons:
            st.markdown("**Why this prediction (top contributing features):**")
            for r in reasons:
                sign = "increased" if r["direction"] == "+" else "decreased"
                st.markdown(f"- **{r['feature']}** {sign} the risk estimate ({r['direction']}{r['pct']}% of total influence)")

        # --- PDF report download (Level 7) ---
        pdf_bytes = build_pdf_report(
            dataset_label=DATASET_LABELS.get(dataset_key, dataset_key),
            disease_label=disease_name,
            patient_values=result["user_values"],
            prediction=pred,
            confidence=proba,
            model_label=model_label,
            doctor=doctor,
            recommendations=recommendations,
            reasons=reasons,
        )
        st.download_button(
            "📄 Download PDF Report", data=pdf_bytes,
            file_name=f"{dataset_key}_prediction_report.pdf", mime="application/pdf",
        )

        st.caption("This is an educational demo, not a medical diagnostic tool. "
                   "Always consult a qualified healthcare professional.")

# ============================================================
# CHATBOT TAB
# ============================================================
with tab_chatbot:
    st.subheader("Symptom Checker Chatbot")
    st.caption("Describe your symptoms in plain language. This is a rule-based screening aid, "
               "not a real diagnosis — always follow up with a doctor.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Hi! Tell me what symptoms you're experiencing "
                                              "and I'll point you toward the right prediction tool."}
        ]

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("e.g. I have chest pain and shortness of breath")
    if user_msg:
        st.session_state["chat_messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        bot_reply = chatbot_respond(user_msg)
        st.session_state["chat_messages"].append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    if st.button("Clear chat"):
        st.session_state["chat_messages"] = []
        st.rerun()

# ============================================================
# HISTORY TAB
# ============================================================
with tab_history:
    st.subheader("Prediction History")

    is_patient = CURRENT_ROLE == "patient"
    if is_patient:
        st.caption("Showing your own predictions only.")

    hist_dataset_filter = st.selectbox(
        "Filter by dataset", ["All"] + list(DATASET_LABELS.keys()),
        format_func=lambda k: "All datasets" if k == "All" else DATASET_LABELS[k],
        key="history_filter",
    )

    rows = fetch_history(
        dataset=None if hist_dataset_filter == "All" else hist_dataset_filter,
        username=CURRENT_USER if is_patient else None,
    )

    if not rows:
        st.info("No predictions logged yet — make one in the Predict tab.")
    else:
        hist_df = pd.DataFrame(rows)
        hist_df["confidence"] = hist_df["confidence"].apply(
            lambda v: f"{v:.1%}" if v is not None else "—"
        )
        hist_df["username"] = hist_df["username"].fillna("—")
        rename_map = {
            "id": "ID", "dataset": "Dataset", "model_name": "Model", "username": "User",
            "patient_age": "Patient Age", "predicted_disease": "Predicted",
            "confidence": "Confidence", "created_at": "Date",
        }
        cols_order = ["ID", "Date", "Dataset", "Model", "Patient Age", "Predicted", "Confidence"]
        if not is_patient:
            cols_order.insert(3, "User")
        display_df = hist_df.rename(columns=rename_map)[cols_order]

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(f"{len(rows)} record(s) shown. Click a row's ID below to inspect its full input.")

        with st.expander("Inspect one record's full input (symptoms/values)"):
            id_options = [r["id"] for r in rows]
            chosen_id = st.selectbox("Record ID", id_options)
            chosen_row = next(r for r in rows if r["id"] == chosen_id)
            st.json(json.loads(chosen_row["symptoms"]))

        if CURRENT_ROLE in ("admin", "doctor"):
            if st.button("🗑️ Clear history" + ("" if hist_dataset_filter == "All" else f" for {DATASET_LABELS[hist_dataset_filter]}")):
                clear_history(dataset=None if hist_dataset_filter == "All" else hist_dataset_filter)
                st.rerun()

# ============================================================
# DASHBOARD TAB (admin/doctor only)
# ============================================================
if tab_dashboard is not None:
    with tab_dashboard:
        st.subheader("Dashboard")

        dash_dataset = st.selectbox(
            "Dataset", list(DATASET_LABELS.keys()), format_func=lambda k: DATASET_LABELS[k],
            key="dash_dataset",
        )

        dash_rows = fetch_history(dataset=dash_dataset, limit=5000)
        n_predictions = len(dash_rows)
        n_positive = sum(r["prediction"] for r in dash_rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", n_predictions)
        c2.metric("Predicted Positive", n_positive)
        c3.metric("Predicted Negative", n_predictions - n_positive)

        if dash_rows:
            st.markdown("**Disease Distribution (from logged predictions)**")
            dist_df = pd.DataFrame(dash_rows)["predicted_disease"].value_counts().reset_index()
            dist_df.columns = ["Predicted Outcome", "Count"]
            fig_dist = px.pie(dist_df, names="Predicted Outcome", values="Count", hole=0.4)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No predictions logged yet for this dataset — make some in the Predict tab "
                    "to populate this chart.")

        st.divider()

        summary_path = os.path.join(REPORTS_DIR, f"{dash_dataset}_summary.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path, index_col=0)
            st.markdown("**Model Comparison**")
            metric_choice = st.selectbox(
                "Metric", ["F1-Score", "Accuracy", "Precision", "Recall", "ROC-AUC", "CV F1 (5-fold)"],
                key="dash_metric",
            )
            fig_acc = px.bar(
                summary_df.reset_index(), x="Model", y=metric_choice, color="Model",
                title=f"{metric_choice} by Model — {DATASET_LABELS[dash_dataset]}",
            )
            fig_acc.update_layout(showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.warning("No training summary found — run `python main.py` first.")
            summary_df = None

        st.divider()

        fs_path = os.path.join(REPORTS_DIR, f"{dash_dataset}_feature_scores.csv")
        if os.path.exists(fs_path):
            fs_df = pd.read_csv(fs_path).sort_values("f_score", ascending=True)
            st.markdown("**Feature Importance (ANOVA F-score)**")
            fig_fs = px.bar(
                fs_df, x="f_score", y="feature", orientation="h", color="selected",
                title=f"Feature Ranking — {DATASET_LABELS[dash_dataset]}",
                labels={"f_score": "F-score", "feature": "Feature", "selected": "Used by model"},
            )
            st.plotly_chart(fig_fs, use_container_width=True)

        st.divider()

        if summary_df is not None:
            st.markdown("**Confusion Matrix & ROC Curve**")
            model_names = summary_df.index.tolist()
            dash_model = st.selectbox("Model", model_names, key="dash_model")
            safe_name = safe_model_filename(dash_model)

            cm_path = os.path.join(REPORTS_DIR, f"{dash_dataset}_{safe_name}_confusion.json")
            roc_path = os.path.join(REPORTS_DIR, f"{dash_dataset}_{safe_name}_roc.json")

            col_cm, col_roc = st.columns(2)

            if os.path.exists(cm_path):
                with open(cm_path) as f:
                    cm_data = json.load(f)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_data["matrix"], x=cm_data["labels"], y=cm_data["labels"],
                    colorscale="Blues", text=cm_data["matrix"], texttemplate="%{text}",
                ))
                fig_cm.update_layout(title=f"Confusion Matrix — {dash_model}",
                                      xaxis_title="Predicted", yaxis_title="Actual")
                col_cm.plotly_chart(fig_cm, use_container_width=True)

            if os.path.exists(roc_path):
                with open(roc_path) as f:
                    roc_data = json.load(f)
                if roc_data["fpr"]:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
                        name=f"AUC={roc_data['auc']}",
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                        line=dict(dash="dash", color="grey"),
                    ))
                    fig_roc.update_layout(title=f"ROC Curve — {dash_model}",
                                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    col_roc.plotly_chart(fig_roc, use_container_width=True)
                else:
                    col_roc.info("This model has no predict_proba output, so no ROC curve.")

# ============================================================
# USERS TAB (admin only)
# ============================================================
if tab_users is not None:
    with tab_users:
        st.subheader("User Management")

        st.markdown("**Existing Users**")
        users = list_users()
        st.dataframe(
            pd.DataFrame(users).rename(columns={
                "id": "ID", "username": "Username", "role": "Role", "created_at": "Created",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("**Create a New User**")
        with st.form("create_user_form"):
            cu_col1, cu_col2, cu_col3 = st.columns(3)
            new_username = cu_col1.text_input("Username")
            new_password = cu_col2.text_input("Password", type="password")
            new_role = cu_col3.selectbox("Role", ROLES)
            create_submitted = st.form_submit_button("Create User")
        if create_submitted:
            if not new_username or not new_password:
                st.error("Username and password are required.")
            else:
                ok, msg = create_user(new_username, new_password, new_role)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()

        st.markdown("**Delete a User**")
        deletable = [u["username"] for u in users if u["username"] != "admin"]
        if deletable:
            del_col1, del_col2 = st.columns([3, 1])
            user_to_delete = del_col1.selectbox("Select user", deletable)
            if del_col2.button("🗑️ Delete"):
                ok, msg = delete_user(user_to_delete)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
        else:
            st.caption("No deletable users yet (the built-in admin account can't be deleted).")
