"""
Streamlit dashboard for the Land Use Regression air quality project.
Run after data download: streamlit run app.py
"""

import pathlib
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.lur_model import (
    run_full_evaluation,
    get_ridge_coefficients,
    get_gb_importance,
    get_dataset_summary,
    get_active_features,
    DATA_PATH,
)

st.set_page_config(page_title="LUR Air Quality Finland", layout="wide")

st.title("Land Use Regression Air Quality Prediction")
st.markdown(
    "This project applies Land Use Regression modelling to predict annual mean "
    "NO\u2082 and PM\u2082.\u2085 concentrations at Finnish monitoring stations. "
    "Measurements come from the EEA Air Quality e-Reporting verified dataset "
    "covering 2019 to 2022. Three models are compared: Ridge regression as the "
    "classical LUR baseline following Morley and Gulliver (2018), Random Forest, "
    "and Gradient Boosting. Both five-fold cross-validation and Leave-One-Out "
    "Cross-Validation are used for evaluation, consistent with LUR best practice."
)

# The data check below is only needed when running locally without the CSV.
# It is commented out for Streamlit Cloud deployment where the CSV is
# committed to the repository and always present.
# if not DATA_PATH.exists():
#     st.error(
#         "Processed data not found. Please run the following command "
#         "from the project folder, then refresh this page."
#     )
#     st.code("python scripts/download_eea_data.py", language="bash")
#     st.stop()

# Pollutant selector — everything on the page responds to this choice
pollutant_key = st.selectbox(
    "Select pollutant to model",
    options=["NO2_ugm3", "PM2_5_ugm3"],
    format_func=lambda x: "NO\u2082 (\u03bcg/m\u00b3)" if x == "NO2_ugm3" else "PM\u2082.\u2085 (\u03bcg/m\u00b3)",
)
pollutant_label = "NO\u2082" if pollutant_key == "NO2_ugm3" else "PM\u2082.\u2085"

active_features = get_active_features(pollutant_key)
if active_features:
    st.info(f"Active predictor variables for {pollutant_label}: {', '.join(active_features)}")

tab_data, tab_perf, tab_scatter, tab_vars, tab_map = st.tabs([
    "Dataset", "Model Performance", "LOOCV Scatter", "Variable Analysis", "Station Map"
])


with tab_data:
    st.subheader("EEA Finland monitoring stations")
    df = get_dataset_summary()

    if df.empty:
        st.warning("Dataset is empty.")
        st.stop()

    # Show metrics for the selected pollutant
    conc_col = pollutant_key
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unique stations",   df["station_id"].nunique() if "station_id" in df.columns else len(df))
    c2.metric("Station-year rows", len(df))
    if conc_col in df.columns:
        c3.metric(f"Mean {pollutant_label} (\u03bcg/m\u00b3)", f"{df[conc_col].mean():.1f}")
        c4.metric(f"Max {pollutant_label} (\u03bcg/m\u00b3)",  f"{df[conc_col].max():.1f}")

    st.dataframe(df, use_container_width=True)

    # Distribution for selected pollutant
    if conc_col in df.columns:
        fig = px.histogram(
            df, x=conc_col, nbins=20,
            title=f"{pollutant_label} distribution across station-years",
            color_discrete_sequence=["#2e6da4" if pollutant_key == "NO2_ugm3" else "#2e8b57"],
            labels={conc_col: f"{pollutant_label} (\u03bcg/m\u00b3)"},
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Trend over years for selected pollutant
    if "year" in df.columns and conc_col in df.columns:
        trend = df.groupby("year")[conc_col].mean().reset_index()
        fig2  = px.line(
            trend, x="year", y=conc_col, markers=True,
            title=f"Annual mean {pollutant_label} across all Finnish stations",
            labels={conc_col: f"{pollutant_label} (\u03bcg/m\u00b3)", "year": "Year"},
            color_discrete_sequence=["#2e6da4" if pollutant_key == "NO2_ugm3" else "#2e8b57"],
        )
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)


with tab_perf:
    st.subheader(f"Cross-validated model performance — {pollutant_label}")
    st.markdown(
        "Both five-fold cross-validation and LOOCV results are shown for the "
        f"selected pollutant ({pollutant_label}). LOOCV is the standard evaluation "
        "method in the LUR literature because it uses every available station as a "
        "validation point, which is important when the dataset is small."
    )

    with st.spinner(f"Fitting models for {pollutant_label}..."):
        try:
            results_df, _ = run_full_evaluation(pollutant_key)
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            st.stop()

    c1, c2, c3 = st.columns(3)
    for i, row in results_df.iterrows():
        col = [c1, c2, c3][i]
        with col:
            st.metric(
                label=row["Model"],
                value=f"LOOCV R\u00b2 = {row['LOOCV R2']:.3f}",
                delta=f"RMSE = {row['LOOCV RMSE']:.2f} \u03bcg/m\u00b3",
                delta_color="off",
            )

    display_cols = ["Model", "5-fold RMSE", "5-fold R2", "LOOCV RMSE", "LOOCV R2"]
    st.dataframe(
        results_df[display_cols].style
            .highlight_max(subset=["5-fold R2","LOOCV R2"], color="#c8f7c5")
            .highlight_min(subset=["5-fold RMSE","LOOCV RMSE"], color="#ffd6d6")
            .format({"5-fold RMSE":"{:.2f}","5-fold R2":"{:.3f}",
                     "LOOCV RMSE":"{:.2f}","LOOCV R2":"{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    colors = ["#2e6da4", "#2e8b57", "#c0622e"]

    fig_r2 = go.Figure()
    for i, row in results_df.iterrows():
        fig_r2.add_trace(go.Bar(
            x=[row["Model"]], y=[row["LOOCV R2"]],
            marker_color=colors[i],
            text=[f"{row['LOOCV R2']:.3f}"],
            textposition="outside",
            name=row["Model"],
            width=0.5,
        ))
    fig_r2.update_layout(
        title=f"LOOCV R\u00b2 by model — {pollutant_label}",
        yaxis=dict(range=[0, 1.15], title="LOOCV R\u00b2", gridcolor="#eeeeee"),
        xaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    fig_rmse = go.Figure()
    for i, row in results_df.iterrows():
        fig_rmse.add_trace(go.Bar(
            x=[row["Model"]], y=[row["LOOCV RMSE"]],
            marker_color=colors[i],
            text=[f"{row['LOOCV RMSE']:.2f}"],
            textposition="outside",
            name=row["Model"],
            width=0.5,
        ))
    max_rmse = results_df["LOOCV RMSE"].max()
    fig_rmse.update_layout(
        title=f"LOOCV RMSE by model — {pollutant_label} (\u03bcg/m\u00b3)",
        yaxis=dict(range=[0, max_rmse * 1.3], title="LOOCV RMSE (\u03bcg/m\u00b3)",
                   gridcolor="#eeeeee"),
        xaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, height=380,
    )
    st.plotly_chart(fig_rmse, use_container_width=True)


with tab_scatter:
    st.subheader(f"LOOCV observed versus predicted — {pollutant_label}")
    st.markdown(
        f"Each plot shows {pollutant_label} leave-one-out predictions against "
        "actual observed concentrations. Points on the dashed diagonal represent "
        "perfect predictions. The RMSE and R\u00b2 values shown on each chart "
        "are computed directly from the LOOCV predictions."
    )

    with st.spinner("Running LOOCV..."):
        try:
            _, loocv_preds = run_full_evaluation(pollutant_key)
        except Exception as e:
            st.error(f"LOOCV failed: {e}")
            st.stop()

    cols = st.columns(3)
    colors = ["#2e6da4", "#2e8b57", "#c0622e"]

    for i, (name, (yt, yp)) in enumerate(loocv_preds.items()):
        mn = min(yt.min(), yp.min()) * 0.9
        mx = max(yt.max(), yp.max()) * 1.1
        rmse_val = np.sqrt(np.mean((yt - yp) ** 2))
        ss_res   = np.sum((yt - yp) ** 2)
        ss_tot   = np.sum((yt - np.mean(yt)) ** 2)
        r2_val   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yt, y=yp, mode="markers",
            marker=dict(color=colors[i], size=9, opacity=0.75),
            hovertemplate=(
                f"Observed: %{{x:.2f}} \u03bcg/m\u00b3<br>"
                f"Predicted: %{{y:.2f}} \u03bcg/m\u00b3<extra></extra>"
            ),
        ))
        fig.add_shape(
            type="line", x0=mn, y0=mn, x1=mx, y1=mx,
            line=dict(color="grey", width=1.5, dash="dash"),
        )
        fig.update_layout(
            title=(
                f"{name}<br>"
                f"<sup>RMSE = {rmse_val:.2f} \u03bcg/m\u00b3 &nbsp; "
                f"R\u00b2 = {r2_val:.3f}</sup>"
            ),
            xaxis_title=f"Observed {pollutant_label} (\u03bcg/m\u00b3)",
            yaxis_title=f"Predicted {pollutant_label} (\u03bcg/m\u00b3)",
            plot_bgcolor="white", paper_bgcolor="white",
            height=400, showlegend=False,
        )
        with cols[i]:
            st.plotly_chart(fig, use_container_width=True)


with tab_vars:
    st.subheader(f"Predictor variable analysis — {pollutant_label}")
    st.markdown(
        f"Predictor importance for {pollutant_label}. The left chart shows "
        "standardised Ridge regression coefficients. A positive value means higher "
        "values of that predictor associate with higher concentrations, which is "
        "expected for road_count_500m. The right chart shows Gradient Boosting "
        "feature importances as mean decrease in impurity across all trees."
    )

    try:
        ridge_coefs = get_ridge_coefficients(pollutant_key)
        gb_imp      = get_gb_importance(pollutant_key)
    except Exception as e:
        st.error(f"Could not compute variable analysis: {e}")
        st.stop()

    ca, cb = st.columns(2)

    with ca:
        fig_r = go.Figure(go.Bar(
            x=ridge_coefs["Coefficient"],
            y=ridge_coefs["Feature"],
            orientation="h",
            marker=dict(
                color=ridge_coefs["Coefficient"],
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="Coeff", len=0.6),
            ),
        ))
        fig_r.update_layout(
            title=f"Ridge coefficients — {pollutant_label}",
            xaxis_title="Standardised coefficient",
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with cb:
        fig_g = go.Figure(go.Bar(
            x=gb_imp["Importance"],
            y=gb_imp["Feature"],
            orientation="h",
            marker=dict(
                color=gb_imp["Importance"],
                colorscale="Teal",
                showscale=True,
                colorbar=dict(title="Importance", len=0.6),
            ),
            text=[f"{v:.3f}" for v in gb_imp["Importance"]],
            textposition="outside",
        ))
        fig_g.update_layout(
            title=f"Gradient Boosting importances — {pollutant_label}",
            xaxis_title="Importance",
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_g, use_container_width=True)


with tab_map:
    st.subheader(f"Finnish EEA monitoring stations — {pollutant_label}")
    df_map = get_dataset_summary()

    has_coords = (
        "latitude"  in df_map.columns and
        "longitude" in df_map.columns and
        df_map["latitude"].notna().any()
    )

    if not has_coords:
        st.info(
            "Station coordinates are not yet in the CSV. Delete "
            "data/processed/eea_finland_annual.csv and "
            "data/raw/metadata_FI.csv, then re-run "
            "python scripts/download_eea_data.py."
        )
    else:
        # Use annual mean of selected pollutant per station for the colour scale
        conc_col = pollutant_key
        if conc_col not in df_map.columns:
            st.warning(f"Column {conc_col} not found in dataset.")
        else:
            map_df = (
                df_map.dropna(subset=["latitude","longitude"])
                      .groupby(["station_id","latitude","longitude"])
                      .agg(mean_conc=(conc_col, "mean"))
                      .reset_index()
                      .rename(columns={"mean_conc": conc_col})
            )

            if len(map_df) == 0:
                st.warning("No stations with coordinates found.")
            else:
                fig_map = px.scatter_mapbox(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    color=conc_col,
                    color_continuous_scale="YlOrRd",
                    size=conc_col,
                    size_max=20,
                    zoom=4,
                    center={"lat": 64.5, "lon": 26.0},
                    mapbox_style="open-street-map",
                    hover_data={
                        "station_id": True,
                        conc_col: ":.2f",
                        "latitude": ":.4f",
                        "longitude": ":.4f",
                    },
                    title=f"Mean {pollutant_label} (\u03bcg/m\u00b3) per station (2019-2022)",
                    labels={conc_col: f"{pollutant_label} (\u03bcg/m\u00b3)"},
                )
                fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_map, use_container_width=True)

                st.markdown(
                    f"Showing {len(map_df)} stations. Circle size and colour both "
                    f"represent mean annual {pollutant_label} concentration. "
                    "Larger, darker circles indicate higher pollution."
                )


st.caption(
    "Data: EEA Air Quality e-Reporting Verified dataset 2019 to 2022. "
    "Road counts from OpenStreetMap via Overpass API. "
    "Methodology follows Morley and Gulliver (2018), "
    "Environmental Modelling and Software, 105, 17-23."
)
