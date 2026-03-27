import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from excel_agent import (
    normalize_path_list,
    load_workbook_all_views,
    search_raw_cells,
    compute_numeric_summary,
    fuzzy_find_column,
    filter_table_by_value,
)

st.set_page_config(page_title="Excel AI Agent (Unstructured)", layout="wide")


# --------------------------
# Load files
# --------------------------
st.title("Excel AI Agent Dashboard (Unstructured / Odd Excel)")

DATA_DIR = st.sidebar.text_input("Data folder", value="data")
work_dir = st.sidebar.text_input("Work folder (for .xls conversion)", value=".work")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(work_dir, exist_ok=True)

files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith((".xlsx", ".xls"))]
files = sorted(files)

if not files:
    st.info("Put your Excel files into the `data/` folder, then refresh.")
    st.stop()

usable_files = normalize_path_list(files, work_dir=work_dir)
file_labels = [os.path.basename(p) for p in usable_files]

selected_file = st.sidebar.selectbox("Select Excel file", options=usable_files, format_func=os.path.basename)

@st.cache_data(show_spinner=True)
def cached_load(file_path: str):
    return load_workbook_all_views(file_path)

wbdata = cached_load(selected_file)
sheet = st.sidebar.selectbox("Select sheet", wbdata.sheets)

raw = wbdata.raw_by_sheet[sheet]
mat = wbdata.matrix_by_sheet[sheet]
table = wbdata.table_by_sheet[sheet]
meta = wbdata.table_meta_by_sheet[sheet]


# --------------------------
# Top KPIs
# --------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("File", os.path.basename(wbdata.path))
kpi2.metric("Sheet", sheet)
kpi3.metric("Raw cells", f"{len(raw):,}")
kpi4.metric("Detected header row", f"{meta.get('header_row')} (conf {meta.get('confidence'):.2f})")


# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Search (Raw)", "📋 Preview (Matrix)", "🧮 Table Ops", "📊 Charts"])

with tab1:
    st.subheader("Search across ALL cells (raw grid)")
    q = st.text_input("Search text (matches any cell value)", value="")
    limit = st.slider("Max results", 50, 1000, 200)

    if q:
        hits = search_raw_cells(raw, q, limit=limit)
        st.write(f"Matches: {len(hits):,}")
        st.dataframe(hits, use_container_width=True, height=450)
    else:
        st.caption("Enter text to search. This works even when the sheet is unstructured.")

with tab2:
    st.subheader("Sheet preview (matrix view)")
    st.caption("This is a faithful grid-like representation. For huge sheets, limit rows/cols below.")

    rmax = st.number_input("Show first N rows", min_value=10, max_value=int(mat.shape[0] or 10), value=min(80, int(mat.shape[0] or 80)))
    cmax = st.number_input("Show first N columns", min_value=5, max_value=int(mat.shape[1] or 5), value=min(30, int(mat.shape[1] or 30)))
    st.dataframe(mat.iloc[: int(rmax), : int(cmax)], use_container_width=True, height=500)

with tab3:
    st.subheader("Operations on detected table (best-effort)")
    if table.empty:
        st.warning("No table detected (or sheet is too irregular). Use Raw Search tab instead.")
    else:
        st.caption("You can filter by a text column, then compute sum/min/max/mean, etc. on numeric columns.")

        st.write("Detected table columns:")
        st.code(", ".join(list(table.columns)))

        col_hint = st.text_input("Filter column (fuzzy name)", value="")
        val_hint = st.text_input("Filter value contains", value="")

        filtered = table.copy()
        if col_hint and val_hint:
            col = fuzzy_find_column(table, col_hint)
            if col:
                filtered = filter_table_by_value(table, col, val_hint)
                st.success(f"Filtering by column: {col} contains '{val_hint}' → {len(filtered):,} rows")
            else:
                st.error("Could not match your filter column hint to a table column name.")

        st.dataframe(filtered.head(300), use_container_width=True, height=350)

        st.markdown("### Numeric summary (sum/min/max/mean/median)")
        summary = compute_numeric_summary(filtered)
        if summary.empty:
            st.info("No numeric columns detected in the table view.")
        else:
            st.dataframe(summary, use_container_width=True, height=350)

with tab4:
    st.subheader("Charts (auto-enabled when numeric data exists)")
    if table.empty:
        st.warning("No table detected for charting. Try another sheet or use raw search.")
    else:
        df = table.copy()
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        st.caption("Pick chart type and columns. If your sheet has geo columns (lat/long), a map will work.")

        chart_type = st.selectbox(
            "Chart type",
            ["Bar", "Pie", "Line", "Scatter", "Heatmap (corr)", "Map (lat/lon if present)"],
        )

        if chart_type in ["Bar", "Line"]:
            x = st.selectbox("X (category)", options=non_numeric_cols if non_numeric_cols else list(df.columns))
            y = st.selectbox("Y (numeric)", options=numeric_cols if numeric_cols else list(df.columns))
            agg = st.selectbox("Aggregation", ["sum", "mean", "max", "min", "count"])
            if x and y:
                if agg == "count":
                    plot_df = df.groupby(x, dropna=False).size().reset_index(name="count")
                    fig = px.bar(plot_df, x=x, y="count", title=f"{agg} by {x}")
                else:
                    plot_df = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
                    fig = px.bar(plot_df, x=x, y=y, title=f"{agg}({y}) by {x}") if chart_type == "Bar" else px.line(plot_df, x=x, y=y, title=f"{agg}({y}) by {x}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie":
            names = st.selectbox("Names (category)", options=non_numeric_cols if non_numeric_cols else list(df.columns))
            values = st.selectbox("Values (numeric)", options=numeric_cols if numeric_cols else list(df.columns))
            agg = st.selectbox("Aggregation", ["sum", "mean", "count"])
            if agg == "count":
                plot_df = df.groupby(names, dropna=False).size().reset_index(name="count")
                fig = px.pie(plot_df, names=names, values="count", title=f"count by {names}")
            else:
                plot_df = df.groupby(names, dropna=False)[values].agg(agg).reset_index()
                fig = px.pie(plot_df, names=names, values=values, title=f"{agg}({values}) by {names}")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter":
            x = st.selectbox("X (numeric)", options=numeric_cols if numeric_cols else list(df.columns))
            y = st.selectbox("Y (numeric)", options=numeric_cols if numeric_cols else list(df.columns), index=1 if len(numeric_cols) > 1 else 0)
            color = st.selectbox("Color (optional)", options=["(none)"] + non_numeric_cols)
            fig = px.scatter(df, x=x, y=y, color=None if color == "(none)" else color, title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Heatmap (corr)":
            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns for correlation heatmap.")
            else:
                corr = df[numeric_cols].corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=True, title="Correlation heatmap")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Map (lat/lon if present)":
            # Try to guess lat/lon columns
            cols_lower = {c.lower(): c for c in df.columns}
            lat_col = None
            lon_col = None
            for k in cols_lower:
                if k in ["lat", "latitude"]:
                    lat_col = cols_lower[k]
                if k in ["lon", "lng", "longitude"]:
                    lon_col = cols_lower[k]

            if not (lat_col and lon_col):
                st.info("No latitude/longitude columns found (expected names like lat/lon or latitude/longitude).")
            else:
                size_col = st.selectbox("Bubble size (optional numeric)", options=["(none)"] + numeric_cols)
                color_col = st.selectbox("Color (optional)", options=["(none)"] + non_numeric_cols)
                fig = px.scatter_mapbox(
                    df.dropna(subset=[lat_col, lon_col]),
                    lat=lat_col,
                    lon=lon_col,
                    size=None if size_col == "(none)" else size_col,
                    color=None if color_col == "(none)" else color_col,
                    zoom=2,
                    height=600,
                    title="Bubble map",
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
