import io
import re
import warnings
from typing import Optional, Tuple

import dtale
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.neighbors import LocalOutlierFactor
    LOF_AVAILABLE = True
except Exception:
    LOF_AVAILABLE = False

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Agensi Nuklear Malaysia Data Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# =========================
# Sidebar: Logo and Developer
# =========================
st.sidebar.image(
    "https://brand.umpsa.edu.my/images/logo-umpsa-full-color2.png",
    use_container_width=True
)

st.sidebar.image(
    "https://www.majalahsains.com/wp-content/uploads/2012/05/Logo-Agensi-Nuklear-Malaysia.png",
    use_container_width=True
)

st.sidebar.markdown("## Agensi Nuklear Malaysia Data Analytics Platform")
st.sidebar.markdown("---")
st.sidebar.markdown("### Developers")
st.sidebar.write("**Assoc. Prof. Dr. Ku Muhammad Naim Ku Khalif**")
st.sidebar.write("Centre for Mathematical Sciences")
st.sidebar.write("Universiti Malaysia Pahang Al-Sultan Abdullah")
st.sidebar.write("Email: kunaim@umpsa.edu.my")
st.sidebar.write("**Dr. Hanafi Ithnin**")
st.sidebar.write("Bahagian Teknologi Industri (BTI)")
st.sidebar.write("Agensi Nuklear Malaysia")
st.sidebar.write("Email: hanafi_i@nm.gov.my")
st.sidebar.markdown("---")

# =========================
# Helpers
# =========================
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df.columns = [re.sub(r"^%\s*", "", c).strip() for c in df.columns]
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def try_convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col].astype(str).str.strip(), errors="ignore")
            df[col] = converted
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                df[col] = parsed
    return df


def read_text_like_file(uploaded_file, sep_option: str) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="ignore")
    preview = "\n".join(text.splitlines()[:20])

    if sep_option == "Auto detect":
        # Common scientific txt/data files may be whitespace separated and may start with % comments.
        try:
            return pd.read_csv(
                io.StringIO(text),
                sep=None,
                engine="python",
                comment="#",
                skipinitialspace=True,
            )
        except Exception:
            pass
        try:
            return pd.read_csv(
                io.StringIO(text),
                sep=r"\s+",
                engine="python",
                comment="#",
                skipinitialspace=True,
            )
        except Exception:
            pass
    elif sep_option == "Whitespace":
        return pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python", comment="#")
    elif sep_option == "Comma":
        return pd.read_csv(io.StringIO(text), sep=",", engine="python", comment="#")
    elif sep_option == "Tab":
        return pd.read_csv(io.StringIO(text), sep="\t", engine="python", comment="#")
    elif sep_option == "Semicolon":
        return pd.read_csv(io.StringIO(text), sep=";", engine="python", comment="#")
    elif sep_option == "Pipe":
        return pd.read_csv(io.StringIO(text), sep="|", engine="python", comment="#")

    raise ValueError(f"Unable to parse text-like file. Preview:\n{preview}")


@st.cache_data(show_spinner=True)
def load_data(uploaded_file, sheet_name=None, sep_option="Auto detect") -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    elif file_name.endswith((".txt", ".data", ".dat", ".tsv")):
        df = read_text_like_file(uploaded_file, sep_option)
    else:
        raise ValueError("Unsupported file type. Please upload CSV, Excel, TXT, DATA, DAT or TSV.")

    df = try_convert_types(df)
    df = df.dropna(axis=1, how="all")
    return df


def get_column_groups(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]
    return numeric_cols, datetime_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def prepare_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    prepared = X.copy()
    for col in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[col]):
            prepared[col] = prepared[col].astype("int64") // 10**9
        elif pd.api.types.is_object_dtype(prepared[col]):
            prepared[col] = prepared[col].astype(str)
        else:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    constant_cols = [col for col in prepared.columns if prepared[col].nunique(dropna=True) <= 1]
    if constant_cols:
        prepared = prepared.drop(columns=constant_cols)
    return prepared


def build_model_pipeline(X_train: pd.DataFrame, task: str, model_choice: str):
    preprocessor = build_preprocessor(X_train)
    steps = [("preprocess", preprocessor)]

    if X_train.shape[1] > 2:
        if task == "Regression":
            k_features = min(8, max(2, X_train.shape[1]))
            steps.append(("feature_selection", SelectKBest(score_func=f_regression, k=k_features)))
        else:
            k_features = min(8, max(2, X_train.shape[1]))
            steps.append(("feature_selection", SelectKBest(score_func=f_classif, k=k_features)))

    if task == "Regression":
        estimator = (
            GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=3)
            if model_choice == "Gradient Boosting Regressor"
            else RandomForestRegressor(n_estimators=250, random_state=42, max_depth=8, min_samples_leaf=2)
        )
    else:
        estimator = (
            RandomForestClassifier(n_estimators=250, random_state=42, max_depth=8, min_samples_leaf=2)
            if model_choice == "Random Forest Classifier"
            else LogisticRegression(max_iter=2000)
        )

    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def make_supervised_timeseries(data: pd.Series, n_lags: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    values = pd.Series(data).dropna().reset_index(drop=True)
    frame = pd.DataFrame({"y": values})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame = frame.dropna()
    X = frame[[f"lag_{lag}" for lag in range(1, n_lags + 1)]]
    y = frame["y"]
    return X, y


def recursive_forecast(model, history, n_lags: int, horizon: int):
    history = list(pd.Series(history).dropna().values)
    preds = []
    for _ in range(horizon):
        row = np.array(history[-n_lags:][::-1]).reshape(1, -1)
        pred = float(model.predict(row)[0])
        preds.append(pred)
        history.append(pred)
    return preds

# =========================
# Main title
# =========================
st.title("Agensi Nuklear Malaysia Data Analytics Platform")
st.caption("The platform is an interactive data analytics platform designed for exploratory data analysis, data cleaning, predictive modelling, time-series forecasting, anomaly detection, and interactive data exploration. It allows users to upload datasets in various formats, visualize patterns, prepare data for modelling, train machine learning models, and generate forecasts for sequential data. The platform is tailored for practical applications in scientific and industrial data analysis, including sensor data, experimental results, and operational monitoring.")

# =========================
# File Upload
# =========================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload data file",
    type=["csv", "xlsx", "xls", "txt", "data", "dat", "tsv"]
)

sheet_name = None
sep_option = "Auto detect"

if uploaded_file is not None:
    file_lower = uploaded_file.name.lower()

    if file_lower.endswith((".xlsx", ".xls")):
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_name = st.sidebar.selectbox("Select Excel sheet", excel_file.sheet_names)
            uploaded_file.seek(0)
        except Exception as e:
            st.sidebar.warning(f"Unable to read sheet list: {e}")

    if file_lower.endswith((".txt", ".data", ".dat", ".tsv")):
        sep_option = st.sidebar.selectbox(
            "Text/Data delimiter",
            ["Auto detect", "Whitespace", "Comma", "Tab", "Semicolon", "Pipe"]
        )

# =========================
# Main App
# =========================
if uploaded_file is not None:
    try:
        df = load_data(uploaded_file, sheet_name=sheet_name, sep_option=sep_option)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    st.success("Dataset uploaded successfully.")

    numeric_cols, datetime_cols, categorical_cols = get_column_groups(df)

    tab_eda, tab_clean, tab_ml, tab_ts, tab_anomaly = st.tabs([
        "1. EDA",
        "2. Cleaning & Feature Engineering",
        "3. Machine Learning",
        "4. Time Series",
        "5. Anomaly Detection",
    ])

    # =========================
    # EDA TAB
    # =========================
    with tab_eda:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Dataset Information")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric", len(numeric_cols))
        c4.metric("Missing Values", int(df.isnull().sum().sum()))
        c5.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Column Summary")
        summary = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values,
            "Missing %": (df.isnull().mean().values * 100).round(2),
            "Unique Values": df.nunique(dropna=True).values,
        })
        st.dataframe(summary, use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.subheader("Missing Values by Column")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df = missing_df[missing_df["Missing Values"] > 0]
        if not missing_df.empty:
            fig_missing = px.bar(missing_df, x="Column", y="Missing Values", title="Missing Values Distribution")
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.info("No missing values found.")

        st.subheader("Data Visualization")
        if numeric_cols:
            selected_col = st.selectbox("Select numerical column", numeric_cols, key="eda_num_col")
            fig_hist = px.histogram(df, x=selected_col, nbins=40, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_box = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
            st.plotly_chart(fig_box, use_container_width=True)

            if len(numeric_cols) >= 2:
                x_axis_cols = numeric_cols + datetime_cols
                x_col = st.selectbox("X-axis", x_axis_cols, index=0, key="eda_x")
                y_col = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="eda_y")
                fig_scatter = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig_scatter, use_container_width=True)

                st.subheader("Correlation Heatmap")
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No numerical columns available for visualization.")

        st.subheader("Interactive D-Tale EDA")
        try:
            d = dtale.show(df, subprocess=False, open_browser=False)
            dtale_url = d._main_url
            st.info("Open the interactive D-Tale view below.")
            st.markdown(f"### [Open D-Tale Interactive EDA]({dtale_url})")
            components.iframe(dtale_url, height=850, scrolling=True)
        except Exception as e:
            st.warning(f"D-Tale could not be launched in this environment: {e}")
            st.info("The rest of the dashboard can still be used normally.")

    # =========================
    # CLEANING TAB
    # =========================
    with tab_clean:
        st.subheader("Data Cleaning")
        cleaned_df = df.copy()
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Missing Values**")
            missing_strategy = st.selectbox(
                "Global missing value strategy",
                ["No action", "Drop rows with missing values", "Fill numeric median + categorical mode", "Forward fill", "Backward fill"],
            )
            if missing_strategy == "Drop rows with missing values":
                cleaned_df = cleaned_df.dropna()
            elif missing_strategy == "Fill numeric median + categorical mode":
                for col in cleaned_df.columns:
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    else:
                        mode_val = cleaned_df[col].mode(dropna=True)
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
            elif missing_strategy == "Forward fill":
                cleaned_df = cleaned_df.ffill()
            elif missing_strategy == "Backward fill":
                cleaned_df = cleaned_df.bfill()

        with c2:
            st.write("**Duplicates**")
            if st.checkbox("Remove duplicate rows"):
                before = cleaned_df.shape[0]
                cleaned_df = cleaned_df.drop_duplicates()
                st.success(f"Removed {before - cleaned_df.shape[0]} duplicate rows.")

        st.subheader("Feature Engineering")
        c1, c2 = st.columns(2)
        numeric_cols_clean, _, _ = get_column_groups(cleaned_df)

        with c1:
            if st.checkbox("Normalize numeric columns") and numeric_cols_clean:
                cols_to_normalize = st.multiselect("Columns to normalize", numeric_cols_clean)
                normalize_method = st.selectbox("Normalization method", ["Min-Max (0-1)", "Z-Score Standardization"])
                for col in cols_to_normalize:
                    if normalize_method == "Min-Max (0-1)":
                        denominator = cleaned_df[col].max() - cleaned_df[col].min()
                        cleaned_df[f"{col}_minmax"] = 0 if denominator == 0 else (cleaned_df[col] - cleaned_df[col].min()) / denominator
                    else:
                        std_val = cleaned_df[col].std()
                        cleaned_df[f"{col}_zscore"] = 0 if std_val == 0 else (cleaned_df[col] - cleaned_df[col].mean()) / std_val

        with c2:
            if st.checkbox("Create interaction features") and len(numeric_cols_clean) >= 2:
                col_1 = st.selectbox("First column", numeric_cols_clean, key="clean_interaction_1")
                col_2 = st.selectbox("Second column", numeric_cols_clean, key="clean_interaction_2", index=1)
                interaction_type = st.selectbox("Interaction type", ["Multiply", "Add", "Subtract", "Divide"])
                if interaction_type == "Multiply":
                    cleaned_df[f"{col_1}_x_{col_2}"] = cleaned_df[col_1] * cleaned_df[col_2]
                elif interaction_type == "Add":
                    cleaned_df[f"{col_1}_add_{col_2}"] = cleaned_df[col_1] + cleaned_df[col_2]
                elif interaction_type == "Subtract":
                    cleaned_df[f"{col_1}_sub_{col_2}"] = cleaned_df[col_1] - cleaned_df[col_2]
                elif interaction_type == "Divide":
                    cleaned_df[f"{col_1}_div_{col_2}"] = cleaned_df[col_1] / cleaned_df[col_2].replace(0, np.nan)

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(100), use_container_width=True)
        cleaned_filename = f"cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            label="Download Cleaned Dataset as CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name=cleaned_filename,
            mime="text/csv",
        )

    # =========================
    # MACHINE LEARNING TAB
    # =========================
    with tab_ml:
        st.subheader("Machine Learning Modelling")
        st.write("Use this section for supervised prediction when your dataset has a clear target column.")

        if df.shape[0] < 20 or df.shape[1] < 2:
            st.warning("Need at least 20 rows and 2 columns for reliable machine learning modelling.")
        else:
            target_col = st.selectbox("Select target column", df.columns, key="ml_target")
            feature_cols = st.multiselect(
                "Select feature columns",
                [c for c in df.columns if c != target_col],
                default=[c for c in df.columns if c != target_col][: min(5, df.shape[1]-1)],
                key="ml_features"
            )

            task_type = st.selectbox("Task type", ["Auto detect", "Regression", "Classification"])
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)

            if feature_cols and st.button("Train Machine Learning Model"):
                ml_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
                if ml_df.empty:
                    st.error("No rows remain after removing missing target values.")
                    st.stop()

                X = prepare_feature_frame(ml_df[feature_cols])
                y = ml_df[target_col]

                if task_type == "Auto detect":
                    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 10:
                        task = "Regression"
                    else:
                        task = "Classification"
                else:
                    task = task_type

                if task == "Classification":
                    if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y.astype(str))
                    else:
                        le = None
                        y_encoded = y.astype(float)

                    if pd.Series(y_encoded).nunique() < 2:
                        st.error("The selected target must have at least two classes for classification.")
                        st.stop()

                    stratify = y_encoded if pd.Series(y_encoded).nunique() > 1 and pd.Series(y_encoded).value_counts().min() >= 2 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=42, stratify=stratify
                    )
                    model_choice = st.selectbox("Classification algorithm", ["Random Forest Classifier", "Logistic Regression"])
                    pipe = build_model_pipeline(X_train, "Classification", model_choice)
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                    st.success("Classification model trained successfully.")
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    st.text("Classification Report")
                    st.code(classification_report(y_test, y_pred))

                else:
                    y_numeric = pd.to_numeric(y, errors="coerce")
                    if y_numeric.isna().sum() > 0:
                        st.warning(f"{int(y_numeric.isna().sum())} target values were dropped because they could not be converted to numbers.")
                    ml_ready = pd.concat([X, y_numeric.rename("target")], axis=1).dropna(subset=["target"])
                    X_ready = ml_ready.drop(columns=["target"])
                    y_ready = ml_ready["target"]

                    if y_ready.nunique(dropna=True) <= 1:
                        st.error("The selected target has too little variation for regression modelling.")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_ready, y_ready, test_size=test_size, random_state=42
                    )
                    model_choice = st.selectbox("Regression algorithm", ["Random Forest Regressor", "Gradient Boosting Regressor"])
                    pipe = build_model_pipeline(X_train, "Regression", model_choice)
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    st.success("Regression model trained successfully.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{r2:.4f}")
                    c2.metric("MAE", f"{mae:.4f}")
                    c3.metric("RMSE", f"{rmse:.4f}")

                    if r2 < 0.2:
                        st.info("The model is still weak on this dataset. Try a more predictable target column or add more informative features.")

                    result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
                    fig_pred = px.scatter(result_df, x="Actual", y="Predicted", title="Actual vs Predicted")
                    st.plotly_chart(fig_pred, use_container_width=True)
                    st.dataframe(result_df.head(100), use_container_width=True)

    # =========================
    # TIME SERIES TAB
    # =========================
    with tab_ts:
        st.subheader("Time-Series Analysis and Forecasting")
        if not numeric_cols:
            st.warning("No numeric column available for time-series forecasting.")
        else:
            time_index_mode = st.radio(
                "Time/index source",
                ["Use row number as sequence", "Use datetime column", "Use numeric x-axis column"],
                horizontal=True,
            )
            date_col = None
            x_col = None
            if time_index_mode == "Use datetime column":
                if datetime_cols:
                    date_col = st.selectbox("Select datetime column", datetime_cols)
                else:
                    st.warning("No datetime column detected. Try using row number or numeric x-axis.")
            elif time_index_mode == "Use numeric x-axis column":
                x_col = st.selectbox("Select numeric x-axis column", numeric_cols, key="ts_x_col")

            value_col = st.selectbox("Select value column to forecast", numeric_cols, key="ts_value_col")
            horizon = st.slider("Forecast horizon", 1, 100, 24)
            n_lags = st.slider("Number of lag features", 2, 30, 5)
            model_type = st.selectbox("Forecasting method", ["Lag Regression - Random Forest", "Lag Regression - Linear Regression", "Exponential Smoothing"])

            ts_df = df[[value_col] + ([date_col] if date_col else []) + ([x_col] if x_col else [])].dropna().copy()
            if date_col:
                ts_df = ts_df.sort_values(date_col)
                plot_x = date_col
            elif x_col:
                ts_df = ts_df.sort_values(x_col)
                plot_x = x_col
            else:
                ts_df["sequence_index"] = np.arange(len(ts_df))
                plot_x = "sequence_index"

            fig_ts = px.line(ts_df, x=plot_x, y=value_col, title=f"Time-Series Plot: {value_col}")
            st.plotly_chart(fig_ts, use_container_width=True)

            if st.button("Run Forecast"):
                series = ts_df[value_col].astype(float).dropna()
                if len(series) <= n_lags + 10:
                    st.warning("Not enough observations for selected lag setting. Reduce lag features or upload more data.")
                else:
                    if model_type == "Exponential Smoothing":
                        if not STATSMODELS_AVAILABLE:
                            st.error("statsmodels is not installed. Install it using: pip install statsmodels")
                        else:
                            model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
                            forecast = model.forecast(horizon).tolist()
                    else:
                        X_ts, y_ts = make_supervised_timeseries(series, n_lags=n_lags)
                        estimator = RandomForestRegressor(n_estimators=200, random_state=42) if "Random Forest" in model_type else LinearRegression()
                        estimator.fit(X_ts, y_ts)
                        forecast = recursive_forecast(estimator, series, n_lags, horizon)

                    future_index = np.arange(len(ts_df), len(ts_df) + horizon)
                    forecast_df = pd.DataFrame({"forecast_step": future_index, "forecast": forecast})

                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=np.arange(len(series)), y=series, mode="lines", name="Actual"))
                    fig_forecast.add_trace(go.Scatter(x=future_index, y=forecast, mode="lines+markers", name="Forecast"))
                    fig_forecast.update_layout(title=f"Forecast for {value_col}", xaxis_title="Sequence", yaxis_title=value_col)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    st.dataframe(forecast_df, use_container_width=True)
                    st.download_button(
                        "Download Forecast CSV",
                        data=forecast_df.to_csv(index=False).encode("utf-8"),
                        file_name="forecast_results.csv",
                        mime="text/csv",
                    )

    # =========================
    # ANOMALY DETECTION TAB
    # =========================
    with tab_anomaly:
        st.subheader("Anomaly Detection")
        if not numeric_cols:
            st.warning("No numeric columns available for anomaly detection.")
        else:
            anomaly_cols = st.multiselect(
                "Select numeric columns for anomaly detection",
                numeric_cols,
                default=numeric_cols[: min(3, len(numeric_cols))]
            )
            algorithm = st.selectbox("Anomaly detection algorithm", ["Isolation Forest", "Z-Score", "IQR", "Local Outlier Factor"])
            contamination = st.slider("Expected anomaly proportion", 0.01, 0.30, 0.05, 0.01)

            if anomaly_cols and st.button("Detect Anomalies"):
                anomaly_df = df.copy()
                X_anom = anomaly_df[anomaly_cols].replace([np.inf, -np.inf], np.nan)
                X_anom = X_anom.fillna(X_anom.median(numeric_only=True))

                if algorithm == "Isolation Forest":
                    detector = IsolationForest(contamination=contamination, random_state=42)
                    labels = detector.fit_predict(X_anom)
                    anomaly_df["anomaly"] = np.where(labels == -1, 1, 0)
                    scores = detector.decision_function(X_anom)
                    anomaly_df["anomaly_score"] = -scores
                elif algorithm == "Local Outlier Factor":
                    if not LOF_AVAILABLE:
                        st.error("LocalOutlierFactor is not available in your scikit-learn installation.")
                        st.stop()
                    detector = LocalOutlierFactor(n_neighbors=min(20, max(2, len(X_anom)-1)), contamination=contamination)
                    labels = detector.fit_predict(X_anom)
                    anomaly_df["anomaly"] = np.where(labels == -1, 1, 0)
                    anomaly_df["anomaly_score"] = -detector.negative_outlier_factor_
                elif algorithm == "Z-Score":
                    z = np.abs((X_anom - X_anom.mean()) / X_anom.std(ddof=0).replace(0, np.nan))
                    anomaly_df["anomaly_score"] = z.max(axis=1)
                    anomaly_df["anomaly"] = (anomaly_df["anomaly_score"] > 3).astype(int)
                else:
                    flags = []
                    score_parts = []
                    for col in anomaly_cols:
                        q1 = X_anom[col].quantile(0.25)
                        q3 = X_anom[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        flag = ((X_anom[col] < lower) | (X_anom[col] > upper)).astype(int)
                        flags.append(flag)
                        score_parts.append(np.maximum(lower - X_anom[col], X_anom[col] - upper).clip(lower=0))
                    anomaly_df["anomaly"] = pd.concat(flags, axis=1).max(axis=1)
                    anomaly_df["anomaly_score"] = pd.concat(score_parts, axis=1).max(axis=1)

                anomaly_count = int(anomaly_df["anomaly"].sum())
                st.metric("Detected anomalies", anomaly_count)
                st.dataframe(anomaly_df[anomaly_df["anomaly"] == 1].head(200), use_container_width=True)

                if len(anomaly_cols) >= 2:
                    fig_anom = px.scatter(
                        anomaly_df,
                        x=anomaly_cols[0],
                        y=anomaly_cols[1],
                        color=anomaly_df["anomaly"].astype(str),
                        title="Anomaly Detection Scatter Plot",
                    )
                    st.plotly_chart(fig_anom, use_container_width=True)
                else:
                    plot_df = anomaly_df.reset_index().rename(columns={"index": "row_index"})
                    fig_anom = px.scatter(
                        plot_df,
                        x="row_index",
                        y=anomaly_cols[0],
                        color=plot_df["anomaly"].astype(str),
                        title="Anomaly Detection by Row Index",
                    )
                    st.plotly_chart(fig_anom, use_container_width=True)

                st.download_button(
                    "Download Anomaly Results CSV",
                    data=anomaly_df.to_csv(index=False).encode("utf-8"),
                    file_name="anomaly_detection_results.csv",
                    mime="text/csv",
                )

else:
    st.info("Please upload CSV, Excel, TXT, DATA, DAT or TSV dataset from the sidebar.")
    st.markdown(
        """
        **Supported formats:** `.csv`, `.xlsx`, `.xls`, `.txt`, `.data`, `.dat`, `.tsv`  
        **Suggested Agensi Nuklear Malaysia workflow:** upload sensor/simulation data, review EDA, clean data, run forecasting for sequential signals, and detect abnormal readings.
        """
    )
