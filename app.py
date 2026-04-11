import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import importlib.util
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Initialize session state to prevent re-running ──
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyCity Profit Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #1a1f36;
    color: #fff;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stRadio > label { 
    font-weight: 600; font-size: 0.8rem; letter-spacing: 0.08em; color: #94a3b8 !important;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 8px 14px;
    border-radius: 8px;
    transition: background 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #2d3654 !important;
  }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  div[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 0.82rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.7rem !important; font-weight: 700; }

  /* ── Section header ── */
  .section-header {
    background: linear-gradient(135deg, #1a56db 0%, #1e429f 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
  }
  .section-header h2 { margin: 0; font-size: 1.5rem; font-weight: 700; }
  .section-header p  { margin: 4px 0 0; font-size: 0.9rem; opacity: 0.85; }

  /* ── Card ── */
  .info-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .info-card h4 { margin: 0 0 6px; color: #1a56db; font-size: 1rem; }
  .info-card p  { margin: 0; color: #374151; font-size: 0.88rem; line-height: 1.5; }

  /* ── Insight badge ── */
  .badge-green  { background:#d1fae5; color:#065f46; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-yellow { background:#fef3c7; color:#92400e; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-red    { background:#fee2e2; color:#991b1b; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-blue   { background:#dbeafe; color:#1e40af; padding:4px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }

  /* ── Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Buttons ── */
  .stButton > button {
    background: #1a56db;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
  }
  .stButton > button:hover { background: #1e429f; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; font-weight: 600; }
  .stTabs [aria-selected="true"] { background: #dbeafe; color: #1e40af; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────────────
DATA_FILENAME = "SkyCity Auckland Restaurants & Bars - SkyCity Auckland Restaurants & Bars.csv"
_local_data_path = Path(__file__).resolve().parent / DATA_FILENAME
_container_data_path = Path("/app/main") / DATA_FILENAME
_root_container_path = Path("/app") / DATA_FILENAME

# Try multiple paths for Streamlit Cloud compatibility
if _local_data_path.exists():
    DATA_PATH = _local_data_path
elif _container_data_path.exists():
    DATA_PATH = _container_data_path
elif _root_container_path.exists():
    DATA_PATH = _root_container_path
else:
    DATA_PATH = _local_data_path  # fallback, will error with clear message

MODEL_DIR  = Path(__file__).resolve().parent / "models"


def save_models_to_disk(results, trained, sc, FEATURES):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"results": results, "trained": trained, "sc": sc, "FEATURES": FEATURES},
        MODEL_DIR / "trained_models.pkl",
    )


def load_models_from_disk():
    path = MODEL_DIR / "trained_models.pkl"
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=True)
def load_data():
    if not DATA_PATH.exists():
        st.error(f"❌ **Data file not found!**\n\nExpected file at: `{DATA_PATH}`\n\nPlease ensure the CSV file is in the repository root or at `/app/main/`")
        st.stop()
    
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"❌ **Error loading data file:** {str(e)}\n\nFile path: `{DATA_PATH}`")
        st.stop()
    
    # Derived features
    df["TotalRevenue"]   = df["InStoreRevenue"] + df["UberEatsRevenue"] + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"]
    df["TotalNetProfit"] = df["InStoreNetProfit"] + df["UberEatsNetProfit"] + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"]
    df["NetProfitPerOrder"] = df["TotalNetProfit"] / df["MonthlyOrders"]
    df["OverallMargin"]     = df["TotalNetProfit"] / df["TotalRevenue"]

    # Interaction terms
    df["Commission_UE"]   = df["CommissionRate"] * df["UE_share"]
    df["DeliveryCost_SD"] = df["DeliveryCostPerOrder"] * df["SD_share"]
    df["GrowthAdj_Orders"] = df["MonthlyOrders"] * df["GrowthFactor"]
    df["CostToRevenue"]   = (df["COGSRate"] + df["OPEXRate"])
    df["ChannelMixScore"] = (df["InStoreShare"] * df["InStoreNetProfit"] + df["UE_share"] * df["UberEatsNetProfit"]) / (df["TotalNetProfit"].replace(0, 1))

    # Revenue-ratio & per-order features
    _total_rev = df["TotalRevenue"].replace(0, np.nan)
    df["InStoreRevRatio"]  = df["InStoreRevenue"]     / _total_rev
    df["UE_RevRatio"]      = df["UberEatsRevenue"]    / _total_rev
    df["DD_RevRatio"]      = df["DoorDashRevenue"]    / _total_rev
    df["SD_RevRatio"]      = df["SelfDeliveryRevenue"]/ _total_rev
    df["RevenuePerOrder"]  = df["TotalRevenue"]        / df["MonthlyOrders"]
    df["ProfitPerOrder"]   = df["TotalNetProfit"]      / df["MonthlyOrders"]
    df[["InStoreRevRatio", "UE_RevRatio", "DD_RevRatio", "SD_RevRatio",
        "RevenuePerOrder", "ProfitPerOrder"]] = \
        df[["InStoreRevRatio", "UE_RevRatio", "DD_RevRatio", "SD_RevRatio",
            "RevenuePerOrder", "ProfitPerOrder"]].fillna(0)

    # Outlier flagging — 3×IQR on TotalNetProfit
    _q1, _q3 = df["TotalNetProfit"].quantile(0.25), df["TotalNetProfit"].quantile(0.75)
    _iqr = _q3 - _q1
    df["IsOutlier"] = (
        (df["TotalNetProfit"] < _q1 - 3 * _iqr) |
        (df["TotalNetProfit"] > _q3 + 3 * _iqr)
    )

    return df


@st.cache_data
def prepare_ml(df):
    le = LabelEncoder()
    df2 = df.copy()
    for col in ["CuisineType", "Segment", "Subregion"]:
        df2[col + "_enc"] = le.fit_transform(df2[col])

    FEATURES = [
        "InStoreShare", "UE_share", "DD_share", "SD_share",
        "CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM", "GrowthFactor",
        "AOV", "MonthlyOrders", "COGSRate", "OPEXRate",
        "Commission_UE", "DeliveryCost_SD", "GrowthAdj_Orders", "CostToRevenue",
        "InStoreRevRatio", "UE_RevRatio", "DD_RevRatio", "SD_RevRatio",
        "RevenuePerOrder", "ProfitPerOrder",
        "CuisineType_enc", "Segment_enc", "Subregion_enc",
    ]
    TARGET = "TotalNetProfit"
    X = df2[FEATURES].fillna(0)  # Fill any NaN values with 0
    y = df2[TARGET].fillna(0)     # Fill any NaN values with 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES


def train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression":      LinearRegression(),
        "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost":                xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }
    results, trained = {}, {}
    for name, m in models.items():
        if name == "Linear Regression":
            m.fit(X_tr_s, y_train)
            pred = m.predict(X_te_s)
            cv_rmse = -cross_val_score(LinearRegression(), X_tr_s, y_train, cv=5,
                                       scoring="neg_root_mean_squared_error")
            cv_r2 = cross_val_score(LinearRegression(), X_tr_s, y_train, cv=5, scoring="r2")
            cv_mae = -cross_val_score(LinearRegression(), X_tr_s, y_train, cv=5,
                                      scoring="neg_mean_absolute_error")
        else:
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            cv_rmse = -cross_val_score(m, X_train, y_train, cv=5,
                                       scoring="neg_root_mean_squared_error")
            cv_r2 = cross_val_score(m, X_train, y_train, cv=5, scoring="r2")
            cv_mae = -cross_val_score(m, X_train, y_train, cv=5,
                                      scoring="neg_mean_absolute_error")
        residuals = y_test.to_numpy() - pred
        residual_low, residual_high = np.quantile(residuals, [0.025, 0.975])
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        mae  = mean_absolute_error(y_test, pred)
        results[name] = {
            "RMSE": rmse,
            "R2": r2,
            "MAE": mae,
            "pred": pred,
            "residuals": residuals,
            "residual_low": residual_low,
            "residual_high": residual_high,
            "CV_RMSE_mean": cv_rmse.mean(),
            "CV_RMSE_std": cv_rmse.std(),
            "CV_R2_mean": cv_r2.mean(),
            "CV_R2_std": cv_r2.std(),
            "CV_MAE_mean": cv_mae.mean(),
            "CV_MAE_std": cv_mae.std(),
        }
        trained[name] = m
    return results, trained


def train_quantile_models(X_train, y_train):
    models = {
        "lower": GradientBoostingRegressor(
            loss="quantile", alpha=0.025, n_estimators=150, random_state=42
        ),
        "median": GradientBoostingRegressor(
            loss="quantile", alpha=0.50, n_estimators=150, random_state=42
        ),
        "upper": GradientBoostingRegressor(
            loss="quantile", alpha=0.975, n_estimators=150, random_state=42
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


@st.cache_data
def prepare_ml_target(df, target):
    """Prepare ML split for a secondary prediction target, excluding outliers."""
    le  = LabelEncoder()
    df2 = df[~df["IsOutlier"]].copy()
    for col in ["CuisineType", "Segment", "Subregion"]:
        df2[col + "_enc"] = le.fit_transform(df2[col])
    FEATURES = [
        "InStoreShare", "UE_share", "DD_share", "SD_share",
        "CommissionRate", "DeliveryCostPerOrder", "DeliveryRadiusKM", "GrowthFactor",
        "AOV", "MonthlyOrders", "COGSRate", "OPEXRate",
        "Commission_UE", "DeliveryCost_SD", "GrowthAdj_Orders", "CostToRevenue",
        "InStoreRevRatio", "UE_RevRatio", "DD_RevRatio", "SD_RevRatio",
        "RevenuePerOrder", "ProfitPerOrder",
        "CuisineType_enc", "Segment_enc", "Subregion_enc",
    ]
    X = df2[FEATURES].fillna(0)  # Fill any NaN values with 0
    y = df2[target].fillna(0)     # Fill any NaN values with 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES


def train_secondary_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test):
    """Train RF and XGBoost for a secondary prediction target."""
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":       xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    }
    sec_results, sec_trained = {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        sec_results[name] = {
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2":   r2_score(y_test, pred),
            "MAE":  mean_absolute_error(y_test, pred),
            "pred": pred,
        }
        sec_trained[name] = m
    return sec_results, sec_trained


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="info-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def prediction_interval(prediction, result):
    return prediction + result["residual_low"], prediction + result["residual_high"]


def encode_profile_values(df, cuisine, segment, subregion):
    le_vals = {column: sorted(df[column].unique()) for column in ["CuisineType", "Segment", "Subregion"]}
    return (
        le_vals["CuisineType"].index(cuisine),
        le_vals["Segment"].index(segment),
        le_vals["Subregion"].index(subregion),
    )


def build_feature_row(
    df,
    in_store,
    ue_share,
    dd_share,
    sd_share,
    commission,
    delivery_cost,
    radius,
    growth,
    aov,
    orders,
    cogs,
    opex,
    cuisine,
    segment,
    subregion,
):
    cuisine_enc, segment_enc, subregion_enc = encode_profile_values(df, cuisine, segment, subregion)
    commission_ue = commission * ue_share
    delivery_sd = delivery_cost * sd_share
    growth_orders = orders * growth
    cost_rev = cogs + opex

    total_revenue = aov * orders
    revenue_instore = total_revenue * in_store
    revenue_ue = total_revenue * ue_share
    revenue_dd = total_revenue * dd_share
    revenue_sd = total_revenue * sd_share

    revenue_per_order = aov
    profit_per_order = (
        total_revenue * (1 - cogs - opex)
        - revenue_ue * commission
        - revenue_dd * commission
        - delivery_cost * orders * sd_share
    ) / orders if orders else 0

    return np.array([[ 
        in_store, ue_share, dd_share, sd_share,
        commission, delivery_cost, radius, growth,
        aov, orders, cogs, opex,
        commission_ue, delivery_sd, growth_orders, cost_rev,
        revenue_instore / total_revenue if total_revenue else 0,
        revenue_ue / total_revenue if total_revenue else 0,
        revenue_dd / total_revenue if total_revenue else 0,
        revenue_sd / total_revenue if total_revenue else 0,
        revenue_per_order, profit_per_order,
        cuisine_enc, segment_enc, subregion_enc,
    ]])


def predict_target(model_name, model, scaler, scenario_row):
    if model_name == "Linear Regression":
        return float(model.predict(scaler.transform(scenario_row))[0])
    return float(model.predict(scenario_row)[0])


def predict_quantile_interval(quantile_models, scenario_row):
    lower = float(quantile_models["lower"].predict(scenario_row)[0])
    median = float(quantile_models["median"].predict(scenario_row)[0])
    upper = float(quantile_models["upper"].predict(scenario_row)[0])
    return min(lower, upper), median, max(lower, upper)


def scenario_risk_from_interval(lower_bound, upper_bound, prediction):
    width_pct = ((upper_bound - lower_bound) / max(abs(prediction), 1)) * 100
    if lower_bound < 0 < upper_bound or width_pct >= 80:
        return "High", width_pct
    if width_pct >= 40:
        return "Medium", width_pct
    return "Low", width_pct


def optimize_channel_mix(
    df,
    trained,
    scaler,
    best_model_name,
    commission,
    delivery_cost,
    radius,
    growth,
    aov,
    orders,
    cogs,
    opex,
    cuisine,
    segment,
    subregion,
):
    share_columns = ["InStoreShare", "UE_share", "DD_share", "SD_share"]
    bounds = []
    initial_guess = []
    for column in share_columns:
        lower = float(df[column].quantile(0.10))
        upper = float(df[column].quantile(0.90))
        bounds.append((lower, upper))
        initial_guess.append(float(df[column].mean()))

    initial_guess = np.array(initial_guess)
    initial_guess = initial_guess / initial_guess.sum()

    def objective(shares):
        row = build_feature_row(
            df,
            shares[0], shares[1], shares[2], shares[3],
            commission, delivery_cost, radius, growth,
            aov, orders, cogs, opex,
            cuisine, segment, subregion,
        )
        return -predict_target(best_model_name, trained[best_model_name], scaler, row)

    result = minimize(
        objective,
        x0=initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda shares: np.sum(shares) - 1}],
        options={"maxiter": 200, "ftol": 1e-6},
    )
    return result, bounds


def scenario_risk_label(lower_bound, upper_bound, prediction):
    return scenario_risk_from_interval(lower_bound, upper_bound, prediction)


PALETTE = ["#1a56db", "#0e9f6e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#10b981"]
HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
OPTIONAL_OLS     = "ols" if HAS_STATSMODELS else None
HAS_FPDF         = importlib.util.find_spec("fpdf") is not None


def model_results_complete(results):
    if not results:
        return False
    required_keys = {
        "CV_RMSE_mean", "CV_RMSE_std", "CV_R2_mean", "CV_R2_std",
        "CV_MAE_mean", "CV_MAE_std", "residual_low", "residual_high",
    }
    return all(required_keys.issubset(result.keys()) for result in results.values())


def generate_scenario_pdf(
    preds_all, breakdown, results, best_model_name, risk_level,
    best_lower, best_upper,
    in_store, ue_share, dd_share, sd_share,
    commission, delivery_cost, radius, growth,
    aov, orders, cogs, opex,
    cuisine, segment, subregion,
):
    """Build a PDF scenario report and return bytes. Returns None if fpdf2 unavailable."""
    if not HAS_FPDF:
        return None
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "SkyCity Profit Optimizer -- Scenario Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    # Scenario parameters
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scenario Parameters", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for label, value in [
        ("In-Store Share", f"{in_store:.0%}"), ("Uber Eats Share", f"{ue_share:.0%}"),
        ("DoorDash Share", f"{dd_share:.0%}"), ("Self-Delivery Share", f"{sd_share:.0%}"),
        ("Commission Rate", f"{commission:.1%}"), ("Delivery Cost/Order", f"${delivery_cost:.2f}"),
        ("Delivery Radius", f"{radius} km"), ("Growth Factor", f"{growth:.3f}"),
        ("AOV", f"${aov:.2f}"), ("Monthly Orders", str(orders)),
        ("COGS Rate", f"{cogs:.1%}"), ("OPEX Rate", f"{opex:.1%}"),
        ("Cuisine Type", cuisine), ("Segment", segment),
        ("Subregion", subregion), ("Scenario Risk", risk_level),
    ]:
        pdf.cell(75, 6, label, border=0)
        pdf.cell(0, 6, str(value), ln=True)
    pdf.ln(4)

    # Model forecasts table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Model Forecasts", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    for hdr, w in [("Model", 58), ("Predicted ($)", 38), ("P2.5 ($)", 35), ("P97.5 ($)", 35)]:
        pdf.cell(w, 7, hdr, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for name, p in preds_all.items():
        lo, hi = prediction_interval(p, results[name])
        pdf.cell(58, 6, name + (" *" if name == best_model_name else ""), border=1)
        pdf.cell(38, 6, f"${p:,.0f}", border=1)
        pdf.cell(35, 6, f"${lo:,.0f}", border=1)
        pdf.cell(35, 6, f"${hi:,.0f}", border=1)
        pdf.ln()
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "* Best model by R². P2.5/P97.5 are 95% prediction intervals.", ln=True)
    pdf.ln(4)

    # Channel breakdown table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Channel Breakdown", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    for hdr, w in [("Channel", 45), ("Revenue ($)", 43), ("Net Profit ($)", 43), ("Margin", 35)]:
        pdf.cell(w, 7, hdr, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, row in breakdown.iterrows():
        pdf.cell(45, 6, str(row["Channel"]), border=1)
        pdf.cell(43, 6, f"${row['Revenue ($)']:,.0f}", border=1)
        pdf.cell(43, 6, f"${row['Net Profit ($)']:,.0f}", border=1)
        pdf.cell(35, 6, f"{row['Margin']:.1%}", border=1)
        pdf.ln()

    return bytes(pdf.output())


def save_dataframe_as_table_image(df, filepath, width=1200, height=420):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color="#1a56db", font_color="white", align="left"),
        cells=dict(values=[df[col] for col in df.columns], fill_color="white", align="left"),
    )])
    fig.update_layout(width=width, height=height, margin=dict(l=5, r=5, t=5, b=5))
    fig.write_image(filepath)


def export_analytics_images(df, results=None, y_test=None, quantile_models=None, output_dir="figures"):
    Path(output_dir).mkdir(exist_ok=True)

    # Exploratory charts
    cuisine_rev = df.groupby("CuisineType")["TotalRevenue"].mean().sort_values()
    fig = px.bar(cuisine_rev, orientation="h", title="Avg Monthly Revenue by Cuisine Type",
                 labels={"value": "Revenue ($)", "index": ""}, color=cuisine_rev.values,
                 color_continuous_scale=["#dbeafe", "#1a56db"])
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
    fig.write_image(Path(output_dir) / "cuisine_revenue.png")

    seg_profit = df.groupby("Segment")["TotalNetProfit"].mean().sort_values()
    fig2 = px.bar(seg_profit, title="Avg Monthly Net Profit by Segment",
                  labels={"value": "Net Profit ($)", "index": ""},
                  color=seg_profit.index, color_discrete_sequence=["#1a56db", "#0e9f6e", "#f59e0b", "#8b5cf6"])
    fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0))
    fig2.write_image(Path(output_dir) / "segment_profit.png")

    ch_vals = [df["InStoreShare"].mean(), df["UE_share"].mean(), df["DD_share"].mean(), df["SD_share"].mean()]
    ch_labels = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
    fig3 = go.Figure(go.Pie(labels=ch_labels, values=ch_vals, hole=0.52,
                            marker_colors=["#1a56db", "#0e9f6e", "#f59e0b", "#0e9f6e"]))
    fig3.update_layout(title="Average Channel Mix", margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="white")
    fig3.write_image(Path(output_dir) / "channel_mix.png")

    if results is not None:
        comp_df = pd.DataFrame({
            "Model": list(results.keys()),
            "RMSE ($)": [v["RMSE"] for v in results.values()],
            "R² Score": [v["R2"] for v in results.values()],
            "MAE ($)": [v["MAE"] for v in results.values()],
        })
        save_dataframe_as_table_image(comp_df.round(4), Path(output_dir) / "model_performance_table.png")

        fig4 = px.bar(comp_df, x="Model", y="RMSE ($)", title="RMSE by Model")
        fig4.write_image(Path(output_dir) / "model_rmse.png")
        fig4b = px.bar(comp_df, x="Model", y="R² Score", title="R² by Model")
        fig4b.write_image(Path(output_dir) / "model_r2.png")

        if y_test is not None:
            best_model_name = max(results, key=lambda k: results[k]["R2"])
            pred_vals = results[best_model_name]["pred"]
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=y_test.values, y=pred_vals, mode="markers",
                                      marker=dict(color="#1a56db", opacity=0.6, size=4), name="Pred"))
            mn, mx = float(y_test.min()), float(y_test.max())
            fig5.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                      line=dict(color="#ef4444", dash="dash"), name="Ideal"))
            fig5.update_layout(title="Predicted vs Actual (best model)",
                               xaxis_title="Actual", yaxis_title="Predicted",
                               plot_bgcolor="white", paper_bgcolor="white")
            fig5.write_image(Path(output_dir) / "pred_vs_actual.png")

    return output_dir


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SkyCity Analytics")
    st.markdown("**Profit Optimization Platform**")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Exploratory Analysis", "Predictive Models", "What-If Simulator", "Optimization Panel"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<small style="color:#64748b">Dataset: SkyCity Auckland<br>Records: 1,696 | Restaurants: ~212<br>Updated: 2024</small>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────────────────
df = load_data()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    section_header("Predictive Modeling and Profit Optimization for Multi-Channel Restaurant Operations", "High-level summary of SkyCity Auckland restaurant operations")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Restaurants",    f"{df['RestaurantID'].nunique():,}")
    c2.metric("Avg Monthly Orders",   f"{df['MonthlyOrders'].mean():,.0f}")
    c3.metric("Avg Monthly Profit",   f"${df['TotalNetProfit'].mean():,.0f}")
    c4.metric("Avg Net Margin",       f"{df['OverallMargin'].mean()*100:.1f}%")
    c5.metric("Avg Commission Rate",  f"{df['CommissionRate'].mean()*100:.1f}%")

    _n_out = int(df["IsOutlier"].sum())
    if _n_out > 0:
        st.info(
            f"**Data quality:** {_n_out} records ({_n_out / len(df) * 100:.1f}%) identified as "
            f"extreme-profit outliers (>3×IQR). Included in all visualizations; "
            f"excluded from secondary ML target training."
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Revenue by cuisine
        cuisine_rev = df.groupby("CuisineType")["TotalRevenue"].mean().sort_values()
        fig = px.bar(cuisine_rev, orientation="h",
                     title="Avg Monthly Revenue by Cuisine Type",
                     labels={"value": "Revenue ($)", "index": ""},
                     color=cuisine_rev.values,
                     color_continuous_scale=["#dbeafe", "#1a56db"])
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Profit by segment
        seg_profit = df.groupby("Segment")["TotalNetProfit"].mean().sort_values()
        colors = ["#1a56db", "#0e9f6e", "#f59e0b", "#8b5cf6"]
        fig2 = px.bar(seg_profit, title="Avg Monthly Net Profit by Segment",
                      labels={"value": "Net Profit ($)", "index": ""},
                      color=seg_profit.index, color_discrete_sequence=colors)
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Channel share donut
        ch_vals = [df["InStoreShare"].mean(), df["UE_share"].mean(),
                   df["DD_share"].mean(), df["SD_share"].mean()]
        ch_labels = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
        fig3 = go.Figure(go.Pie(labels=ch_labels, values=ch_vals, hole=0.52,
                                marker_colors=["#1a56db", "#0e9f6e", "#f59e0b", "#8b5cf6"]))
        fig3.update_layout(title="Average Channel Mix", margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Subregion comparison
        sub = df.groupby("Subregion")[["TotalRevenue", "TotalNetProfit"]].mean().reset_index()
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Revenue", x=sub["Subregion"], y=sub["TotalRevenue"],
                              marker_color="#dbeafe"))
        fig4.add_trace(go.Bar(name="Net Profit", x=sub["Subregion"], y=sub["TotalNetProfit"],
                              marker_color="#1a56db"))
        fig4.update_layout(title="Revenue vs Net Profit by Subregion", barmode="group",
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Key Observations")
    cols = st.columns(3)
    with cols[0]:
        top_cuisine = df.groupby("CuisineType")["TotalNetProfit"].mean().idxmax()
        card("Best Performing Cuisine",
             f"<b>{top_cuisine}</b> leads in average net profit across all segments and subregions.")
    with cols[1]:
        top_channel = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"][
            [df["InStoreNetProfit"].mean(), df["UberEatsNetProfit"].mean(),
             df["DoorDashNetProfit"].mean(), df["SelfDeliveryNetProfit"].mean()].index(
                max(df["InStoreNetProfit"].mean(), df["UberEatsNetProfit"].mean(),
                    df["DoorDashNetProfit"].mean(), df["SelfDeliveryNetProfit"].mean()))]
        card("Most Profitable Channel",
             f"<b>{top_channel}</b> generates the highest average net profit contribution.")
    with cols[2]:
        high_comm = df[df["CommissionRate"] > 0.30]["TotalNetProfit"].mean()
        low_comm  = df[df["CommissionRate"] <= 0.30]["TotalNetProfit"].mean()
        delta = ((low_comm - high_comm) / abs(high_comm)) * 100 if high_comm != 0 else 0
        card("Commission Impact",
             f"Restaurants with commission ≤30% earn <b>{abs(delta):.1f}% {'more' if delta > 0 else 'less'}</b> net profit than those above 30%.")

    st.markdown("---")
    with st.expander("View Raw Dataset (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    section_header("Exploratory Data Analysis", "Deep-dive into distributions, correlations, and channel dynamics")

    if not HAS_STATSMODELS:
        st.info("Trendlines are disabled because `statsmodels` is not installed in the current environment.")

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution & Revenue", "Cost Analysis", "Channel Dynamics", "Correlation Matrix"])

    # ── Tab 1 ──
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="TotalNetProfit", nbins=50, title="Distribution of Total Net Profit",
                               color_discrete_sequence=["#1a56db"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_title="Net Profit ($)", yaxis_title="Count",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(df, x="CuisineType", y="TotalNetProfit", color="CuisineType",
                          title="Net Profit Distribution by Cuisine",
                          color_discrete_sequence=PALETTE)
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_title="", yaxis_title="Net Profit ($)",
                               margin=dict(l=0, r=0, t=40, b=0))
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df, x="MonthlyOrders", y="TotalNetProfit",
                              color="Segment", size="AOV",
                              title="Orders vs Net Profit (size = AOV)",
                              color_discrete_sequence=PALETTE,
                              hover_data=["RestaurantName", "CuisineType"])
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.violin(df, x="Segment", y="OverallMargin", color="Segment",
                             title="Profit Margin Distribution by Segment",
                             color_discrete_sequence=PALETTE, box=True)
            fig4.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 2 ──
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="CommissionRate", y="UberEatsNetProfit",
                             color="CuisineType", title="Commission Rate vs Uber Eats Net Profit",
                             color_discrete_sequence=PALETTE,
                             trendline=OPTIONAL_OLS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              xaxis_tickformat=".0%",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df, x="DeliveryCostPerOrder", y="SelfDeliveryNetProfit",
                              color="Subregion", title="Delivery Cost/Order vs Self-Delivery Profit",
                              color_discrete_sequence=PALETTE, trendline=OPTIONAL_OLS)
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df, x="COGSRate", y="OPEXRate", color="TotalNetProfit",
                              title="COGS Rate vs OPEX Rate (color = Net Profit)",
                              color_continuous_scale="Blues",
                              hover_data=["RestaurantName"])
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            cost_data = pd.DataFrame({
                "Cost Component": ["COGS", "OPEX", "Commission", "Delivery Cost"],
                "Avg % of Revenue": [
                    df["COGSRate"].mean() * 100,
                    df["OPEXRate"].mean() * 100,
                    df["CommissionRate"].mean() * 100,
                    (df["SD_DeliveryTotalCost"] / df["TotalRevenue"]).mean() * 100,
                ]
            })
            fig4 = px.bar(cost_data, x="Cost Component", y="Avg % of Revenue",
                          title="Average Cost Structure (% of Revenue)",
                          color="Cost Component", color_discrete_sequence=PALETTE)
            fig4.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3 ──
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            ch_profit = pd.DataFrame({
                "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
                "Avg Net Profit": [df["InStoreNetProfit"].mean(), df["UberEatsNetProfit"].mean(),
                                   df["DoorDashNetProfit"].mean(), df["SelfDeliveryNetProfit"].mean()],
                "Avg Revenue":    [df["InStoreRevenue"].mean(), df["UberEatsRevenue"].mean(),
                                   df["DoorDashRevenue"].mean(), df["SelfDeliveryRevenue"].mean()],
            })
            fig = px.bar(ch_profit, x="Channel", y=["Avg Revenue", "Avg Net Profit"],
                         title="Revenue vs Net Profit by Channel", barmode="group",
                         color_discrete_sequence=["#dbeafe", "#1a56db"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            heat_data = df.groupby(["CuisineType", "Segment"])[["TotalNetProfit"]].mean().reset_index()
            heat_pivot = heat_data.pivot(index="CuisineType", columns="Segment", values="TotalNetProfit")
            fig2 = px.imshow(heat_pivot, title="Avg Net Profit Heatmap: Cuisine × Segment",
                             color_continuous_scale="Blues", text_auto=".0f",
                             aspect="auto")
            fig2.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.scatter(df, x="InStoreShare", y="InStoreNetProfit",
                              color="Segment", title="In-Store Share vs In-Store Net Profit",
                              trendline=OPTIONAL_OLS, color_discrete_sequence=PALETTE)
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.scatter(df, x="DeliveryRadiusKM", y="SelfDeliveryNetProfit",
                              color="CuisineType", title="Delivery Radius vs Self-Delivery Profit",
                              trendline=OPTIONAL_OLS, color_discrete_sequence=PALETTE)
            fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 4 ──
    with tab4:
        num_cols = ["TotalNetProfit", "TotalRevenue", "MonthlyOrders", "AOV",
                    "InStoreShare", "UE_share", "DD_share", "SD_share",
                    "COGSRate", "OPEXRate", "CommissionRate",
                    "DeliveryRadiusKM", "DeliveryCostPerOrder", "GrowthFactor"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, title="Pearson Correlation Matrix", text_auto=".2f",
                        color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=60, b=0),
                          width=900, height=600)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTIVE MODELS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Models":
    section_header("Predictive Modeling", "Train and compare 4 ML models for net profit forecasting")

    X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES = prepare_ml(df)
    quantile_models = train_quantile_models(X_train, y_train)

    # Auto-load persisted models if session is fresh
    if "results" not in st.session_state:
        _disk = load_models_from_disk()
        if _disk:
            st.session_state.update({
                "results":  _disk["results"],
                "trained":  _disk["trained"],
                "sc":       _disk["sc"],
                "FEATURES": _disk["FEATURES"],
                "X_test":   X_test,
                "y_test":   y_test,
            })
            st.info("Loaded persisted models from disk. Click **Re-train** to refresh.")

    _btn_col1, _btn_col2 = st.columns([3, 1])
    with _btn_col1:
        _do_train = st.button("Train All 4 Models")
    with _btn_col2:
        _do_retrain = st.button("Re-train (force)")

    if _do_train or _do_retrain:
        with st.spinner("Training Linear Regression, Random Forest, Gradient Boosting & XGBoost..."):
            results, trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state.update({
            "results":  results,
            "trained":  trained,
            "sc":       sc,
            "FEATURES": FEATURES,
            "X_test":   X_test,
            "y_test":   y_test,
        })
        save_models_to_disk(results, trained, sc, FEATURES)
        st.success("Models trained and saved to disk!")

    if "results" not in st.session_state:
        st.info("Click **Train All 4 Models** to begin.")
        st.stop()

    if not model_results_complete(st.session_state["results"]):
        with st.spinner("Refreshing model diagnostics for this session..."):
            refreshed_results, refreshed_trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state["results"] = refreshed_results
        st.session_state["trained"] = refreshed_trained
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

    results = st.session_state["results"]
    y_test_v = st.session_state["y_test"]

    # ── Export images shortcut ──
    if st.button("Export figures for LaTeX/PDF (figures/)"):
        output_path = export_analytics_images(df, results=results, y_test=y_test_v, quantile_models=quantile_models)
        st.success(f"Saved figures and tables to: {output_path}")

    # ── Model comparison table ──
    st.markdown("### Model Performance Comparison")
    comp = pd.DataFrame({
        "Model":   list(results.keys()),
        "RMSE ($)": [f"{v['RMSE']:,.0f}" for v in results.values()],
        "R² Score": [f"{v['R2']:.4f}"   for v in results.values()],
        "MAE ($)":  [f"{v['MAE']:,.0f}"  for v in results.values()],
    })
    best_r2 = max(results, key=lambda k: results[k]["R2"])
    st.dataframe(comp.set_index("Model"), use_container_width=True)
    st.markdown(f'Best model by R²: <span class="badge-green">{best_r2} — R² = {results[best_r2]["R2"]:.4f}</span>', unsafe_allow_html=True)

    st.markdown("### Cross-Validation Summary (5-Fold)")
    cv_comp = pd.DataFrame({
        "Model": list(results.keys()),
        "CV RMSE ($)": [f"{v['CV_RMSE_mean']:,.0f} ± {v['CV_RMSE_std']:,.0f}" for v in results.values()],
        "CV R²": [f"{v['CV_R2_mean']:.4f} ± {v['CV_R2_std']:.4f}" for v in results.values()],
        "CV MAE ($)": [f"{v['CV_MAE_mean']:,.0f} ± {v['CV_MAE_std']:,.0f}" for v in results.values()],
    })
    st.dataframe(cv_comp.set_index("Model"), use_container_width=True)

    st.markdown("---")

    # ── Metric bar charts ──
    col1, col2, col3 = st.columns(3)
    names = list(results.keys())

    with col1:
        fig = px.bar(x=names, y=[results[n]["RMSE"] for n in names],
                     title="RMSE (lower is better)", color=names,
                     color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
        fig.update_xaxes(tickangle=15)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(x=names, y=[results[n]["R2"] for n in names],
                      title="R² Score (higher is better)", color=names,
                      color_discrete_sequence=PALETTE)
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
        fig2.update_xaxes(tickangle=15)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.bar(x=names, y=[results[n]["MAE"] for n in names],
                      title="MAE (lower is better)", color=names,
                      color_discrete_sequence=PALETTE)
        fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_title="", margin=dict(l=0, r=0, t=40, b=0))
        fig3.update_xaxes(tickangle=15)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Predicted vs Actual ──
    st.markdown("### Predicted vs Actual Net Profit")
    selected_model = st.selectbox("Select model", list(results.keys()))
    pred_vals = results[selected_model]["pred"]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=y_test_v.values, y=pred_vals, mode="markers",
                               name="Predictions",
                               marker=dict(color="#1a56db", opacity=0.5, size=5)))
    mn, mx = float(y_test_v.min()), float(y_test_v.max())
    fig4.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                               name="Perfect Prediction",
                               line=dict(color="#ef4444", dash="dash")))
    fig4.update_layout(title=f"{selected_model} — Predicted vs Actual",
                       xaxis_title="Actual Net Profit ($)", yaxis_title="Predicted Net Profit ($)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ── Feature importance (tree models) ──
    st.markdown("### Feature Importance (Permutation-Based)")
    tree_models = {k: v for k, v in st.session_state["trained"].items() if k != "Linear Regression"}
    fi_model = st.selectbox("Select model for feature importance", list(tree_models.keys()), key="fi_sel")
    m = tree_models[fi_model]
    
    # Use permutation importance for scale-independent, more robust ranking
    from sklearn.inspection import permutation_importance
    perm_result = permutation_importance(m, st.session_state["X_test"], st.session_state["y_test"],
                                         n_repeats=10, random_state=42, n_jobs=-1)
    importances = perm_result.importances_mean
    fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance")
    fig5 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                  title=f"Feature Importance — {fi_model} (Permutation-Based)",
                  color="Importance", color_continuous_scale=["#dbeafe", "#1a56db"])
    fig5.update_layout(coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0), height=500)
    fig5.update_xaxes(type="log", title="Importance (log scale)")
    st.caption("Permutation importance: average decrease in model performance when each feature is randomly shuffled (scale-independent, more reliable)")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ── Residual distribution ──
    st.markdown("### Residual Analysis")
    residuals = y_test_v.values - pred_vals
    fig6 = px.histogram(x=residuals, nbins=50, title=f"Residual Distribution — {selected_model}",
                        color_discrete_sequence=["#1a56db"])
    fig6.add_vline(x=0, line_color="red", line_dash="dash")
    fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Residual ($)", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # ── Formal uncertainty modeling ──
    st.markdown("### Formal Uncertainty Modeling")
    q_lower_test = quantile_models["lower"].predict(X_test)
    q_median_test = quantile_models["median"].predict(X_test)
    q_upper_test = quantile_models["upper"].predict(X_test)
    q_coverage = ((y_test_v.values >= q_lower_test) & (y_test_v.values <= q_upper_test)).mean() * 100
    q_width = np.mean(q_upper_test - q_lower_test)
    q_mae = mean_absolute_error(y_test_v, q_median_test)
    u1, u2, u3 = st.columns(3)
    u1.metric("Quantile Coverage", f"{q_coverage:.1f}%", help="Share of test observations inside the 95% quantile interval")
    u2.metric("Avg Interval Width", f"${q_width:,.0f}")
    u3.metric("Median-Model MAE", f"${q_mae:,.0f}")

    q_preview = pd.DataFrame({
        "Actual": y_test_v.values,
        "Lower": q_lower_test,
        "Median": q_median_test,
        "Upper": q_upper_test,
    }).sort_values("Actual").reset_index(drop=True).head(120)
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=q_preview.index, y=q_preview["Upper"], mode="lines",
                               line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig_q.add_trace(go.Scatter(x=q_preview.index, y=q_preview["Lower"], mode="lines",
                               line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(14, 159, 110, 0.16)", name="95% quantile band"))
    fig_q.add_trace(go.Scatter(x=q_preview.index, y=q_preview["Actual"], mode="markers",
                               marker=dict(color="#1a56db", size=4, opacity=0.6), name="Actual"))
    fig_q.add_trace(go.Scatter(x=q_preview.index, y=q_preview["Median"], mode="lines",
                               line=dict(color="#0e9f6e", width=2), name="Quantile median"))
    fig_q.update_layout(title="Quantile Regression Interval Preview (sorted sample)",
                        xaxis_title="Sorted Test Sample", yaxis_title="Net Profit ($)",
                        plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("---")

    # ── Multi-Target Modeling ──
    st.markdown("### Multi-Target Modeling")
    st.markdown(
        "Separate **Random Forest** and **XGBoost** models trained for "
        "**Net Profit per Order** and **Overall Margin** (outliers excluded from training)."
    )
    _mt_col1, _mt_col2 = st.columns(2)
    for _i, (_tgt_col, _tgt_label) in enumerate([
        ("NetProfitPerOrder", "Net Profit per Order ($)"),
        ("OverallMargin",     "Overall Margin (%)"),
    ]):
        _Xmt_tr, _Xmt_te, _ymt_tr, _ymt_te, _Xmt_tr_s, _Xmt_te_s, _, _ = prepare_ml_target(df, _tgt_col)
        with st.spinner(f"Training secondary models for {_tgt_label}..."):
            _mt_res, _ = train_secondary_models(
                _Xmt_tr_s, _Xmt_te_s, _Xmt_tr, _Xmt_te, _ymt_tr, _ymt_te
            )
        _best_mt = max(_mt_res, key=lambda k: _mt_res[k]["R2"])
        _mt_df = pd.DataFrame({
            "Model":   list(_mt_res.keys()),
            "RMSE":    [f"{v['RMSE']:,.4f}" for v in _mt_res.values()],
            "R\u00b2":       [f"{v['R2']:.4f}"   for v in _mt_res.values()],
            "MAE":     [f"{v['MAE']:,.4f}"  for v in _mt_res.values()],
        })
        with [_mt_col1, _mt_col2][_i]:
            st.markdown(f"**{_tgt_label}**")
            st.dataframe(_mt_df.set_index("Model"), use_container_width=True)
            st.markdown(
                f'Best: <span class="badge-green">{_best_mt} — R\u00b2 = {_mt_res[_best_mt]["R2"]:.4f}</span>',
                unsafe_allow_html=True,
            )
            _fig_mt = go.Figure()
            _fig_mt.add_trace(go.Scatter(
                x=_ymt_te.values, y=_mt_res[_best_mt]["pred"],
                mode="markers", marker=dict(color="#1a56db", opacity=0.5, size=4),
                name="Predictions",
            ))
            _mn_mt, _mx_mt = float(_ymt_te.min()), float(_ymt_te.max())
            _fig_mt.add_trace(go.Scatter(
                x=[_mn_mt, _mx_mt], y=[_mn_mt, _mx_mt], mode="lines",
                name="Perfect Fit", line=dict(color="#ef4444", dash="dash"),
            ))
            _fig_mt.update_layout(
                title=f"Predicted vs Actual — {_tgt_label}",
                xaxis_title="Actual", yaxis_title="Predicted",
                plot_bgcolor="white", paper_bgcolor="white",
                height=280, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(_fig_mt, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — WHAT-IF SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
elif page == "What-If Simulator":
    section_header("What-If Scenario Simulator", "Adjust channel mix and cost parameters to forecast profit outcomes")

    # Ensure models are trained
    if "trained" not in st.session_state or "results" not in st.session_state or not model_results_complete(st.session_state["results"]):
        X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES = prepare_ml(df)
        with st.spinner("Training models..."):
            results, trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state["results"] = results
        st.session_state["trained"] = trained
        st.session_state["sc"]       = sc
        st.session_state["FEATURES"] = FEATURES
        _, X_test2, _, y_test2, _, _, _, _ = prepare_ml(df)
        st.session_state["X_test"]  = X_test2
        st.session_state["y_test"]  = y_test2
    else:
        if "sc" not in st.session_state or "FEATURES" not in st.session_state:
            _, X_test2, _, y_test2, _, _, sc2, FEATURES2 = prepare_ml(df)
            st.session_state["sc"]       = sc2
            st.session_state["FEATURES"] = FEATURES2

    sc       = st.session_state["sc"]
    FEATURES = st.session_state["FEATURES"]
    results  = st.session_state["results"]
    trained  = st.session_state["trained"]
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    Xq_train, _, yq_train, _, _, _, _, _ = prepare_ml(df)
    quantile_models = train_quantile_models(Xq_train, yq_train)

    st.markdown("#### Configure Scenario Parameters")
    st.info("Adjust the sliders below. Channel shares must sum to approximately 1.0")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        with st.expander("Channel Mix", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                in_store = st.slider("In-Store Share",       0.03, 0.55, 0.23, 0.01, format="%.2f")
                ue_share = st.slider("Uber Eats Share",      0.35, 0.60, 0.49, 0.01, format="%.2f")
            with c2:
                dd_share = st.slider("DoorDash Share",       0.20, 0.30, 0.27, 0.01, format="%.2f")
                sd_share = st.slider("Self-Delivery Share",  0.15, 0.45, 0.25, 0.01, format="%.2f")

        total_share = in_store + ue_share + dd_share + sd_share
        if abs(total_share - 1.0) > 0.15:
            st.warning(f"Channel shares sum to {total_share:.2f}. Consider normalising to 1.0.")
        else:
            st.success(f"Channel mix total: {total_share:.2f}")

        with st.expander("Cost & Operations", expanded=True):
            c3, c4 = st.columns(2)
            with c3:
                commission    = st.slider("Commission Rate",         0.27, 0.33, 0.30, 0.005, format="%.3f")
                delivery_cost = st.slider("Delivery Cost/Order ($)", 0.89, 5.31, 3.12, 0.10,  format="%.2f")
            with c4:
                radius        = st.slider("Delivery Radius (km)",    3, 18, 10, 1)
                growth        = st.slider("Growth Factor",           0.99, 1.05, 1.03, 0.005, format="%.3f")

        with st.expander("Restaurant Profile", expanded=True):
            c5, c6 = st.columns(2)
            with c5:
                cuisine  = st.selectbox("Cuisine Type", sorted(df["CuisineType"].unique()))
                segment  = st.selectbox("Segment",      sorted(df["Segment"].unique()))
                subregion = st.selectbox("Subregion",   sorted(df["Subregion"].unique()))
            with c6:
                aov    = st.slider("Avg Order Value ($)",  29.79, 47.23, 38.52, 0.10, format="%.2f")
                orders = st.slider("Monthly Orders",       441, 2337, 1190, 10)
                cogs   = st.slider("COGS Rate",            0.20, 0.40, 0.28, 0.005, format="%.3f")
                opex   = st.slider("OPEX Rate",            0.20, 0.55, 0.41, 0.005, format="%.3f")

    scenario = build_feature_row(
        df,
        in_store, ue_share, dd_share, sd_share,
        commission, delivery_cost, radius, growth,
        aov, orders, cogs, opex,
        cuisine, segment, subregion,
    )

    preds_all = {}
    for name, m in trained.items():
        preds_all[name] = predict_target(name, m, sc, scenario)

    best_pred = preds_all[best_model_name]
    best_lower, _, best_upper = predict_quantile_interval(quantile_models, scenario)
    risk_level, interval_width_pct = scenario_risk_label(best_lower, best_upper, best_pred)

    with col_r:
        st.markdown("#### Scenario Forecast")
        for name, p in preds_all.items():
            badge = "badge-green" if p > 0 else "badge-red"
            is_best = " (Best Model)" if name == best_model_name else ""
            if name == best_model_name:
                lower_bound, upper_bound = best_lower, best_upper
                interval_label = "95% quantile interval"
            else:
                lower_bound, upper_bound = prediction_interval(p, results[name])
                interval_label = "95% empirical interval"
            st.markdown(f"""
            <div class="info-card">
              <h4>{name}{is_best}</h4>
              <p>Predicted Net Profit:<br>
                            <span class="{badge}" style="font-size:1.1rem">${p:,.0f}</span><br>
                            {interval_label}: ${lower_bound:,.0f} to ${upper_bound:,.0f}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Uncertainty & Risk")
        r1, r2, r3 = st.columns(3)
        r1.metric("Best-Model P2.5", f"${best_lower:,.0f}")
        r2.metric("Best-Model P97.5", f"${best_upper:,.0f}")
        r3.metric("Scenario Risk Level", risk_level, help=f"Prediction interval width is {interval_width_pct:.1f}% of the forecast magnitude.")

    st.markdown("---")

    # ── Channel-level breakdown ──
    st.markdown("#### Estimated Channel-Level Profit Breakdown")
    rev_instore = aov * orders * in_store
    rev_ue      = aov * orders * ue_share
    rev_dd      = aov * orders * dd_share
    rev_sd      = aov * orders * sd_share

    profit_is = rev_instore  * (1 - cogs - opex)
    profit_ue = rev_ue       * (1 - cogs - opex - commission)
    profit_dd = rev_dd       * (1 - cogs - opex - commission)
    sd_total  = delivery_cost * (orders * sd_share)
    profit_sd = rev_sd       * (1 - cogs - opex) - sd_total

    breakdown = pd.DataFrame({
        "Channel":    ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
        "Revenue ($)": [rev_instore, rev_ue, rev_dd, rev_sd],
        "Net Profit ($)": [profit_is, profit_ue, profit_dd, profit_sd],
        "Margin":      [profit_is/rev_instore if rev_instore else 0,
                        profit_ue/rev_ue if rev_ue else 0,
                        profit_dd/rev_dd if rev_dd else 0,
                        profit_sd/rev_sd if rev_sd else 0],
    })

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(breakdown, x="Channel", y="Net Profit ($)", color="Channel",
                     title="Projected Net Profit by Channel",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(breakdown, x="Channel", y="Margin", color="Channel",
                      title="Projected Margin by Channel",
                      color_discrete_sequence=PALETTE)
        fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                           yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    # ── Sensitivity sweep ──
    st.markdown("#### Sensitivity Sweep: Commission Rate Impact on Net Profit")
    comm_range = np.linspace(0.27, 0.33, 30)
    sweep_profits = []
    sweep_lower = []
    sweep_upper = []
    for cr in comm_range:
        s = build_feature_row(
            df,
            in_store, ue_share, dd_share, sd_share,
            cr, delivery_cost, radius, growth,
            aov, orders, cogs, opex,
            cuisine, segment, subregion,
        )
        sweep_pred = predict_target(best_model_name, trained[best_model_name], sc, s)
        sweep_lo, _, sweep_hi = predict_quantile_interval(quantile_models, s)
        sweep_profits.append(sweep_pred)
        sweep_lower.append(sweep_lo)
        sweep_upper.append(sweep_hi)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=comm_range * 100, y=sweep_upper, mode="lines",
                               line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig3.add_trace(go.Scatter(x=comm_range * 100, y=sweep_lower, mode="lines",
                               line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(26, 86, 219, 0.14)",
                               name="95% prediction band"))
    fig3.add_trace(go.Scatter(x=comm_range * 100, y=sweep_profits, mode="lines+markers",
                               line=dict(color="#1a56db", width=2),
                               marker=dict(size=5), name="Predicted net profit"))
    fig3.add_vline(x=commission * 100, line_color="#ef4444", line_dash="dash",
                   annotation_text=f"Current: {commission*100:.1f}%")
    fig3.update_layout(title=f"Commission Rate vs Predicted Net Profit ({best_model_name})",
                       xaxis_title="Commission Rate (%)", yaxis_title="Predicted Net Profit ($)",
                       plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Export Scenario Results")
    scenario_inputs = pd.DataFrame({
        "Metric": [
            "In-Store Share", "Uber Eats Share", "DoorDash Share", "Self-Delivery Share",
            "Commission Rate", "Delivery Cost/Order", "Delivery Radius (km)", "Growth Factor",
            "Cuisine Type", "Segment", "Subregion", "Avg Order Value", "Monthly Orders",
            "COGS Rate", "OPEX Rate", "Best Model", "Best-Model Risk"
        ],
        "Value": [
            in_store, ue_share, dd_share, sd_share,
            commission, delivery_cost, radius, growth,
            cuisine, segment, subregion, aov, orders,
            cogs, opex, best_model_name, risk_level
        ]
    })
    scenario_forecast = pd.DataFrame([
        {
            "Model": name,
            "Predicted Net Profit ($)": round(prediction, 2),
            "P2.5 ($)": round(prediction_interval(prediction, results[name])[0], 2),
            "P97.5 ($)": round(prediction_interval(prediction, results[name])[1], 2),
            "CV RMSE Mean ($)": round(results[name]["CV_RMSE_mean"], 2),
            "CV R² Mean": round(results[name]["CV_R2_mean"], 4),
            "CV MAE Mean ($)": round(results[name]["CV_MAE_mean"], 2),
        }
        for name, prediction in preds_all.items()
    ])
    export_summary = pd.concat([
        scenario_inputs.assign(Section="Scenario Inputs"),
        scenario_forecast.rename(columns={"Model": "Metric", "Predicted Net Profit ($)": "Value"}).assign(Section="Model Forecasts")[["Section", "Metric", "Value"]],
    ], ignore_index=True)
    export_breakdown = breakdown.copy()
    export_breakdown["Margin"] = export_breakdown["Margin"].round(4)
    sweep_df = pd.DataFrame({
        "Commission Rate (%)": np.round(comm_range * 100, 2),
        "Predicted Net Profit ($)": np.round(sweep_profits, 2),
        "P2.5 ($)": np.round(sweep_lower, 2),
        "P97.5 ($)": np.round(sweep_upper, 2),
    })

    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "Download Scenario Summary CSV",
        data=export_summary.to_csv(index=False).encode("utf-8"),
        file_name="scenario_summary.csv",
        mime="text/csv",
    )
    e2.download_button(
        "Download Channel Breakdown CSV",
        data=export_breakdown.to_csv(index=False).encode("utf-8"),
        file_name="scenario_breakdown.csv",
        mime="text/csv",
    )
    e3.download_button(
        "Download Sensitivity Sweep CSV",
        data=sweep_df.to_csv(index=False).encode("utf-8"),
        file_name="scenario_sensitivity_sweep.csv",
        mime="text/csv",
    )

    if HAS_FPDF:
        _pdf_bytes = generate_scenario_pdf(
            preds_all, breakdown, results, best_model_name, risk_level,
            best_lower, best_upper,
            in_store, ue_share, dd_share, sd_share,
            commission, delivery_cost, radius, growth,
            aov, orders, cogs, opex,
            cuisine, segment, subregion,
        )
        if _pdf_bytes:
            st.download_button(
                "Download Scenario Report (PDF)",
                data=_pdf_bytes,
                file_name="scenario_report.pdf",
                mime="application/pdf",
            )
    else:
        st.caption("Install `fpdf2` to enable PDF export.")

    st.markdown("---")
    st.markdown("#### Scenario Comparison")
    _sc_name = st.text_input(
        "Scenario name",
        value=f"Scenario {len(st.session_state.get('saved_scenarios', [])) + 1}",
        key="sc_name_input",
    )
    _sv_col, _cl_col, _ = st.columns([1, 1, 4])
    with _sv_col:
        if st.button("Save Scenario"):
            _saved = st.session_state.get("saved_scenarios", [])
            _entry = {
                "Name":       _sc_name,
                "In-Store":   f"{in_store:.0%}",
                "UE":         f"{ue_share:.0%}",
                "DD":         f"{dd_share:.0%}",
                "SD":         f"{sd_share:.0%}",
                "Commission": f"{commission:.1%}",
                "COGS":       f"{cogs:.1%}",
                "OPEX":       f"{opex:.1%}",
                "Del Cost":   f"${delivery_cost:.2f}",
                **{f"{_n} ($)": f"${_p:,.0f}" for _n, _p in preds_all.items()},
                "Risk":       risk_level,
            }
            _saved.append(_entry)
            st.session_state["saved_scenarios"] = _saved[-5:]
            st.success(f"Saved '{_sc_name}'")
    with _cl_col:
        if st.button("Clear All"):
            st.session_state["saved_scenarios"] = []
            st.rerun()

    _saved_list = st.session_state.get("saved_scenarios", [])
    if len(_saved_list) >= 2:
        _comp_df = pd.DataFrame(_saved_list).set_index("Name")
        st.dataframe(_comp_df, use_container_width=True)
        _best_col = f"{best_model_name} ($)"
        if _best_col in _comp_df.columns:
            _profit_vals = _comp_df[_best_col].str.replace(r"[$,]", "", regex=True).astype(float)
            _fig_comp = px.bar(
                x=_profit_vals.index.tolist(), y=_profit_vals.values,
                title=f"Scenario Comparison — {best_model_name} Predicted Net Profit",
                color=_profit_vals.values,
                color_continuous_scale=["#fee2e2", "#dbeafe", "#d1fae5"],
                labels={"x": "Scenario", "y": "Predicted Net Profit ($)"},
            )
            _fig_comp.update_layout(
                showlegend=False, coloraxis_showscale=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(_fig_comp, use_container_width=True)
    elif len(_saved_list) == 1:
        st.info(f"1 scenario saved: **{_saved_list[0]['Name']}**. Save another to compare.")
    else:
        st.info("Save 2 or more scenarios to compare them side by side.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — OPTIMIZATION PANEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "Optimization Panel":
    section_header("Prescriptive Optimization", "Identify optimal channel mix, safe operating ranges, and strategic recommendations")

    # Ensure models trained
    if "trained" not in st.session_state or "results" not in st.session_state or not model_results_complete(st.session_state["results"]):
        X_train, X_test, y_train, y_test, X_tr_s, X_te_s, sc, FEATURES = prepare_ml(df)
        with st.spinner("Training models for optimization..."):
            results, trained = train_models(X_tr_s, X_te_s, X_train, X_test, y_train, y_test)
        st.session_state.update({"results": results, "trained": trained, "sc": sc, "FEATURES": FEATURES,
                                  "X_test": X_test, "y_test": y_test})
    results = st.session_state["results"]
    trained = st.session_state["trained"]
    sc = st.session_state["sc"]
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    X_opt_train, _, y_opt_train, _, _, _, _, _ = prepare_ml(df)
    quantile_models = train_quantile_models(X_opt_train, y_opt_train)

    # ── KPI summary ──
    df_opt = df.copy()
    df_opt["ProfitSensitivityIndex"] = df_opt["TotalNetProfit"].std() / df_opt["TotalRevenue"].mean() * 100
    channel_efficiency = {
        "In-Store":      df_opt["InStoreNetProfit"].mean()  / df_opt["InStoreShare"].mean(),
        "Uber Eats":     df_opt["UberEatsNetProfit"].mean() / df_opt["UE_share"].mean(),
        "DoorDash":      df_opt["DoorDashNetProfit"].mean() / df_opt["DD_share"].mean(),
        "Self-Delivery": df_opt["SelfDeliveryNetProfit"].mean() / df_opt["SD_share"].mean(),
    }
    best_channel   = max(channel_efficiency, key=channel_efficiency.get)
    breakeven_comm = df_opt[df_opt["UberEatsNetProfit"] > 0]["CommissionRate"].max()
    opt_uplift_pct = (df_opt["TotalNetProfit"].quantile(0.75) - df_opt["TotalNetProfit"].mean()) / abs(df_opt["TotalNetProfit"].mean()) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit Sensitivity Index",    f"{df_opt['ProfitSensitivityIndex'].mean():.2f}%", help="Volatility relative to revenue")
    c2.metric("Most Efficient Channel",      best_channel)
    c3.metric("Break-Even Commission Rate",  f"{breakeven_comm*100:.1f}%", help="Max commission where UberEats stays profitable")
    c4.metric("Optimization Uplift Potential", f"{opt_uplift_pct:.1f}%",  help="P75 vs mean profit gap")

    st.markdown("---")

    st.markdown("#### Formal Channel-Mix Optimizer")
    if HAS_SCIPY:
        st.caption("Uses constrained SLSQP optimization to maximize predicted monthly net profit with channel shares summing to 100% and staying inside observed operating ranges.")
        opt_left, opt_right = st.columns(2)
        with opt_left:
            opt_cuisine = st.selectbox("Cuisine Type", sorted(df["CuisineType"].unique()), key="opt_cuisine")
            opt_segment = st.selectbox("Segment", sorted(df["Segment"].unique()), key="opt_segment")
            opt_subregion = st.selectbox("Subregion", sorted(df["Subregion"].unique()), key="opt_subregion")
            opt_commission = st.slider("Commission Rate", 0.27, 0.33, 0.30, 0.005, format="%.3f", key="opt_commission")
            opt_delivery_cost = st.slider("Delivery Cost/Order ($)", 0.89, 5.31, 3.12, 0.10, format="%.2f", key="opt_delivery_cost")
        with opt_right:
            opt_radius = st.slider("Delivery Radius (km)", 3, 18, 10, 1, key="opt_radius")
            opt_growth = st.slider("Growth Factor", 0.99, 1.05, 1.03, 0.005, format="%.3f", key="opt_growth")
            opt_aov = st.slider("Avg Order Value ($)", 29.79, 47.23, 38.52, 0.10, format="%.2f", key="opt_aov")
            opt_orders = st.slider("Monthly Orders", 441, 2337, 1190, 10, key="opt_orders")
            opt_cogs = st.slider("COGS Rate", 0.20, 0.40, 0.28, 0.005, format="%.3f", key="opt_cogs")
            opt_opex = st.slider("OPEX Rate", 0.20, 0.55, 0.41, 0.005, format="%.3f", key="opt_opex")

        if st.button("Run Formal Optimizer"):
            with st.spinner("Solving constrained channel mix optimization..."):
                opt_result, share_bounds = optimize_channel_mix(
                    df, trained, sc, best_model_name,
                    opt_commission, opt_delivery_cost, opt_radius, opt_growth,
                    opt_aov, opt_orders, opt_cogs, opt_opex,
                    opt_cuisine, opt_segment, opt_subregion,
                )
            if opt_result.success:
                baseline_shares = np.array([
                    float(df["InStoreShare"].mean()),
                    float(df["UE_share"].mean()),
                    float(df["DD_share"].mean()),
                    float(df["SD_share"].mean()),
                ])
                baseline_shares = baseline_shares / baseline_shares.sum()
                baseline_row = build_feature_row(
                    df,
                    baseline_shares[0], baseline_shares[1], baseline_shares[2], baseline_shares[3],
                    opt_commission, opt_delivery_cost, opt_radius, opt_growth,
                    opt_aov, opt_orders, opt_cogs, opt_opex,
                    opt_cuisine, opt_segment, opt_subregion,
                )
                optimized_row = build_feature_row(
                    df,
                    opt_result.x[0], opt_result.x[1], opt_result.x[2], opt_result.x[3],
                    opt_commission, opt_delivery_cost, opt_radius, opt_growth,
                    opt_aov, opt_orders, opt_cogs, opt_opex,
                    opt_cuisine, opt_segment, opt_subregion,
                )
                baseline_profit = predict_target(best_model_name, trained[best_model_name], sc, baseline_row)
                optimized_profit = predict_target(best_model_name, trained[best_model_name], sc, optimized_row)
                opt_lower, _, opt_upper = predict_quantile_interval(quantile_models, optimized_row)

                f1, f2, f3 = st.columns(3)
                f1.metric("Baseline Forecast", f"${baseline_profit:,.0f}")
                f2.metric("Optimized Forecast", f"${optimized_profit:,.0f}")
                f3.metric("Model Uplift", f"${optimized_profit - baseline_profit:,.0f}")
                st.caption(f"Formal 95% quantile interval at optimum: ${opt_lower:,.0f} to ${opt_upper:,.0f}")

                opt_df = pd.DataFrame({
                    "Channel": ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"],
                    "Baseline": baseline_shares,
                    "Optimized": opt_result.x,
                    "Observed Safe Range": [
                        f"{low:.0%} to {high:.0%}" for low, high in share_bounds
                    ],
                })
                o1, o2 = st.columns(2)
                with o1:
                    st.dataframe(opt_df.set_index("Channel"), use_container_width=True)
                with o2:
                    fig_opt = px.bar(
                        opt_df,
                        x="Channel",
                        y=["Baseline", "Optimized"],
                        barmode="group",
                        title=f"Channel Mix Recommendation ({best_model_name})",
                        color_discrete_sequence=["#dbeafe", "#0e9f6e"],
                    )
                    fig_opt.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_opt, use_container_width=True)
            else:
                st.warning("Optimizer did not converge for the current constraints. Try adjusting the fixed operating inputs.")
    else:
        st.info("Install SciPy to enable the formal optimization solver.")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Channel Efficiency", "Commission Analysis", "Self-Delivery Threshold", "Recommendations"])

    # ── Tab 1: Channel Efficiency ──
    with tab1:
        st.markdown("#### Channel Mix Efficiency Score (Profit per Unit Share)")
        eff_df = pd.DataFrame(list(channel_efficiency.items()), columns=["Channel", "Profit per Share Point"])
        fig = px.bar(eff_df, x="Channel", y="Profit per Share Point",
                     color="Channel", color_discrete_sequence=PALETTE,
                     title="Channel Mix Efficiency Score")
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Profitability vs Channel Share — Optimal Mix Region")
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.scatter(df, x="InStoreShare", y="TotalNetProfit",
                              color="Segment", trendline=OPTIONAL_OLS,
                              title="In-Store Share vs Total Net Profit",
                              color_discrete_sequence=PALETTE)
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.scatter(df, x="SD_share", y="SelfDeliveryNetProfit",
                              color="Subregion", trendline=OPTIONAL_OLS,
                              title="Self-Delivery Share vs Self-Delivery Profit",
                              color_discrete_sequence=PALETTE)
            fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickformat=".0%", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Top 20 Most Profitable Restaurant Configurations")
        top20 = df.nlargest(20, "TotalNetProfit")[
            ["RestaurantName", "CuisineType", "Segment", "Subregion",
             "InStoreShare", "UE_share", "SD_share", "CommissionRate",
             "TotalNetProfit", "OverallMargin"]
        ].copy()
        top20["InStoreShare"] = top20["InStoreShare"].map("{:.0%}".format)
        top20["UE_share"]     = top20["UE_share"].map("{:.0%}".format)
        top20["SD_share"]     = top20["SD_share"].map("{:.0%}".format)
        top20["OverallMargin"]= top20["OverallMargin"].map("{:.1%}".format)
        top20["TotalNetProfit"]= top20["TotalNetProfit"].map("${:,.0f}".format)
        top20["CommissionRate"]= top20["CommissionRate"].map("{:.1%}".format)
        st.dataframe(top20.set_index("RestaurantName"), use_container_width=True)

    # ── Tab 2: Commission Analysis ──
    with tab2:
        st.markdown("#### Commission Rate vs Profitability — Safe Operating Zones")
        df["CommBucket"] = pd.cut(df["CommissionRate"],
                                   bins=[0.26, 0.28, 0.30, 0.32, 0.34],
                                   labels=["27-28%", "28-30%", "30-32%", "32-33%"])
        bucket_stats = df.groupby("CommBucket", observed=True)["TotalNetProfit"].agg(["mean","median","std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mean Profit",   x=bucket_stats["CommBucket"].astype(str),
                             y=bucket_stats["mean"],   marker_color="#1a56db"))
        fig.add_trace(go.Bar(name="Median Profit", x=bucket_stats["CommBucket"].astype(str),
                             y=bucket_stats["median"], marker_color="#0e9f6e"))
        fig.update_layout(title="Net Profit by Commission Rate Bucket", barmode="group",
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Commission Rate", yaxis_title="Net Profit ($)",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.box(df, x="CommBucket", y="UberEatsNetProfit",
                          title="Uber Eats Net Profit by Commission Bucket",
                          color="CommBucket", color_discrete_sequence=PALETTE)
            fig2.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.box(df, x="CommBucket", y="DoorDashNetProfit",
                          title="DoorDash Net Profit by Commission Bucket",
                          color="CommBucket", color_discrete_sequence=PALETTE)
            fig3.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # Safe range
        safe = df.groupby("CommBucket", observed=True)["UberEatsNetProfit"].apply(lambda x: (x > 0).mean() * 100).reset_index()
        safe.columns = ["CommBucket", "% Profitable"]
        fig4 = px.bar(safe, x="CommBucket", y="% Profitable",
                      title="% of UberEats Orders Profitable by Commission Bucket",
                      color="% Profitable", color_continuous_scale=["#fee2e2", "#d1fae5"],
                      text_auto=".1f")
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3: Self-Delivery Threshold ──
    with tab3:
        st.markdown("#### Self-Delivery Investment Break-Even Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x="DeliveryCostPerOrder", y="SelfDeliveryNetProfit",
                             color="Subregion", trendline=OPTIONAL_OLS,
                             title="Delivery Cost vs Self-Delivery Net Profit",
                             color_discrete_sequence=PALETTE)
            fig.add_hline(y=0, line_color="red", line_dash="dash",
                          annotation_text="Break-Even Line")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df, x="DeliveryRadiusKM", y="SelfDeliveryNetProfit",
                              color="Segment", trendline=OPTIONAL_OLS,
                              title="Delivery Radius vs Self-Delivery Net Profit",
                              color_discrete_sequence=PALETTE)
            fig2.add_hline(y=0, line_color="red", line_dash="dash",
                           annotation_text="Break-Even Line")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Break-even cost threshold
        be_cost = df[df["SelfDeliveryNetProfit"] > 0]["DeliveryCostPerOrder"].max()
        be_radius = df[df["SelfDeliveryNetProfit"] > 0]["DeliveryRadiusKM"].max()
        profitable_sd_pct = (df["SelfDeliveryNetProfit"] > 0).mean() * 100

        st.markdown("#### Self-Delivery Operating Thresholds")
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Profitable Delivery Cost",    f"${be_cost:.2f}/order")
        c2.metric("Max Profitable Delivery Radius",  f"{be_radius} km")
        c3.metric("% SD Operations Profitable",      f"{profitable_sd_pct:.1f}%")

        # Cost vs radius heatmap
        df["CostBin"]   = pd.cut(df["DeliveryCostPerOrder"], bins=5)
        df["RadiusBin"] = pd.cut(df["DeliveryRadiusKM"],    bins=5)
        hm = df.groupby(["CostBin","RadiusBin"], observed=True)["SelfDeliveryNetProfit"].mean().reset_index()
        hm_piv = hm.pivot(index="CostBin", columns="RadiusBin", values="SelfDeliveryNetProfit")
        hm_piv.index   = [str(x) for x in hm_piv.index]
        hm_piv.columns = [str(x) for x in hm_piv.columns]
        fig3 = px.imshow(hm_piv, title="Self-Delivery Profit: Cost vs Radius Heatmap",
                         color_continuous_scale="RdYlGn", text_auto=".0f",
                         labels=dict(color="Avg Net Profit ($)"))
        fig3.update_layout(paper_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: Recommendations ──
    with tab4:
        st.markdown("#### Strategic Recommendations for SkyCity Auckland")
        st.markdown("")

        # Compute key insights
        top_seg  = df.groupby("Segment")["TotalNetProfit"].mean().idxmax()
        top_sub  = df.groupby("Subregion")["TotalNetProfit"].mean().idxmax()
        top_cuis = df.groupby("CuisineType")["TotalNetProfit"].mean().idxmax()
        opt_is   = df.groupby(pd.cut(df["InStoreShare"], bins=5), observed=True)["TotalNetProfit"].mean().idxmax()
        opt_sd   = df.groupby(pd.cut(df["SD_share"], bins=5), observed=True)["TotalNetProfit"].mean().idxmax()
        best_r2_name = max(results, key=lambda k: results[k]["R2"])

        recs = [
            ("Channel Optimization",       f"Prioritise <b>In-Store</b> and <b>Self-Delivery</b> channels over pure aggregator reliance. In-Store consistently delivers the highest average net profit (${ df['InStoreNetProfit'].mean():,.0f}/month)."),
            ("Commission Negotiation",     f"Target commission rates at or below <b>30%</b>. The break-even commission rate is <b>{breakeven_comm*100:.1f}%</b> — negotiations above this threshold erode Uber Eats profitability."),
            ("Self-Delivery Expansion",    f"Self-delivery is profitable up to <b>${be_cost:.2f}/order</b> delivery cost and <b>{be_radius} km</b> radius. <b>{profitable_sd_pct:.1f}%</b> of current SD operations are profitable — consider expanding."),
            ("Best Performing Segment",    f"<b>{top_seg}</b> segment leads in average net profit. Allocate more marketing and operational resources to this format."),
            ("Subregion Focus",            f"<b>{top_sub}</b> is the highest-performing subregion. Prioritise new openings and expansions in this area."),
            ("Cuisine Mix Strategy",       f"<b>{top_cuis}</b> cuisine type generates the highest average net profit. Consider expanding this offering across underperforming subregions."),
            ("Predictive Model Adoption",  f"Use the <b>{best_r2_name}</b> model (R² = {results[best_r2_name]['R2']:.3f}) for profit forecasting. It explains {results[best_r2_name]['R2']*100:.1f}% of profit variance."),
            ("Cost Structure Management",  f"Restaurants with combined COGS+OPEX below <b>70%</b> of revenue consistently outperform. Target COGS ≤28% and OPEX ≤40%."),
        ]

        for i, (title, body) in enumerate(recs):
            badge_cls = ["badge-blue", "badge-yellow", "badge-green", "badge-blue",
                         "badge-green", "badge-blue", "badge-green", "badge-yellow"][i]
            num = i + 1
            st.markdown(f"""
            <div class="info-card">
              <h4><span class="{badge_cls}">#{num}</span> &nbsp; {title}</h4>
              <p>{body}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Profit Optimization Score by Restaurant")
        df_score = df.copy()
        df_score["OptScore"] = (
            (df_score["TotalNetProfit"] / df_score["TotalNetProfit"].max()) * 40 +
            (1 - df_score["CommissionRate"] / df_score["CommissionRate"].max()) * 30 +
            (df_score["InStoreShare"]) * 20 +
            (df_score["GrowthFactor"] / df_score["GrowthFactor"].max()) * 10
        ).round(2)

        top_opt = df_score.nlargest(15, "OptScore")[
            ["RestaurantName", "CuisineType", "Segment", "Subregion", "OptScore",
             "TotalNetProfit", "CommissionRate"]
        ]
        top_opt["TotalNetProfit"] = top_opt["TotalNetProfit"].map("${:,.0f}".format)
        top_opt["CommissionRate"] = top_opt["CommissionRate"].map("{:.1%}".format)
        top_opt["OptScore"]       = top_opt["OptScore"].map("{:.2f}".format)
        st.dataframe(top_opt.set_index("RestaurantName"), use_container_width=True)
