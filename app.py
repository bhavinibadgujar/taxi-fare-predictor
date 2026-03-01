"""
🚖 Taxi Fare Prediction System
Deployed with Streamlit | Models: Random Forest & XGBoost/GradientBoosting
Simulates Uber/Ola Dynamic Surge Pricing
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taxi Fare Predictor",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0f0f1a; }

.hero-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid #e94560;
    box-shadow: 0 8px 32px rgba(233,69,96,0.2);
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-4px); }

.fare-display {
    background: linear-gradient(135deg, #e94560, #c62a47);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 12px 40px rgba(233,69,96,0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 12px 40px rgba(233,69,96,0.4); }
    50% { box-shadow: 0 12px 60px rgba(233,69,96,0.7); }
}

.surge-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.9rem;
}
.surge-low { background: #1a3a1a; color: #4ade80; border: 1px solid #4ade80; }
.surge-med { background: #3a2a0a; color: #fbbf24; border: 1px solid #fbbf24; }
.surge-high { background: #3a0a0a; color: #f87171; border: 1px solid #f87171; }

.stSlider > div > div > div > div { background: #e94560 !important; }
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c62a47) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: transform 0.2s !important;
}
.stButton > button:hover { transform: scale(1.02) !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model (auto-trains inline if not found — works on Streamlit Cloud) ────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'fare_model.pkl')

    if not os.path.exists(model_path):
        # ── Inline training (no subprocess needed) ────────────────────────────
        import numpy as np_t
        import pandas as pd_t
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        np_t.random.seed(42)
        N = 10000
        def _hav(la1, lo1, la2, lo2):
            R = 6371
            dlat = np_t.radians(la2 - la1); dlon = np_t.radians(lo2 - lo1)
            a = np_t.sin(dlat/2)**2 + np_t.cos(np_t.radians(la1))*np_t.cos(np_t.radians(la2))*np_t.sin(dlon/2)**2
            return R * 2 * np_t.arcsin(np_t.sqrt(a))

        la1 = np_t.random.uniform(18.90, 19.20, N); lo1 = np_t.random.uniform(72.77, 73.05, N)
        la2 = np_t.random.uniform(18.90, 19.20, N); lo2 = np_t.random.uniform(72.77, 73.05, N)
        distance = np_t.clip(_hav(la1, lo1, la2, lo2), 0.5, 40)
        hour = np_t.random.randint(0, 24, N)
        day  = np_t.random.randint(0, 7,  N)
        pax  = np_t.random.randint(1, 7,  N)
        wthr = np_t.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1])
        surge = np_t.ones(N)
        surge += np_t.where((hour >= 8)  & (hour <= 10), 0.5, 0)
        surge += np_t.where((hour >= 17) & (hour <= 20), 0.6, 0)
        surge += np_t.where((day  >= 5)  & (hour >= 20), 0.4, 0)
        surge += np_t.where(wthr == 1, 0.3, 0)
        surge += np_t.where(wthr == 2, 0.7, 0)
        surge  = np_t.clip(surge, 1.0, 3.0)
        is_peak = (((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))).astype(int)
        fare = np_t.clip(30 + 14 * distance * surge + pax * 2 + np_t.random.normal(0, 8, N), 40, 2000)

        df = pd_t.DataFrame({'distance_km': distance, 'hour': hour, 'day_of_week': day,
                             'passengers': pax, 'weather': wthr, 'is_peak': is_peak,
                             'surge_multiplier': surge, 'fare': fare.round(2)})
        X = df.drop('fare', axis=1); y = df['fare']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # Try XGBoost first, fall back to Random Forest
        try:
            from xgboost import XGBRegressor
            mdl = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                               random_state=42, verbosity=0)
            model_name = "XGBoost"
        except Exception:
            mdl = RandomForestRegressor(n_estimators=200, max_depth=12,
                                        random_state=42, n_jobs=-1)
            model_name = "RandomForest"

        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        r2  = r2_score(y_te, pred)
        mae = mean_absolute_error(y_te, pred)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({'model': mdl, 'features': list(X.columns),
                         'model_name': model_name, 'r2': r2, 'mae': mae}, f)

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model_data = load_model()


# ── Helper Functions ──────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def compute_surge(hour, day, weather):
    surge = 1.0
    if 8 <= hour <= 10:  surge += 0.5
    if 17 <= hour <= 20: surge += 0.6
    if day >= 5 and 20 <= hour <= 23: surge += 0.4
    if weather == 1: surge += 0.3
    if weather == 2: surge += 0.7
    return round(min(surge, 3.0), 2)

def get_surge_label(surge):
    if surge <= 1.3:   return "Normal", "surge-low"
    elif surge <= 1.8: return "Moderate Surge", "surge-med"
    else:              return "High Surge 🔥", "surge-high"

MUMBAI_LOCATIONS = {
    "Chhatrapati Shivaji Terminal":  (18.9400, 72.8350),
    "Bandra Station":                (19.0544, 72.8396),
    "Andheri Station":               (19.1197, 72.8468),
    "Dadar Station":                 (19.0186, 72.8433),
    "Kurla Station":                 (19.0658, 72.8792),
    "Thane Station":                 (19.1863, 72.9706),
    "Borivali Station":              (19.2307, 72.8568),
    "Nariman Point":                 (18.9255, 72.8242),
    "Powai Lake":                    (19.1176, 72.9060),
    "BKC (Bandra Kurla Complex)":    (19.0596, 72.8654),
    "Juhu Beach":                    (19.0948, 72.8258),
    "Worli Sea Face":                (19.0130, 72.8153),
    "Custom Location":               None,
}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <h1 style="color:#e94560;margin:0;font-size:2.5rem;">🚖 Taxi Fare Predictor</h1>
  <p style="color:#8892b0;margin:0.5rem 0 0;">AI-Powered Dynamic Pricing | Simulating Uber & Ola Surge Models</p>
</div>
""", unsafe_allow_html=True)


# ── Model Info Bar ────────────────────────────────────────────────────────────
if model_data:
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("🤖 Model", model_data.get('model_name', 'ML Model')),
        ("📊 Accuracy (R²)", f"{model_data.get('r2', 0)*100:.2f}%"),
        ("💰 Avg Error (MAE)", f"₹{model_data.get('mae', 0):.2f}"),
        ("📦 Training Data", "10,000 rides"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <p style="color:#8892b0;margin:0;font-size:0.8rem;">{label}</p>
            <h3 style="color:#e94560;margin:0.3rem 0 0;">{val}</h3>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ Trip Details")
    st.markdown("---")

    # Pickup
    st.markdown("**📍 Pickup Location**")
    pickup_name = st.selectbox("Pickup", list(MUMBAI_LOCATIONS.keys()), key="pickup")
    if MUMBAI_LOCATIONS[pickup_name] is None:
        plat = st.number_input("Pickup Latitude",  value=19.0760, format="%.4f")
        plon = st.number_input("Pickup Longitude", value=72.8777, format="%.4f")
    else:
        plat, plon = MUMBAI_LOCATIONS[pickup_name]

    st.markdown("**🏁 Drop Location**")
    drop_name = st.selectbox("Drop", list(MUMBAI_LOCATIONS.keys()), index=3, key="drop")
    if MUMBAI_LOCATIONS[drop_name] is None:
        dlat = st.number_input("Drop Latitude",  value=18.9400, format="%.4f")
        dlon = st.number_input("Drop Longitude", value=72.8350, format="%.4f")
    else:
        dlat, dlon = MUMBAI_LOCATIONS[drop_name]

    st.markdown("---")
    st.markdown("## ⏰ Ride Conditions")
    hour = st.slider("🕐 Hour of Day", 0, 23, 18,
                     help="Peak hours: 8–10 AM, 5–8 PM")
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day = st.selectbox("📅 Day of Week", range(7), format_func=lambda x: day_names[x], index=4)
    passengers = st.slider("👥 Passengers", 1, 6, 1)
    weather_map = {0: "☀️ Clear", 1: "🌧️ Rain", 2: "⛈️ Storm"}
    weather = st.selectbox("🌤️ Weather", [0, 1, 2], format_func=lambda x: weather_map[x])

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Fare", use_container_width=True)


# ── Main Content ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    # Compute values
    distance = haversine(plat, plon, dlat, dlon)
    surge = compute_surge(hour, day, weather)
    is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
    surge_label, surge_class = get_surge_label(surge)

    # Trip summary
    st.markdown("### 📋 Trip Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("📏 Distance", f"{distance:.2f} km")
    s2.metric("⚡ Surge", f"{surge}x")
    s3.metric("🕐 Peak Hour", "Yes ✅" if is_peak else "No ❌")

    st.markdown(f"""
    <div style="margin:1rem 0;">
        Surge Status: <span class="surge-badge {surge_class}">{surge_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Surge breakdown
    st.markdown("### 🔍 Surge Pricing Breakdown")
    breakdown_df = pd.DataFrame({
        "Factor": ["Base", "Morning Peak (8–10 AM)", "Evening Peak (5–8 PM)",
                   "Weekend Night", "Rain", "Storm"],
        "Applied": [
            "✅ Always",
            "✅" if 8 <= hour <= 10 else "❌",
            "✅" if 17 <= hour <= 20 else "❌",
            "✅" if day >= 5 and 20 <= hour <= 23 else "❌",
            "✅" if weather == 1 else "❌",
            "✅" if weather == 2 else "❌",
        ],
        "Multiplier Add": ["+1.0x", "+0.5x", "+0.6x", "+0.4x", "+0.3x", "+0.7x"]
    })
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # How fare is calculated
    st.markdown("### 💡 Fare Formula")
    st.code(f"""
Base Fare         = ₹30
Distance Charge   = {distance:.2f} km × ₹14 = ₹{distance*14:.2f}
Surge Multiplier  = {surge}x
Passenger Add     = {passengers} × ₹2 = ₹{passengers*2}
─────────────────────────────────
Estimated Total   ≈ ₹{(30 + distance*14*surge + passengers*2):.2f}
    """, language="text")


with col_right:
    st.markdown("### 💰 Prediction Result")

    if predict_btn and model_data:
        features = np.array([[distance, hour, day, passengers,
                               weather, is_peak, surge]])
        feat_df = pd.DataFrame(features, columns=model_data['features'])
        predicted_fare = model_data['model'].predict(feat_df)[0]

        st.markdown(f"""
        <div class="fare-display">
            <p style="color:rgba(255,255,255,0.7);margin:0;font-size:1rem;">Predicted Fare</p>
            <h1 style="color:white;margin:0.5rem 0;font-size:3.5rem;font-weight:700;">
                ₹{predicted_fare:.0f}
            </h1>
            <p style="color:rgba(255,255,255,0.6);margin:0;font-size:0.85rem;">
                {distance:.1f} km | {surge}x surge | {day_names[day]}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Price range
        low = predicted_fare * 0.92
        high = predicted_fare * 1.08
        st.info(f"📊 Estimated Range: ₹{low:.0f} – ₹{high:.0f}")

        # Comparison
        st.markdown("#### 🆚 Platform Comparison")
        ola_fare    = predicted_fare * np.random.uniform(0.95, 1.05)
        rapido_fare = predicted_fare * np.random.uniform(0.80, 0.92)
        comp_df = pd.DataFrame({
            "Platform":    ["🚖 Uber", "🟢 Ola", "🔵 Rapido"],
            "Fare (₹)":    [f"₹{predicted_fare:.0f}", f"₹{ola_fare:.0f}", f"₹{rapido_fare:.0f}"],
            "Surge":       [f"{surge}x", f"{surge*0.95:.1f}x", f"{surge*0.8:.1f}x"],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.success(f"✅ Model Accuracy: {model_data.get('r2',0)*100:.2f}% R² Score")

    elif predict_btn and not model_data:
        st.error("❌ Model not found. Please run `python models/train_model.py` first.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#8892b0;border:2px dashed #0f3460;border-radius:16px;">
            <h2 style="font-size:3rem;margin:0;">🚖</h2>
            <p style="margin:1rem 0 0;">Set your trip details on the left<br>and click <b>Predict Fare</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Live surge indicator
    st.markdown("#### ⚡ Live Surge Conditions")
    hours_range = list(range(24))
    surge_vals = [compute_surge(h, day, weather) for h in hours_range]
    chart_df = pd.DataFrame({"Hour": hours_range, "Surge Multiplier": surge_vals})
    st.line_chart(chart_df.set_index("Hour"), color="#e94560")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.85rem;padding:1rem;">
    🚖 Taxi Fare Prediction System | Built with Random Forest & Gradient Boosting |
    Streamlit Deployed | Simulates Uber & Ola Dynamic Pricing
</div>
""", unsafe_allow_html=True)
