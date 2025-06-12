import streamlit as st
from datetime import datetime
import pytz
import os

# Set page config
st.set_page_config(page_title="Google Play Store Dashboard", layout="wide")

# Get current IST time
ist = pytz.timezone('Asia/Kolkata')
now = datetime.now(ist)
hour = now.hour

st.title("📱 Google Play Store Data Analytics Dashboard")
st.caption(f"🕒 Current IST Time: {now.strftime('%I:%M %p')}")

def show_chart(path, label):
    st.subheader(label)
    st.components.v1.html(open(path, 'r', encoding='utf-8').read(), height=600, scrolling=True)

charts_dir = "charts"

# Always show these
show_chart(os.path.join(charts_dir, "sentiment_distribution.html"), "1. Sentiment Distribution")
show_chart(os.path.join(charts_dir, "revenue_vs_installs_paid_apps.html"), "2. Revenue vs Installs (Paid Apps)")

# Time-based charts
if 13 <= hour < 14:
    show_chart(os.path.join(charts_dir, "dual_axis_chart.html"), "3. Dual-Axis Chart")
else:
    st.warning("🔒 3. Dual-Axis Chart is only visible between 1 PM and 2 PM IST.")

if 18 <= hour < 20:
    show_chart(os.path.join(charts_dir, "choropleth_installs_by_category.html"), "4. Choropleth Map")
else:
    st.warning("🔒 4. Choropleth Map is only visible between 6 PM and 8 PM IST.")

if 15 <= hour < 17:
    show_chart(os.path.join(charts_dir, "grouped_bar_chart.html"), "5. Grouped Bar Chart")
else:
    st.warning("🔒 5. Grouped Bar Chart is only visible between 3 PM and 5 PM IST.")

if 18 <= hour < 21:
    show_chart(os.path.join(charts_dir, "time_series_installs_chart.html"), "6. Time Series Chart")
else:
    st.warning("🔒 6. Time Series Chart is only visible between 6 PM and 9 PM IST.")

if 17 <= hour < 19:
    show_chart(os.path.join(charts_dir, "bubble_chart_app_size_rating.html"), "7. Bubble Chart")
else:
    st.warning("🔒 7. Bubble Chart is only visible between 5 PM and 7 PM IST.")
