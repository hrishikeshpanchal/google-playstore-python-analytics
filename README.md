# 📱 Google Play Store Data Analytics Dashboard (Python + Streamlit)

This project visualizes insights from Google Play Store app data using Python, Plotly, and Streamlit. It includes 7 interactive charts and a time-aware dashboard that renders specific charts only during their allowed time windows in IST.

---

## 📊 Visualizations Included

| Chart Type             | Description                                                                 | Time Visibility (IST)         |
|------------------------|-----------------------------------------------------------------------------|-------------------------------|
| **Sentiment Bar Chart** | Stacked bar chart of positive/neutral/negative sentiment vs rating groups   | Always visible                |
| **Paid App Scatter**    | Installs vs revenue (scatter plot with trendline)                           | Always visible                |
| **Dual Axis Chart**     | Avg installs vs revenue (Free vs Paid apps, top 3 categories)               | 🕐 1 PM – 2 PM IST             |
| **Choropleth Map**      | Installs by category & region (filtered)                                    | 🕐 6 PM – 8 PM IST             |
| **Grouped Bar Chart**   | Avg rating & total reviews by top 10 categories                             | 🕐 3 PM – 5 PM IST             |
| **Time Series Chart**   | Monthly installs with >20% MoM growth shaded                                | 🕐 6 PM – 9 PM IST             |
| **Bubble Chart**        | App size vs rating (bubble size = installs) with filters & translations     | 🕐 5 PM – 7 PM IST             |

---

## 📁 Folder Structure

```
google-playstore-python-analytics/
├── dashboard_app.py                  # Streamlit app (run this!)
├── playstore_analytics.py           # Script to generate chart HTML files
├── requirements.txt                 # Python dependencies
├── README.md                        # You're here :)
├── /charts/
│   ├── *.html (chart files)
├── /data/
│   ├── Play Store Data.csv
│   ├── User Reviews.csv
```

---

## ▶️ How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/google-playstore-python-analytics.git
cd google-playstore-python-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If running for the first time, also run:

```bash
python -m nltk.downloader vader_lexicon
```

### 3. Generate chart files (if not already present)

```bash
python playstore_analytics.py
```

### 4. Run the dashboard

```bash
streamlit run dashboard_app.py
```

Open the local URL shown in terminal (typically `http://localhost:8501`).

---

## 🌍 Time-Based Chart Control

The dashboard shows or hides each chart based on your **current time in IST**, using Streamlit and `pytz`.

If a chart is outside its allowed time, you’ll see a lock message like:

> 🔒 This chart is only visible between 6 PM and 9 PM IST.

---

## 🛠 Tools Used

- Python (Pandas, Plotly, Numpy, TextBlob, NLTK)
- Streamlit (dashboard interface)
- Plotly Express & Graph Objects (charts)
- Jupyter Notebooks (initial analysis)
- Git & GitHub

---

## 📌 Notes

- All charts are interactive `.html` files rendered inline using Streamlit components.
- The dashboard is responsive and minimal, built using Python only.
- To deploy this project online, use [Streamlit Cloud](https://streamlit.io/cloud).

---

## 👨‍💻 Developed by

**Your Name**  
GitHub: [yourusername](https://github.com/yourusername)
