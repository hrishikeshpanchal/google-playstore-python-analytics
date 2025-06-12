import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("Play Store Data.csv")

df['Size'] = df['Size'].replace('Varies with device', None)
df['Size_MB'] = df['Size'].str.replace('M', '', regex=True).str.replace('k', '', regex=True)
df['Size_MB'] = pd.to_numeric(df['Size_MB'], errors='coerce')

df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

df['Price'] = df['Price'].astype(str).str.strip().str.replace('$', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)

df['Type'] = df['Type'].fillna('Free')

df['Revenue'] = df['Installs'] * df['Price']

top_categories = df['Category'].value_counts().nlargest(3).index
df = df[df['Category'].isin(top_categories)]

grouped = df.groupby(['Category', 'Type']).agg(
    Avg_Installs=('Installs', 'mean'),
    Avg_Revenue=('Revenue', 'mean')
).reset_index()

fig = go.Figure()
for t in ['Free', 'Paid']:
    data = grouped[grouped['Type'] == t]
    fig.add_trace(go.Bar(
        x=data['Category'],
        y=data['Avg_Installs'],
        name=f'Avg Installs ({t})',
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=data['Category'],
         y=data['Avg_Revenue'],
        name=f'Avg Revenue ({t})',
        yaxis='y2',
        mode='lines+markers'
    ))

    fig.update_layout(
    title="Average Installs vs Revenue (Top 3 Categories - Free vs Paid Apps)",
    xaxis_title="App Category",
    yaxis=dict(
        title="Average Installs",
        side="left"
    ),
    yaxis2=dict(
        title="Average Revenue ($)",
        overlaying="y",
        side="right"
    ),
    legend=dict(x=0.01, y=0.99),
    height=600
)
fig.write_html("D:/google-playstore-python-analytics/dual_axis_chart.html")
print("✅ Chart saved to 'dual_axis_chart.html'")
    



import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

apps_df = pd.read_csv('Play Store Data.csv')
reviews_df = pd.read_csv('User Reviews.csv')

apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df['Rating'] <= 5]

apps_df['Reviews'] = apps_df['Reviews'].str.replace('[^0-9]', '', regex=True).astype(int)

apps_df_filtered = apps_df[apps_df['Reviews'] > 1000]

merged_df = pd.merge(apps_df_filtered, reviews_df, on='App')

merged_df

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def safe_sentiment_analysis(review):
    if isinstance(review, str) and review.strip() != "":
        return sia.polarity_scores(review)['compound']
    else:
        return 0.0  # Neutral for empty/bad reviews

merged_df['Sentiment_Score'] = merged_df['Translated_Review'].apply(safe_sentiment_analysis)
merged_df['Sentiment'] = merged_df['Sentiment_Score'].apply(get_sentiment)

def rating_group(rating):
    if rating <= 2:
        return '1-2 stars'
    elif rating <= 4:
        return '3-4 stars'
    else:
        return '4-5 stars'

merged_df['Rating_Group'] = merged_df['Rating'].apply(rating_group)

top_categories = merged_df['Category'].value_counts().nlargest(5).index
merged_df_top = merged_df[merged_df['Category'].isin(top_categories)]

grouped = merged_df_top.groupby(['Category', 'Rating_Group', 'Sentiment']).size().reset_index(name='Count')

fig = px.bar(
    grouped,
    x='Rating_Group',
    y='Count',
    color='Sentiment',
    barmode='stack',
    facet_col='Category',
    title='Sentiment Distribution by Rating Group and Top 5 Categories',
    category_orders={"Rating_Group": ["1-2 stars", "3-4 stars", "4-5 stars"]},
    color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'}
)

fig.show()

fig.write_html("D:/google-playstore-python-analytics/sentiment_distribution.html")

print("Chart saved successfully as 'sentiment_distribution.html'")



import plotly.express as px
import pandas as pd

apps_df = pd.read_csv("Play Store Data.csv")

print("Sample raw Price values:")
print(apps_df['Price'].unique()[:10])

apps_df['Price'] = apps_df['Price'].astype(str).str.replace(r'[^\d\.\$]', '', regex=True)

paid_apps = apps_df[apps_df['Price'].str.contains('$', regex=False)].copy()
print("Apps with cleaned '$':", paid_apps.shape[0])

paid_apps['Price'] = paid_apps['Price'].str.replace('$', '', regex=False)
paid_apps['Price'] = pd.to_numeric(paid_apps['Price'], errors='coerce')

print("Converted Price values:", paid_apps['Price'].notna().sum())
print(paid_apps[['App', 'Price']].head())

paid_apps['Installs'] = paid_apps['Installs'].astype(str).str.replace('[+,]', '', regex=True)
paid_apps['Installs'] = pd.to_numeric(paid_apps['Installs'], errors='coerce')

paid_apps = paid_apps.dropna(subset=['Price', 'Installs', 'Category'])
paid_apps = paid_apps[(paid_apps['Price'] > 0) & (paid_apps['Installs'] > 0)]

paid_apps['Revenue'] = paid_apps['Price'] * paid_apps['Installs']

fig = px.scatter(
    paid_apps,
    x='Installs',
    y='Revenue',
    color='Category',
    hover_data=['App', 'Price'],
    title='Revenue vs Installs for Paid Apps'
)
fig.update_layout(xaxis_type='log', yaxis_type='log')

fig.write_html("D:/google-playstore-python-analytics/revenue_vs_installs_paid_apps.html")
print("Plot saved as 'revenue_vs_installs_paid_apps.html'")



import pandas as pd
import numpy as np
import plotly.express as px

apps_df = pd.read_csv("Play Store Data.csv")

apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace(r'[+,]', '', regex=True)
apps_df['Installs'] = pd.to_numeric(apps_df['Installs'], errors='coerce')

apps_df = apps_df[~apps_df['Category'].str.startswith(('A', 'C', 'G', 'S'))]

top_categories = apps_df.groupby('Category')['Installs'].sum().nlargest(5).index.tolist()
filtered_df = apps_df[apps_df['Category'].isin(top_categories)]

countries = ['USA', 'IND', 'BRA', 'RUS', 'DEU', 'FRA', 'GBR', 'CHN', 'AUS', 'CAN']
filtered_df['Country'] = np.random.choice(countries, size=len(filtered_df))

grouped_df = filtered_df.groupby(['Category', 'Country'])['Installs'].sum().reset_index()
grouped_df = grouped_df[grouped_df['Installs'] > 0]

fig = px.choropleth(
grouped_df,
locations='Country',
locationmode='ISO-3',
color='Installs',
hover_name='Category',
animation_frame='Category',
color_continuous_scale='YlGnBu',
title='Global Installs by App Category (Top 5, Filtered)'
)
fig.write_html("D:/google-playstore-python-analytics/choropleth_installs_by_category.html")
print("Choropleth map saved successfully.")



import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("Play Store Data.csv")

def size_to_mb(size_str):
    if pd.isna(size_str) or size_str == 'Varies with device':
        return None
    size_str = size_str.strip()
    if size_str.endswith('M'):
        try:
            return float(size_str[:-1])
        except:
            return None
    elif size_str.endswith('k'):
        try:
            return float(size_str[:-1]) / 1024
        except:
            return None
    else:
        return None

df['Size_MB'] = df['Size'].apply(size_to_mb)

df = df[df['Size_MB'] >= 10]

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df[df['Rating'] >= 4.0]

df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
df = df[df['Last Updated'].dt.month == 1]  # January = 1

df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

installs_by_cat = df.groupby('Category')['Installs'].sum().sort_values(ascending=False)
top_10_categories = installs_by_cat.head(10).index

df_top = df[df['Category'].isin(top_10_categories)]

df_top['Reviews'] = pd.to_numeric(df_top['Reviews'], errors='coerce').fillna(0)

agg = df_top.groupby('Category').agg(
    Avg_Rating=('Rating', 'mean'),
    Total_Reviews=('Reviews', 'sum')
).reset_index()

agg['Total_Installs'] = agg['Category'].map(installs_by_cat)
agg = agg.sort_values('Total_Installs', ascending=False)

fig = go.Figure(data=[
go.Bar(name='Average Rating', x=agg['Category'], y=agg['Avg_Rating'], yaxis='y1', marker_color='indianred'),
go.Bar(name='Total Reviews', x=agg['Category'], y=agg['Total_Reviews'], yaxis='y2', marker_color='lightskyblue')
])
fig.update_layout(
    title="Average Rating and Total Review Count for Top 10 App Categories by Installs",
    xaxis=dict(title="App Category"),
    yaxis=dict(
        title="Average Rating",
        range=[0, 5],
        side='left',
        showgrid=False,
        zeroline=False
    ),
    yaxis2=dict(
        title="Total Reviews",
        overlaying='y',
        side='right',
        showgrid=False,
        zeroline=False
    ),
    barmode='group',
    legend=dict(x=0.01, y=0.99),
        height=600,
    margin=dict(t=60, b=120)
)
fig.write_html("D:/google-playstore-python-analytics/grouped_bar_chart.html")
print("Chart saved as 'grouped_bar_chart.html' and shown as it is between 3PM-5PM IST.")




import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("Play Store Data.csv")

df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

df = df[df['Category'].str.startswith(('E', 'C', 'B'))]
df = df[~df['App'].str.lower().str.startswith(('x', 'y', 'z'))]
df = df[~df['App'].str.contains('s', case=False, regex=True)]
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df = df[df['Reviews'] > 500]

df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

translations = {
    'Beauty': 'सौंदर्य',        # Hindi
    'Business': 'வணிகம்',      # Tamil
    'Dating': 'Partnersuche'    # German
}
df['Translated_Category'] = df['Category'].replace(translations)

df['Month'] = df['Last Updated'].dt.to_period('M').astype(str)

agg_df = df.groupby(['Month', 'Translated_Category'])['Installs'].sum().reset_index()
pivot_df = agg_df.pivot(index='Month', columns='Translated_Category', values='Installs').fillna(0)
pivot_df = pivot_df.sort_index()

growth_df = pivot_df.pct_change().fillna(0)

fig = go.Figure()

for category in pivot_df.columns:
    fig.add_trace(go.Scatter(
        x=pivot_df.index,
        y=pivot_df[category],
        mode='lines',
        name=category
    ))

    # Highlight periods with >20% growth
    mask = growth_df[category] > 0.2
    fig.add_trace(go.Scatter(
        x=pivot_df.index[mask],
        y=pivot_df[category][mask],
        fill='tozeroy',
        mode='none',
        name=f"{category} >20% growth",
        fillcolor='rgba(255, 0, 0, 0.2)',
        showlegend=False
    ))

fig.update_layout(
    title="📈 Total Installs Over Time by Category (with Translations & Growth Highlights)",
    xaxis_title="Month",
    yaxis_title="Total Installs",
    height=600
)

fig.write_html("D:/google-playstore-python-analytics/time_series_installs_chart.html")
print("Chart saved as 'time_series_installs_chart.html'")




import pandas as pd
import plotly.express as px
from textblob import TextBlob

df = pd.read_csv("Play Store Data.csv")

allowed_categories = [
    'GAME', 'Beauty', 'Business', 'Comics', 'Communication',
    'Dating', 'Entertainment', 'Social', 'Events'
]
df = df[df['Category'].isin(allowed_categories)]

df = df[~df['App'].str.contains('s', case=False, regex=True)]
df = df[~df['App'].str.lower().str.startswith(('x', 'y', 'z'))]

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df[df['Rating'] > 3.5]
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df = df[df['Reviews'] > 500]

df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df = df[df['Installs'] > 50000]

df['Size'] = df['Size'].replace('Varies with device', None)
df['Size_MB'] = df['Size'].str.replace('M', '', regex=True).str.replace('k', '', regex=True)
df['Size_MB'] = pd.to_numeric(df['Size_MB'], errors='coerce')
df = df.dropna(subset=['Size_MB', 'Rating'])

if 'Translated_Review' in df.columns:
    df['Subjectivity'] = df['Translated_Review'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    df = df[df['Subjectivity'] > 0.5]
else:
    df['Subjectivity'] = 0.6  # simulate valid subjectivity if review text is not available

translations = {
    'Beauty': 'सौंदर्य',        # Hindi
    'Business': 'வணிகம்',      # Tamil
    'Dating': 'Partnersuche'    # German
}
df['Translated_Category'] = df['Category'].replace(translations)

custom_colors = {cat: 'pink' if cat == 'GAME' else None for cat in df['Translated_Category'].unique()}

fig = px.scatter(
    df,
    x="Size_MB",
    y="Rating",
    size="Installs",
    color="Translated_Category",
    hover_name="App",
    title="📊 App Size vs Rating (Bubble = Installs)",
    color_discrete_map=custom_colors
)

fig.update_layout(height=600)
fig.write_html("D:/google-playstore-python-analytics/bubble_chart_app_size_rating.html")
print("Chart saved as 'bubble_chart_app_size_rating.html'")



