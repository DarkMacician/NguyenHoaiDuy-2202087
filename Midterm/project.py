import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import time
import requests
import pymongo

# MongoDB connection
client = pymongo.MongoClient(
    "mongodb+srv://hoaiduy:introdatabase2024@cluster0.kvp0p.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    serverSelectionTimeoutMS=5000  # Set timeout for stability
)
try:
    client.server_info()  # Check if the connection works
    print("✅ Connected to MongoDB!")
except Exception as e:
    print("❌ MongoDB Connection Error:", e)

db = client["Sloth"]
collection = db["data"]

# Binance API endpoint
BINANCE_API_URL = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1s&limit=1"

# Dash app layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Real-Time BTC Price Chart", style={'textAlign': 'center'}),
    dcc.Graph(
        id='live-chart',
        config={'displayModeBar': False}  # Hide toolbars
    ),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every second
        n_intervals=0
    )
])

def fetch_and_store_data():
    response = requests.get(BINANCE_API_URL)
    if response.status_code == 200:
        data = response.json()
        latest_entry = data[-1]  # Get the latest price record

        record = {
            "timestamp": int(latest_entry[0]),
            "price": float(latest_entry[4])
        }

        # Insert only the latest record (avoid duplicates)
        collection.update_one({"timestamp": record["timestamp"]}, {"$set": record}, upsert=True)

@app.callback(
    Output('live-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_chart(n):
    fetch_and_store_data()  # Fetch only the latest data

    cursor = collection.find().sort("timestamp", -1).limit(100)
    data = list(cursor)[::-1]  # Reverse order to keep ascending time

    if not data:
        print("No data available yet!")
        return go.Figure()  # Return an empty figure

    x_data = [time.strftime('%H:%M:%S', time.gmtime(item["timestamp"] // 1000)) for item in data]
    y_data = [item["price"] for item in data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        marker=dict(size=6, color='white', line=dict(width=2, color='blue')),
        line=dict(shape='spline', smoothing=0.5),
        connectgaps=True,
        hovertemplate="<b>Time:</b> %{x}<br><b>Price:</b> %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Live BTC Price Stream",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis=dict(type="category"),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="x unified"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)