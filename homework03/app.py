from flask import Flask, render_template
import dash
from dash import html, dcc, dash_table
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.colors as mcolors
from itertools import combinations, chain
import plotly.graph_objects as go

# загрузка данных: эту часть необходимо изменить и дополнить в соответствии с вашими данными
kdf = pd.read_csv("titles_keywords.csv")

df = kdf["title"].value_counts().reset_index()[:10]
fig = px.bar(df, x="title", y="count", title="Самые частые вакансии")

# создание стартовой страницы
server = Flask(__name__)

@server.route('/')
def index():
    return render_template("index.html")


# страница со статистикой по исходным данным
dash_dashboard_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dashboard/',
    suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(style={
    'fontFamily': 'Segoe UI',
    'textAlign': 'center',
    'padding': '10px',
    'backgroundColor': '#f0f8ff'
}, children=[
    html.H2("📊 Исходные данные"),
    html.A("← Назад", href='/', style={
        'color': '#28a745',
        'textDecoration': 'none',
        'fontSize': '1.1em'
    }),
    dcc.Graph(figure=fig, style={'marginBottom': '10px', 'marginTop': '10px'}),
    dash_table.DataTable(
        data=kdf.to_dict('records'),
        columns=[{"name": i, "id": i} for i in kdf.columns],
        style_cell={'textAlign': 'center', 'padding': '1px'},
        style_header={
            'backgroundColor': '#28a745',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_table={'width': '100%', 'margin': '0 auto'}
    ),
    html.Br(),
])


# страница с визуализацией графов
dash_dashboard_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/network/',
    suppress_callback_exceptions=True
)

# эту часть необходимо дописать
dash_dashboard_app.layout = html.Div()

# запуск приложения
if __name__ == '__main__':
    server.run(debug=False)
