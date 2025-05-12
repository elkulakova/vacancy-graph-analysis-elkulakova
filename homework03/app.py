"""Оформление веб-приложения для демонстрации проанализированных данных"""

from copy import deepcopy

import dash
import networkx as nx
import pandas as pd
import plotly.express as px  # type: ignore
from dash import dash_table, dcc, html # type: ignore
from flask import Flask, render_template

from network import *

# загрузка данных: эту часть необходимо изменить и дополнить в соответствии с вашими данными
#orig_df = pd.read_csv("python_300_vac.csv")
#orig_df["tokens"] = orig_df.requirement.apply(preprocess_text)
#kdf = orig_df

# для сокращения времени выполнения кода заранее сохранила файл с токенами
kdf = pd.read_csv("python_300_vac_tokens.csv")

df = kdf["title"].value_counts().reset_index()[:10]
fig = px.bar(df, x="title", y="count", title="Самые частые вакансии")

# ищем ключевые слова
keywords = get_keywords(kdf)

# отдельный дф с ключевыми словами
keywords_show = deepcopy(keywords)
keywords_show.keywords = keywords_show.keywords.apply(", ".join)


# создание стартовой страницы
server = Flask(__name__)


@server.route("/")
def index():
    """Настройки главной страницы"""
    return render_template("index.html")


# страница со статистикой по исходным данным
dash_dashboard_app = dash.Dash(
    __name__, server=server, url_base_pathname="/dashboard/", suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(
    style={"fontFamily": "Segoe UI", "textAlign": "center", "padding": "10px", "backgroundColor": "#f0f8ff"},
    children=[
        html.H2("📊 Исходные данные"),
        html.A("← Назад", href="/", style={"color": "#28a745", "textDecoration": "none", "fontSize": "1.1em"}),
        dcc.Graph(figure=fig, style={"marginBottom": "10px", "marginTop": "10px"}),
        dash_table.DataTable( # type: ignore
            data=keywords_show.to_dict("records"),
            columns=[{"name": i, "id": i} for i in keywords_show.columns],
            style_cell={"textAlign": "center", "padding": "1px"},
            style_header={"backgroundColor": "#28a745", "color": "white", "fontWeight": "bold"},
            style_table={"width": "100%", "margin": "0 auto"},
        ),
        html.Br(),
    ],
)


# строим связи
graph_edges = create_network(keywords)
# сам граф
G: nx.Graph = nx.Graph()
G.add_edges_from(graph_edges)
# сообщества
comms, subgr = get_communities(G)
whole_graph = plot_communities(comms, subgr)
# визуализация каждого из сообществ отдельно с навыками
vis_data = []
for i, comm in enumerate(comms):
    subgraph = plot_one_community(subgr, comm, i + 1)
    skills_df = frequently_used_skills(comm, keywords)[:10]
    subfig = px.bar(skills_df, x="skill", y="frequency", title="Самые частые навыки")
    vis_data.append((subgraph, subfig))

# страница с визуализацией графов
dash_dashboard_app = dash.Dash(
    __name__, server=server, url_base_pathname="/network/", suppress_callback_exceptions=True
)

# эту часть необходимо дописать
dash_dashboard_app.layout = html.Div(
    style={"fontFamily": "Segoe UI", "textAlign": "center", "padding": "10px", "backgroundColor": "#f0f8ff"},
    children=[
        html.H2("📊 Аналитика данных"),
        html.A("← Назад", href="/", style={"color": "#28a745", "textDecoration": "none", "fontSize": "1.1em"}),
        html.H3("🌀 Граф всех вакансий"),
        dcc.Graph(figure=whole_graph, style={"marginBottom": "5px", "marginTop": "5px"}),
        *[
            html.Div(
                [
                    html.H3(f"🌀 Граф вакансий сообщества №{n+1}"),
                    dcc.Graph(figure=subgr, style={"marginBottom": "5px", "marginTop": "5px"}),
                    dcc.Graph(figure=skills, style={"marginBottom": "10px", "marginTop": "10px"}),
                ]
            )
            for n, (subgr, skills) in enumerate(vis_data)
        ],
        html.Br(),
    ],
)

# запуск приложения
if __name__ == "__main__":
    server.run(debug=False)
