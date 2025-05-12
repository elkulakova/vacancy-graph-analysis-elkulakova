"""Анализ данных, выделение ключевых слов и создание графов"""

import math
import random
import re
from collections import Counter
from itertools import chain

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import spacy
from matplotlib import colormaps
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore


def preprocess_text(text: str) -> str:
    """
    Принимает на вход текст с требуемыми навыками.
    Выполняет очистку, лемматизацию и фильтрацию по части речи.
    Возвращает отфильтрованные леммы через пробел без запятых сплошным текстом.
    """
    if not isinstance(text, str):
        text = " "
        return text
    nlp = spacy.load("ru_core_news_sm")
    word_pat = r"[^A-Za-zА-Яа-яЁё-]"
    clear_text = re.sub(word_pat, " ", text)
    clear_text = re.sub(r" - ", " ", clear_text)
    clear_text = re.sub(r"highlighttext", "", clear_text)
    doc = nlp(clear_text)

    preprocessed_text = " ".join(map(str, [token.lemma_ for token in doc if token.pos_ in ("NOUN", "PROPN", "X")]))
    return preprocessed_text


def get_keywords(df, n_keywords=5):
    """
    Принимает на вход датафрейм с вакансиями и полем обработанных навыков.
    Возвращает датафрейм, состоящий из двух столбцов: название вакансии и столбец с ключевыми словами.
    df: входной датафрейм
    n_keywords: число ключевых слов, которое надо извлечь
    """
    # а по-другому и не передать обработанный дф...
    if "tokens" not in df.columns:
        df["tokens"] = df.requirement.apply(preprocess_text)
        df.to_csv("python_300_vac_tokens.csv", index=False)
        print("done")

    tokens = df.tokens.to_list()
    vectorizer = TfidfVectorizer()
    vectorized_text = vectorizer.fit_transform(tokens)
    feat_names = vectorizer.get_feature_names_out()

    dense_matrix = vectorized_text.todense()

    keywords = []
    for doc in range(len(tokens)):
        vac = df.iloc[doc].title
        vac_keywords = {}
        for w, word in enumerate(feat_names):
            tfidf_value = dense_matrix[doc, w]
            vac_keywords[word] = (
                tfidf_value
                # тут может получиться меньше 5 слов, которые важны значительно, поэтому выписываем все слова, чтобы точно 5 нашлись
            )
        vac_keywords = sorted(
            vac_keywords, key=vac_keywords.get, reverse=True
        )  # нам нужны только слова, так что так и делаем
        # vac_keys[vac] = " ".join(map(str, vac_dict[:5]))
        keywords.append((vac, vac_keywords[:n_keywords]))

    graf_df = pd.DataFrame(keywords, columns=["title", "keywords"])

    return graf_df


def create_network(df):
    """
    Принимает на вход датафрейм с вакансиями и ключевыми словами.
    Возвращает список кортежей из пар вакансий и количества их общих ключевых слов.
    Вид кортежа внутри списка ожидается такой: (ребро1, ребро2, {'weight': вес_ребра})
    """
    df.columns = ["title", "keywords"]
    vacancies = df.title.unique()
    checked_edges = set()
    edges = []
    for v1 in vacancies:
        for v2 in vacancies:
            if (tuple(sorted([v1, v2])) not in checked_edges) and (v1 != v2):
                pair = tuple(sorted([v1, v2]))
                checked_edges.add(pair)
                vac_1, vac_2 = pair
                v1_keys = set(
                    chain(*df[df.title == v1].keywords.tolist())
                )  # тут такая сложная махинация, потому что вакансии в плане названий могут повторяться, поэтому все навыки надо собрать со всех индексов дф
                v2_keys = set(chain(*df[df.title == v2].keywords.tolist()))
                weight = len(v1_keys.intersection(v2_keys))
                if weight > 0:
                    edges.append((vac_1, vac_2, {"weight": weight}))
    edges = sorted(edges)
    return edges


def plot_network(vac_edges):
    """
    Строит визуализацию графа с помощью matplotlib.
    """
    G = nx.Graph()
    G.add_edges_from(vac_edges)
    nx.draw(G, with_labels=False, font_weight="bold", node_size=30)
    plt.show()


def get_communities(graph):
    """
    Извлекает сообщества из графа и фильтрует их так,
    чтобы оставались только сообщества с более чем 5 узлами.
    Возвращает граф и сообщества в формате датафрейма.
    """
    communities = nx.community.louvain_communities(graph, resolution=1.2)
    comm_data = [{"n_of_nodes": len(comm), "nodes": comm} for comm in communities]
    cdf = pd.DataFrame(comm_data)
    nodes = list(chain(*cdf["nodes"].tolist()))
    S = graph.subgraph(nodes)
    communities = cdf.query("n_of_nodes>5")["nodes"].tolist()
    nodes = list(chain(*cdf.query("n_of_nodes>5")["nodes"].tolist()))
    s = S.subgraph(nodes)
    return communities, s


def create_community_node_colors(graph, communities, gr_type="multi"):
    """
    Создает список цветов, соответствующих вершинам графа
    """
    cmap = colormaps.get_cmap("tab20b")
    colors = cmap(np.arange(len(communities)))
    colors = [mcolors.rgb2hex(color[:3]) for color in colors]
    node_colors = []
    if gr_type == "multi":
        for node in graph:
            current_community_index = 0
            for community in communities:
                if node in community:
                    node_colors.append(colors[current_community_index])
                    break
                current_community_index += 1
        return node_colors

    node_color = random.choice(colors)
    node_colors = [node_color] * len(communities)
    return node_colors


def plot_communities(communities, graph):
    """
    Строит интерактивную визуализацию графа с сообществами с помощью plotly.
    """
    pos = nx.spring_layout(graph, iterations=1000, seed=30, k=3 / math.sqrt(len(graph)), scale=10.0)

    # edges coordinates
    x_nodes, y_nodes = zip(*pos.values())
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # degree labels
    node_labels = [f"Node {n}<br>Degree: {graph.degree(n)}" for n in graph.nodes()]
    node_degrees = [graph.degree(n) for n in graph.nodes()]
    node_colors_list = create_community_node_colors(graph, communities)

    fig = go.Figure()
    # add edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="gray"),
        )
    )

    # add nodes
    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers",
            marker=dict(size=[deg * 0.5 for deg in node_degrees], color=node_colors_list),
            hoverinfo="text",
            text=node_labels,
        )
    )

    # background settings
    fig.update_layout(
        title="Interactive Visualization of All Suitable Communities",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    # fig.show()
    return fig


def plot_one_community(graph, community, number_of_comm):
    """
    Строит график в plotly для одного сообщества.
    """
    subgraph = graph.subgraph(list(community))
    pos = nx.spring_layout(subgraph, iterations=1000, seed=30, k=3 / math.sqrt(len(subgraph)), scale=10.0)

    # edges coordinates
    x_nodes, y_nodes = zip(*pos.values())
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # degree labels
    node_labels = [f"Node {n}<br>Degree: {subgraph.degree(n)}" for n in subgraph.nodes()]
    node_degrees = [subgraph.degree(n) for n in subgraph.nodes()]
    node_color = create_community_node_colors(subgraph, community, gr_type="one")

    fig = go.Figure()
    # add edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="gray"),
        )
    )

    # add nodes
    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers",
            marker=dict(size=[deg * 2.5 for deg in node_degrees], color=node_color),
            hoverinfo="text",
            text=node_labels,
        )
    )

    # background settings
    fig.update_layout(
        title=f"Interactive Visualization of Community №{number_of_comm}",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    # fig.show()
    return fig


def frequently_used_skills(community, kdf):
    """
    Строит дф из навыков для сообщества, сортируя от наиболее часто требуемых к менее часто требуемым
    """
    all_skills = []
    for vacancy in community:
        all_skills.extend(list(chain(*kdf[kdf.title == vacancy].keywords.tolist())))
    skills = Counter(all_skills)
    analytics_df = pd.DataFrame({"skill": key, "frequency": value} for key, value in skills.items())
    analytics_df = analytics_df.sort_values(by=["frequency"], ascending=False)
    return analytics_df


if __name__ == "__main__":
    # импортируем дф
    # my_vacancies = pd.read_csv('python_300_vac.csv')
    my_vacancies = pd.read_csv("python_300_vac_tokens.csv")

    # ищем ключевые слова
    keywrds = get_keywords(my_vacancies)

    # строим связи
    graf_edges = create_network(keywrds)

    # граф обычный самый
    # plot_network(graf_edges)

    # сам граф
    graf: nx.Graph = nx.Graph()
    graf.add_edges_from(graf_edges)

    # сообщества
    comms, ss = get_communities(graf)

    # whole_graph = plot_communities(comms, ss)
    # whole_graph.show()

    # визуализация одного из сообществ
    # comm = comms[3]
    # comm3 = plot_one_community(ss, comm, 4)
    # comm3.show()
