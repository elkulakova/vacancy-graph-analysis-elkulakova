# https://api.hh.ru/openapi/redoc#section/Obshaya-informaciya

import requests
import pandas as pd
import time

import session


def get_vacancies(pages, tag):
    """
    Функция принимает на вход количество страниц, с которых надо собрать информацию (pages)
    и ключевое слово, по которому должен осуществляться поиск (tag).
    Вернуть необходимо список уникальных по айди вакнсий.
    """
    url = 'https://api.hh.ru/vacancies'
    headers = {"User-Agent": "example@yandex.ru"} 
    # put your code here
    pass


if __name__ == "__main__":
    vacancies = get_vacancies(3, "python")
    df = pd.DataFrame(vacancies, columns=["id", "title", "requirement", "responsibility"])
    df.to_csv("python_300_vac.csv", index=False)
