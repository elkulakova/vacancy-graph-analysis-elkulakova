"""Сбор данных о вакансиях с hh.ru"""

# https://api.hh.ru/openapi/redoc#section/Obshaya-informaciya

import time

import pandas as pd
import requests

import session


def get_vacancies(pages, tag):
    """
    Функция принимает на вход количество страниц, с которых надо собрать информацию (pages)
    и ключевое слово, по которому должен осуществляться поиск (tag).
    Вернуть необходимо список уникальных по айди вакансий.
    """
    url = "https://api.hh.ru/vacancies"
    headers = {"User-Agent": "example@yandex.ru"}
    found_vacancies = []
    seen_ids = set()

    s = session.Session(url)

    try:
        response_try = s.get(url="", params={"text": tag, "per_page": 100}, headers=headers)
        response_try.raise_for_status()

        for p in range(pages):
            response = s.get(url="", params={"text": tag, "per_page": 100, "page": p}, headers=headers)
            response.raise_for_status()

            for r in response.json()["items"]:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    found_vacancies.append(
                        (r["id"], r["name"], r["snippet"].get("requirement"), r["snippet"].get("responsibility"))
                    )

            time.sleep(0.1)

    except requests.exceptions.RequestException as e:
        print("Error occurred while pooling data:")
        raise e

    return found_vacancies


if __name__ == "__main__":
    vacancies = get_vacancies(3, "python")
    df = pd.DataFrame(vacancies, columns=["id", "title", "requirement", "responsibility"])
    df.to_csv("python_300_vac.csv", index=False)
