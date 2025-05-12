"""Реализация класса сессии для отслеживания ошибок при запросах"""

import typing as tp

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore


class Session:
    """
    Сессия.

    :param base_url: Базовый адрес, на который будут выполняться запросы.
    :param timeout: Максимальное время ожидания ответа от сервера.
    :param max_retries: Максимальное число повторных запросов.
    :param backoff_factor: Коэффициент экспоненциального нарастания задержки.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def get(self, url: str, *args: tp.Any, **kwargs: tp.Any) -> requests.Response:
        """
        Проверка get-запроса
        """
        whole_url = f"{self.base_url}/{url}".lstrip("/")

        s = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=args if args else [500, 502, 503, 504, 408, 429],
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))

        for _ in range(self.max_retries + 1):
            try:
                response = s.get(
                    url=whole_url,
                    params=kwargs.get("params"),
                    headers=kwargs.get("headers"),
                    timeout=self.timeout,
                    **{k: v for k, v in kwargs.items() if k not in ("params", "headers")},
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                raise e
        return s.get(url=whole_url, params=kwargs.get("params"), headers=kwargs.get("headers"), timeout=self.timeout, **{k: v for k, v in kwargs.items() if k not in ("params", "headers")},)

    def post(self, url: str, *args: tp.Any, **kwargs: tp.Any) -> requests.Response:
        """
        Проверка post-запроса
        """
        whole_url = f"{self.base_url}/{url}".lstrip("/")

        s = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=args if args else [500, 502, 503, 504, 408, 429],
        )
        s.mount("https://", HTTPAdapter(max_retries=retries))
        s.mount("http://", HTTPAdapter(max_retries=retries))

        for _ in range(self.max_retries + 1):
            try:
                response = s.post(
                    url=whole_url,
                    params=kwargs.get("params"),
                    headers=kwargs.get("headers"),
                    timeout=self.timeout,
                    **{k: v for k, v in kwargs.items() if k not in ("params", "headers")},
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                raise e
        return s.post(url=whole_url, params=kwargs.get("params"), headers=kwargs.get("headers"), timeout=self.timeout, **{k: v for k, v in kwargs.items() if k not in ("params", "headers")},)
