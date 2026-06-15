"""Unit tests for Box 1 data-prep connector tools (no network, no API keys)."""

import json

import pandas as pd

from data_prep.connectors.financial_news import tools as news_tools


def _point_asset_dir(monkeypatch, path):
    monkeypatch.setattr(news_tools, "get_asset_dir", lambda: str(path))


def test_check_news_cache_miss(monkeypatch, tmp_path):
    _point_asset_dir(monkeypatch, tmp_path)
    result = news_tools.check_news_cache("20260101", "20260131")
    assert "CACHE MISS" in result


def test_save_then_cache_hit_roundtrip(monkeypatch, tmp_path):
    _point_asset_dir(monkeypatch, tmp_path)
    articles = [{"headline": "Markets rally", "date": "20260102"}]

    save_msg = news_tools.save_news_to_csv(json.dumps(articles), "20260101", "20260131")
    assert "SUCCESS" in save_msg

    saved = tmp_path / "financial_news_20260101_20260131.csv"
    assert saved.exists()
    assert len(pd.read_csv(saved)) == 1

    hit = news_tools.check_news_cache("20260101", "20260131")
    assert "CACHE HIT" in hit and "1 news" in hit


def test_save_empty_articles(monkeypatch, tmp_path):
    _point_asset_dir(monkeypatch, tmp_path)
    assert "No articles" in news_tools.save_news_to_csv("[]", "20260101", "20260131")


def test_save_bad_json_returns_error(monkeypatch, tmp_path):
    _point_asset_dir(monkeypatch, tmp_path)
    assert "ERROR" in news_tools.save_news_to_csv("{not json", "20260101", "20260131")
