from __future__ import annotations

import json
from pathlib import Path

import docs.build_wordpress_qos_series as series


def test_series_sources_and_claim_boundary() -> None:
    sources = sorted(series.SOURCE_DIR.glob("[0-9][0-9]_*.md"))
    english_sources = sorted(series.SOURCE_DIR_EN.glob("[0-9][0-9]_*.md"))
    assert len(sources) == 9
    assert len(english_sources) == 9
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    combined_english = "\n".join(
        path.read_text(encoding="utf-8") for path in english_sources
    )
    assert "32.738" in combined
    assert "24.576" in combined
    assert "0,50000" in combined
    assert "0,53125" in combined
    assert "17/32" in combined
    assert "0,43750" in combined
    assert "26 quantumseconden" in combined
    assert "QOS-geïnspireerde" in combined
    assert "genenzoekmachine" in combined
    assert "time-to-feature-generation advantage" in combined
    assert "99,1" in combined
    assert "we are not building a gene-search engine" in combined_english
    assert "17/32" in combined_english
    assert "0.43750" in combined_english
    assert "26 quantum seconds" in combined_english
    assert "time-to-feature-generation advantage" in combined_english
    assert "99.1" in combined_english
    assert "no held-out predictive quantum advantage" in combined_english


def test_builder_produces_hub_and_eight_navigable_articles(tmp_path: Path) -> None:
    original_out = series.OUT
    original_out_en = series.OUT_EN
    try:
        series.OUT = tmp_path / "nl"
        series.OUT_EN = tmp_path / "en"
        series.main()
    finally:
        series.OUT = original_out
        series.OUT_EN = original_out_en
    manifest = json.loads(
        (tmp_path / "nl" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_en = json.loads(
        (tmp_path / "en" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["hub"]["slug"] == "quantum-oracle-sketching-qml-genexpressie"
    assert manifest_en["hub"]["slug"] == "quantum-oracle-sketching-qml-gene-expression"
    assert len(manifest["articles"]) == 8
    assert len(manifest_en["articles"]) == 8
    hub = (tmp_path / "nl" / "series_page.html").read_text(encoding="utf-8")
    hub_en = (tmp_path / "en" / "series_page.html").read_text(encoding="utf-8")
    assert "qos-language-choice" in hub
    assert 'href="#nederlands"' in hub
    assert 'href="#english"' in hub
    assert "quantum-oracle-sketching-qml-gene-expression" in hub
    assert "qos-series-nav" in hub_en
    assert "quantum-oracle-sketching-qml-genexpressie" in hub_en
    for index in range(1, 9):
        article = (tmp_path / "nl" / "articles" / f"{index:02d}.html").read_text(
            encoding="utf-8"
        )
        article_en = (
            tmp_path / "en" / "articles" / f"{index:02d}.html"
        ).read_text(
            encoding="utf-8"
        )
        assert article.count("qos-series-nav") == 2
        assert article_en.count("qos-series-nav") == 2
        assert "<strong>Nederlands</strong>" in article
        assert ">Projectpagina</a>" in article
        assert "<strong>English</strong>" in article_en
        assert ">Project page</a>" in article_en
        assert "https://arxiv.org/abs/2604.07639" in article
        assert "https://arxiv.org/abs/2604.07639" in article_en
        assert "](" not in article
        assert "](" not in article_en
        assert "</strong>*" not in article
        assert "</strong>*" not in article_en


def test_github_pages_landing_page_contains_bilingual_result() -> None:
    page = (series.ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    stylesheet = (series.ROOT / "docs" / "pages.css").read_text(encoding="utf-8")
    assert 'data-language="nl"' in page
    assert 'data-language="en"' in page
    assert "17/32" in page
    assert "16/32" in page
    assert "14/32" in page
    assert "26 quantumseconden" in page
    assert "26 quantum seconds" in page
    assert "99,1" in page
    assert "99.1" in page
    assert "2335848" in page
    assert "@media (max-width: 720px)" in stylesheet
