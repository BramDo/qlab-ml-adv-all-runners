from __future__ import annotations

import json
from pathlib import Path

import docs.build_wordpress_qos_series as series


def test_series_sources_and_claim_boundary() -> None:
    sources = sorted(series.SOURCE_DIR.glob("[0-9][0-9]_*.md"))
    assert len(sources) == 8
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "32.738" in combined
    assert "24.576" in combined
    assert "0,50000" in combined
    assert "0,53125" in combined
    assert "geen empirisch quantumvoordeel" in combined
    assert "QOS-geïnspireerde" in combined
    assert "genenzoekmachine" in combined


def test_builder_produces_hub_and_seven_navigable_articles(tmp_path: Path) -> None:
    original_out = series.OUT
    try:
        series.OUT = tmp_path
        series.main()
    finally:
        series.OUT = original_out
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hub"]["slug"] == "quantum-oracle-sketching-qml-genexpressie"
    assert len(manifest["articles"]) == 7
    hub = (tmp_path / "series_page.html").read_text(encoding="utf-8")
    assert "qos-series-nav" in hub
    for index in range(1, 8):
        article = (tmp_path / "articles" / f"{index:02d}.html").read_text(
            encoding="utf-8"
        )
        assert article.count("qos-series-nav") == 2
        assert "https://arxiv.org/abs/2604.07639" in article
        assert "](" not in article
