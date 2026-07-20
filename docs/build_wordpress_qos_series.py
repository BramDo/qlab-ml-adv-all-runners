from __future__ import annotations

import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "articles" / "qos_edukaizen"
SOURCE_DIR_EN = ROOT / "articles" / "qos_edukaizen_en"
OUT = ROOT / "output" / "wordpress_qos_series"
OUT_EN = ROOT / "output" / "wordpress_qos_series_en"
SITE = "https://edukaizen.nl"
HUB_SLUG = "quantum-oracle-sketching-qml-genexpressie"
HUB_SLUG_EN = "quantum-oracle-sketching-qml-gene-expression"
GITHUB_BASE = "https://github.com/BramDo/qlab-ml-adv-all-runners"

ARTICLES = [
    (1, "01_WAT_IS_DE_QML_TAAK.md", "wat-is-de-qml-taak-cellen-classificeren", "Wat is de QML-taak? Cellen classificeren, geen genen opzoeken"),
    (2, "02_DE_THEORIE_VAN_QOS.md", "quantum-oracle-sketching-theorie", "De theorie van Quantum Oracle Sketching"),
    (3, "03_VAN_GENEXPRESSIE_NAAR_QUBITS.md", "pbmc68k-van-genexpressie-naar-qubits", "Van PBMC68k-genexpressie naar 40 qubits"),
    (4, "04_QOS_OP_ECHTE_HARDWARE.md", "qos-naar-40-qubit-hardware-fire-opal", "Van JAX naar een 40-qubit hardwarecircuit"),
    (5, "05_READOUT_EN_CLASSIFIER.md", "quantum-readout-405-observabelen-classifier", "405 observabelen en een lekvrije classifier"),
    (6, "06_HET_HARDWARERESULTAAT.md", "resultaat-hardware-versus-klassiek", "Het resultaat: hardware 16, klassiek 17"),
    (7, "07_WAT_NODIG_IS_VOOR_VOORDEEL.md", "route-naar-quantumvoordeel-qml", "Wat is nog nodig voor quantumvoordeel?"),
    (8, "08_VOORSTEL_60Q_VERVOLGSTUDIE.md", "voorstel-60-qubit-qml-vervolgstudie", "Voorstel: een 60-qubit QML-vervolgstudie"),
]

ARTICLES_EN = [
    (1, "01_WHAT_IS_THE_QML_TASK.md", "what-is-the-qml-task-cell-classification", "What is the QML task? Classifying cells, not looking up genes"),
    (2, "02_QUANTUM_ORACLE_SKETCHING_THEORY.md", "quantum-oracle-sketching-theory", "The theory of Quantum Oracle Sketching"),
    (3, "03_FROM_GENE_EXPRESSION_TO_QUBITS.md", "pbmc68k-from-gene-expression-to-qubits", "From PBMC68k gene expression to 40 qubits"),
    (4, "04_FROM_JAX_TO_HARDWARE.md", "qos-to-40-qubit-hardware-fire-opal", "From JAX to a 40-qubit hardware circuit"),
    (5, "05_READOUT_AND_CLASSIFIER.md", "quantum-readout-405-observables-classifier", "405 observables and a leakage-free classifier"),
    (6, "06_HARDWARE_RESULT.md", "result-hardware-versus-classical", "The result: hardware 16, classical 17"),
    (7, "07_ROUTE_TO_QUANTUM_ADVANTAGE.md", "route-to-quantum-advantage-qml", "What is still required for quantum advantage?"),
    (8, "08_PROPOSAL_60Q_FOLLOW_UP.md", "proposal-60-qubit-qml-follow-up-study", "Proposal: a 60-qubit QML follow-up study"),
]


def public_url(slug: str, hub_slug: str = HUB_SLUG) -> str:
    return f"{SITE}/{hub_slug}/{slug}/"


def normalize_link(target: str) -> str:
    if target.startswith("http://") or target.startswith("https://"):
        return target
    return f"{GITHUB_BASE}/blob/agent/add-q40-fire-opal-hardware-milestone/{target}"


def render_inline(text: str) -> str:
    pattern = re.compile(
        r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|(\$[^$]+\$)|\*\*([^*]+)\*\*|\*([^*]+)\*"
    )
    out: list[str] = []
    pos = 0
    for match in pattern.finditer(text):
        out.append(html.escape(text[pos : match.start()]))
        if match.group(1):
            out.append(f"<code>{html.escape(match.group(1))}</code>")
        elif match.group(2):
            out.append(
                f'<a href="{html.escape(normalize_link(match.group(3)), quote=True)}">'
                f"{render_inline(match.group(2))}</a>"
            )
        elif match.group(4):
            out.append(f"[latex]{html.escape(match.group(4)[1:-1])}[/latex]")
        elif match.group(5):
            out.append(f"<strong>{html.escape(match.group(5))}</strong>")
        else:
            out.append(f"<em>{html.escape(match.group(6))}</em>")
        pos = match.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def render_table(lines: list[str]) -> str:
    header = split_table_row(lines[0])
    rows = [split_table_row(line) for line in lines[2:]]
    out = ["<table>\n<thead><tr>"]
    out.extend(f"<th>{render_inline(cell)}</th>" for cell in header)
    out.append("</tr></thead>\n<tbody>\n")
    for row in rows:
        out.append("<tr>")
        out.extend(f"<td>{render_inline(cell)}</td>" for cell in row)
        out.append("</tr>\n")
    out.append("</tbody>\n</table>\n")
    return "".join(out)


def render_list(lines: list[str], ordered: bool) -> str:
    tag = "ol" if ordered else "ul"
    marker = re.compile(r"^\s*(?:-\s+|\d+\.\s+)")
    items = "".join(
        f"<li>{render_inline(marker.sub('', line).strip())}</li>\n" for line in lines
    )
    return f"<{tag}>\n{items}</{tag}>\n"


def render_markdown(lines: list[str], *, skip_h1: bool = True) -> str:
    out: list[str] = []
    i = 0
    if skip_h1:
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].startswith("# "):
            i += 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip() or "text"
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            if language == "math":
                out.append(
                    '[latex syntax="display"]\n'
                    + html.escape("\n".join(block))
                    + "\n[/latex]\n"
                )
            else:
                out.append(
                    f'<pre><code class="language-{html.escape(language)}">'
                    f"{html.escape(chr(10).join(block))}</code></pre>\n"
                )
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            table = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            out.append(render_table(table))
            continue
        heading = re.match(r"^(#{2,6})\s+(.+)$", stripped)
        if heading:
            level = min(len(heading.group(1)), 4)
            out.append(f"<h{level}>{render_inline(heading.group(2))}</h{level}>\n")
            i += 1
            continue
        if re.match(r"^\s*-\s+", lines[i]):
            items: list[str] = []
            while i < len(lines) and re.match(r"^\s*-\s+", lines[i]):
                item = lines[i]
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(item)
            out.append(render_list(items, ordered=False))
            continue
        if re.match(r"^\s*\d+\.\s+", lines[i]):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(lines[i])
                i += 1
            out.append(render_list(items, ordered=True))
            continue
        paragraph = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("```")
                or nxt.startswith("|")
                or re.match(r"^\s*(?:-|\d+\.)\s+", lines[i])
            ):
                break
            paragraph.append(nxt)
            i += 1
        out.append(f"<p>{render_inline(' '.join(paragraph))}</p>\n")
    return "".join(out)


def nav_html(part: int | None, language: str) -> str:
    english = language == "en"
    articles = ARTICLES_EN if english else ARTICLES
    other_articles = ARTICLES if english else ARTICLES_EN
    hub_slug = HUB_SLUG_EN if english else HUB_SLUG
    other_hub_slug = HUB_SLUG if english else HUB_SLUG_EN
    links = [
        f'<a href="{SITE}/{hub_slug}/">{"Series hub" if english else "Seriehub"}</a>'
    ]
    if part is not None and part > 1:
        links.append(
            f'<a href="{public_url(articles[part - 2][2], hub_slug)}">'
            f'{"Previous part" if english else "Vorig deel"}</a>'
        )
    if part is not None and part < len(articles):
        links.append(
            f'<a href="{public_url(articles[part][2], hub_slug)}">'
            f'{"Next part" if english else "Volgend deel"}</a>'
        )
    language_url = (
        f"{SITE}/{other_hub_slug}/"
        if part is None
        else public_url(other_articles[part - 1][2], other_hub_slug)
    )
    links.append(
        f'<a href="{language_url}">{"Nederlands" if english else "English"}</a>'
    )
    links.extend(
        [
            '<a href="https://arxiv.org/abs/2604.07639">QOS paper</a>',
            '<a href="https://github.com/haimengzhao/quantum-oracle-sketching">'
            + ("Official code" if english else "Officiële code")
            + "</a>",
            f'<a href="{GITHUB_BASE}">{"Hardware code" if english else "Hardwarecode"}</a>',
        ]
    )
    return (
        '<nav class="qos-series-nav" style="margin:1rem 0;padding:.75rem 0;'
        'border-top:1px solid #ddd;border-bottom:1px solid #ddd;">'
        + " | ".join(links)
        + "</nav>\n"
    )


def build_series(
    *,
    source_dir: Path,
    out: Path,
    hub_slug: str,
    articles: list[tuple[int, str, str, str]],
    language: str,
    hub_title: str,
) -> None:
    article_dir = out / "articles"
    article_dir.mkdir(parents=True, exist_ok=True)
    hub_filename = "00_SERIES_HUB.md" if language == "en" else "00_SERIEHUB.md"
    hub_source = source_dir / hub_filename
    hub_output = out / "series_page.html"
    hub_output.write_text(
        nav_html(None, language)
        + render_markdown(hub_source.read_text(encoding="utf-8").splitlines()),
        encoding="utf-8",
    )

    manifest_articles: list[dict[str, object]] = []
    for part, filename, slug, title in articles:
        source = source_dir / filename
        lines = source.read_text(encoding="utf-8").splitlines()
        content = (
            nav_html(part, language)
            + render_markdown(lines)
            + nav_html(part, language)
        )
        output = article_dir / f"{part:02d}.html"
        output.write_text(content, encoding="utf-8")
        paragraphs = [
            line.strip()
            for line in lines[1:]
            if line.strip() and not line.startswith("#")
        ]
        manifest_articles.append(
            {
                "part": part,
                "title": title,
                "slug": slug,
                "post_type": "page",
                "source": source.relative_to(ROOT).as_posix(),
                "file": output.relative_to(out).as_posix(),
                "url": public_url(slug, hub_slug),
                "excerpt": " ".join(paragraphs[:2])[:300],
            }
        )
    manifest = {
        "site": SITE,
        "language": language,
        "hub": {
            "title": hub_title,
            "slug": hub_slug,
            "post_type": "page",
            "file": hub_output.relative_to(out).as_posix(),
            "url": f"{SITE}/{hub_slug}/",
        },
        "articles": manifest_articles,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(out / "manifest.json")


def main() -> None:
    build_series(
        source_dir=SOURCE_DIR,
        out=OUT,
        hub_slug=HUB_SLUG,
        articles=ARTICLES,
        language="nl",
        hub_title="Van genexpressie naar 40 qubits",
    )
    build_series(
        source_dir=SOURCE_DIR_EN,
        out=OUT_EN,
        hub_slug=HUB_SLUG_EN,
        articles=ARTICLES_EN,
        language="en",
        hub_title="From gene expression to 40 qubits",
    )


if __name__ == "__main__":
    main()
