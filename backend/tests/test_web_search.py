"""
Offline tests for the web-search query pipeline.

Background: a "chart Samsung's 2025 revenue by quarter" request routed correctly to
Web_Search but came back empty and fell through to the knowledge base. Three causes:
the history-rewritten query (a question aimed at the user, with **markdown**) was the
only string searched; the model's query-expansion output was near-JSON with missing
commas so strict parsing threw; and the Google News RSS fallback asked bs4 for an XML
parser that isn't installed. These tests pin the fixes without touching the network.

Run:  cd backend && python -m pytest tests/test_web_search.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.web_search import (
    _clean_query,
    _parse_rss_items,
    build_search_queries,
    parse_query_list,
)

RAW = "chart Samsung's 2025 revenue by quarter"
REWRITTEN = "What chart would you like to see of **Samsung**'s projected revenue for each quarter in 2025?"

# Verbatim output of qwen2.5:3b for the expansion prompt — note: no commas between items.
QWEN_NEAR_JSON = (
    '[\n    "What is Samsung\'s projected quarterly revenue for each quarter in 2025?"\n'
    '    "How might Samsung distribute its total 2025 revenue evenly across the four quarters?"\n'
    '    "Can you provide detailed quarterly projections of Samsung for the year 2025?"\n]'
)


# ── _clean_query ────────────────────────────────────────────────────────────

def test_clean_query_strips_markdown_emphasis():
    assert _clean_query(REWRITTEN) == (
        "What chart would you like to see of Samsung's projected revenue for each quarter in 2025?"
    )


def test_clean_query_collapses_whitespace_and_handles_none():
    assert _clean_query("  a   b\n\tc  ") == "a b c"
    assert _clean_query(None) == ""
    assert _clean_query("`code` _under_ ~strike~ **bold**") == "code under strike bold"


# ── parse_query_list ────────────────────────────────────────────────────────

def test_parse_query_list_strict_json():
    assert parse_query_list('["a b", "c d"]') == ["a b", "c d"]


def test_parse_query_list_tolerates_code_fences():
    assert parse_query_list('```json\n["one query", "two query"]\n```') == ["one query", "two query"]


def test_parse_query_list_recovers_from_missing_commas():
    out = parse_query_list(QWEN_NEAR_JSON)
    assert len(out) == 3
    assert out[0] == "What is Samsung's projected quarterly revenue for each quarter in 2025?"


def test_parse_query_list_garbage_gives_empty_not_exception():
    assert parse_query_list("Sure! Here are some ideas.") == []
    assert parse_query_list("") == []
    assert parse_query_list(None) == []


# ── build_search_queries ────────────────────────────────────────────────────

def test_raw_user_words_are_searched_first():
    queries = build_search_queries(RAW, REWRITTEN, parse_query_list(QWEN_NEAR_JSON))
    assert queries[0] == RAW, "the user's literal words must be the first search"


def test_rewritten_query_is_kept_but_cleaned():
    # RAW is a "by quarter" series request, so per-period searches are inserted after
    # it; the rewritten form must still be present, cleaned of markdown.
    queries = build_search_queries(RAW, REWRITTEN, [])
    assert queries[0] == RAW
    assert _clean_query(REWRITTEN) in queries
    assert not any("**" in q for q in queries)
    # For a plain (non-series) request the list is exactly [raw, rewritten].
    assert build_search_queries("Samsung revenue", "**Samsung** revenue 2025", []) == ["Samsung revenue", "Samsung revenue 2025"]


def test_build_dedupes_case_insensitively_and_drops_empties():
    queries = build_search_queries("Apple revenue", "apple revenue", ["", "  ", "**Apple Revenue**", "iPhone sales"])
    assert queries == ["Apple revenue", "iPhone sales"]


def test_build_respects_limit():
    exp = [f"q{i}" for i in range(10)]
    assert len(build_search_queries("raw", "rewritten", exp, limit=4)) == 4
    assert build_search_queries("raw", "rewritten", exp, limit=4)[:2] == ["raw", "rewritten"]


# ── _parse_rss_items (Google News fallback) ─────────────────────────────────

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
<item>
  <title>Samsung posts record Q4 2025 revenue - Reuters</title>
  <link>https://news.google.com/rss/articles/abc</link>
  <description>&lt;a href="https://x"&gt;Samsung posts record Q4&lt;/a&gt;&amp;nbsp;&lt;font&gt;Reuters&lt;/font&gt;</description>
</item>
<item>
  <title>No description item</title>
  <link>https://news.google.com/rss/articles/def</link>
</item>
<item>
  <title>Second real item</title>
  <link>https://news.google.com/rss/articles/ghi</link>
  <description>plain text body</description>
</item>
</channel></rss>"""


def test_rss_parses_items_with_stdlib_and_strips_html_from_description():
    items = _parse_rss_items(RSS, max_results=5)
    assert [i["title"] for i in items] == ["Samsung posts record Q4 2025 revenue - Reuters", "Second real item"]
    assert items[0]["href"] == "https://news.google.com/rss/articles/abc"
    assert "<" not in items[0]["body"]
    assert "Samsung posts record Q4" in items[0]["body"]


def test_rss_respects_max_results():
    assert len(_parse_rss_items(RSS, max_results=1)) == 1


def test_rss_malformed_xml_raises_so_caller_logs_and_falls_through():
    with pytest.raises(Exception):
        _parse_rss_items(b"<rss><channel><item><title>x", max_results=3)


# ── PDF extraction, content-type dispatch, result merging ───────────────────
# Background (second round): with search fixed, "chart Samsung's 2025 revenue by
# quarter" still only knew Q4. The quarterly table lives in Samsung's earnings PDFs,
# which the fetcher skipped (HTML only); the newsroom HTML stalled past the timeout;
# duplicate URLs across the four query variants filled the context; and Ollama
# truncated the prompt to 4096 tokens from the front. These pin the fixes offline.

import asyncio
import io

from retrieval.web_search import _extract_pdf_text, _fetch_and_extract, merge_search_results


def _make_pdf(lines: list[str]) -> bytes:
    """A real PDF with extractable text, built with matplotlib (already installed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8, 4))
        for i, line in enumerate(lines):
            fig.text(0.05, 0.9 - i * 0.12, line, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def test_pdf_text_is_extracted():
    pdf = _make_pdf(["Sales 75.8 86.1 93.8", "Operating profit 6.5 12.2 20.1"])
    text = _extract_pdf_text(pdf)
    assert "75.8" in text and "93.8" in text
    assert "Operating profit" in text


def test_pdf_page_limit_is_respected():
    # The fixture page needs digits, or the cover-page filter (correctly) drops it.
    pdf = _make_pdf(["page one: revenue 2025 = 12.5"])
    assert "12.5" in _extract_pdf_text(pdf, max_pages=1)
    assert _extract_pdf_text(pdf, max_pages=0) == ""


class _FakeResp:
    def __init__(self, status, ctype, content=b"", text=""):
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        self.content = content
        self.text = text


def test_fetch_dispatches_pdf_by_content_type(monkeypatch):
    pdf = _make_pdf(["Quarterly revenue 42.0"])
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(200, "application/pdf", content=pdf))
    assert "42.0" in asyncio.run(_fetch_and_extract("https://x.example/report"))


def test_fetch_dispatches_pdf_by_url_suffix_even_with_octet_stream(monkeypatch):
    pdf = _make_pdf(["Quarterly revenue 43.0"])
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(200, "application/octet-stream", content=pdf))
    assert "43.0" in asyncio.run(_fetch_and_extract("https://x.example/2025_4Q.pdf?dl=1"))


def test_fetch_html_still_uses_structured_extractor(monkeypatch):
    html = "<html><body><h1>Title</h1><table><tr><th>Q</th><th>Rev</th></tr><tr><td>Q1</td><td>10</td></tr></table></body></html>"
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(200, "text/html; charset=utf-8", text=html))
    out = asyncio.run(_fetch_and_extract("https://x.example/page"))
    assert "| Q1 | 10 |" in out


def test_fetch_non_200_and_timeouts_yield_empty_not_exception(monkeypatch):
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(503, "text/html", text="<p>down</p>"))
    assert asyncio.run(_fetch_and_extract("https://x.example/a")) == ""

    def boom(*a, **k):
        raise TimeoutError("read timed out")
    monkeypatch.setattr("retrieval.web_search.requests.get", boom)
    assert asyncio.run(_fetch_and_extract("https://x.example/b")) == ""


def test_merge_drops_duplicate_urls_keeps_first_seen_order():
    per_query = [
        (["A text", "B text"], ["https://a.example/p", "https://b.example/p"]),
        (["A again", "C text"], ["https://a.example/p/", "https://c.example/p"]),   # trailing slash = same page
        (["B AGAIN", "A frag"], ["https://B.example/p", "https://a.example/p#sec"]),  # case / fragment = same page
    ]
    docs, srcs = merge_search_results(per_query)
    assert srcs == ["https://a.example/p", "https://b.example/p", "https://c.example/p"]
    assert docs == ["A text", "B text", "C text"]


def test_merge_caps_total_and_clips_rather_than_drops():
    per_query = [(["x" * 100, "y" * 100, "z" * 100], ["https://1", "https://2", "https://3"])]
    docs, srcs = merge_search_results(per_query, max_total_chars=250)
    assert [len(d) for d in docs] == [100, 100, 50]
    assert srcs == ["https://1", "https://2", "https://3"]
    docs, _ = merge_search_results(per_query, max_total_chars=200)
    assert [len(d) for d in docs] == [100, 100]


def test_merge_skips_empty_docs():
    docs, srcs = merge_search_results([(["", "real"], ["https://e", "https://r"])])
    assert docs == ["real"] and srcs == ["https://r"]


# ── Period-aware expansion and PDF page selection ───────────────────────────
# Background (third round): with fetching fixed, the merged context still held only
# Q3/Q4/FY — the model-guessed expansions never targeted Q1 and Q2, whose figures
# live in their own releases; and the 3Q interim PDF spent its whole per-doc budget
# on cover/contents pages.

from retrieval.web_search import period_queries


def test_period_queries_quarterly_request_yields_one_search_per_quarter():
    qs = period_queries("chart Samsung's 2025 revenue by quarter")
    # Financial metric: phrased to surface the quarter's own filing (measured: the
    # possessive "Samsung's 2025 revenue Q1 results" returns only news).
    assert qs == [f"Samsung Q{i} 2025 earnings release pdf" for i in (1, 2, 3, 4)]


def test_period_queries_non_financial_series_is_plain():
    assert period_queries("plot Tesla deliveries Q1 to Q4 2023") == [f"Tesla deliveries Q{i} 2023" for i in (1, 2, 3, 4)]


def test_period_labels_primitive():
    from retrieval.web_search import period_labels
    assert period_labels("chart Samsung's 2025 revenue by quarter") == ["Q1", "Q2", "Q3", "Q4"]
    assert period_labels("Nvidia revenue by half year 2024") == ["H1", "H2"]
    assert period_labels("what is Samsung's revenue?") == []


def test_period_queries_recognises_other_phrasings():
    assert len(period_queries("Apple quarterly revenue 2024")) == 4
    assert len(period_queries("plot Tesla deliveries Q1 to Q4 2023")) == 4
    assert len(period_queries("Nvidia revenue by half year 2024")) == 2


def test_period_queries_is_empty_for_non_series_requests():
    assert period_queries("chart Samsung's 2025 revenue") == []
    assert period_queries("what is the capital of France?") == []
    assert period_queries("make a bar chart: Jan 100, Feb 75") == []


def test_build_search_queries_puts_period_searches_before_model_expansions():
    raw = "chart Samsung's 2025 revenue by quarter"
    qs = build_search_queries(raw, "Samsung 2025 revenue per quarter", ["generic expansion one", "generic expansion two"])
    assert qs[0] == raw
    assert qs[1:5] == period_queries(raw)
    assert len(qs) >= 5, "limit stretches so every period fits"


def test_build_search_queries_limit_unchanged_for_plain_requests():
    qs = build_search_queries("raw", "rewritten", [f"e{i}" for i in range(10)])
    assert len(qs) == 4


def test_pdf_skips_low_digit_pages_and_keeps_numeric_ones():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8, 4))
        fig.text(0.05, 0.8, "Interim Report - Table of Contents and legal disclaimer text only", fontsize=12)
        pdf.savefig(fig); plt.close(fig)
        fig = plt.figure(figsize=(8, 4))
        fig.text(0.05, 0.8, "Sales 75.8 86.1 93.8 300.9 333.6", fontsize=12)
        pdf.savefig(fig); plt.close(fig)
    text = _extract_pdf_text(buf.getvalue())
    assert "93.8" in text
    assert "Table of Contents" not in text


# ── Bot-challenge handling and dual fallback ────────────────────────────────
# Background (fourth round): a burst of parallel searches earned the address a
# DuckDuckGo CAPTCHA. The code read the challenge page as "no results", fell to
# Wikipedia, and Wikipedia's irrelevant-but-non-empty answer masked the outage.

from retrieval.web_search import (
    SearchChallenge, _search_duckduckgo, _search_duckduckgo_once, search_web,
)

CHALLENGE_HTML = "<html><body><div class='anomaly-modal'>Unfortunately, bots use DuckDuckGo too. Please complete the following challenge</div></body></html>"
RESULTS_HTML = """<html><body>
<div class="result__body"><h2 class="result__title"><a href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2Fq2">A</a></h2><a class="result__snippet">Q2 revenue 74.1</a></div>
</body></html>"""


def test_challenge_page_is_detected_not_treated_as_empty(monkeypatch):
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(202, "text/html", text=CHALLENGE_HTML))
    with pytest.raises(SearchChallenge):
        _search_duckduckgo_once("anything")
    # the public wrapper swallows it into [] (caller falls through), no retry
    calls = []
    def counting(*a, **k):
        calls.append(1); return _FakeResp(202, "text/html", text=CHALLENGE_HTML)
    monkeypatch.setattr("retrieval.web_search.requests.get", counting)
    assert _search_duckduckgo("anything") == []
    assert len(calls) == 1


def test_normal_results_still_parse_and_unwrap_redirects(monkeypatch):
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(200, "text/html", text=RESULTS_HTML))
    r = _search_duckduckgo("q")
    assert r and r[0]["href"] == "https://a.example/q2" and "74.1" in r[0]["body"]


def test_fallback_runs_news_and_wikipedia_together_and_interleaves(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)  # the container sets it; keep this offline
    monkeypatch.setattr("retrieval.web_search._search_duckduckgo", lambda q, n: [])
    monkeypatch.setattr("retrieval.web_search._search_wikipedia_rest", lambda q, n: [{"title": "W1", "href": "https://w/1", "body": "wiki one"}, {"title": "W2", "href": "https://w/2", "body": "wiki two"}])
    monkeypatch.setattr("retrieval.web_search._search_google_news_rss", lambda q, n: [{"title": "N1", "href": "https://n/1", "body": "news one"}])
    monkeypatch.setattr("retrieval.web_search._fetch_and_extract", _async_empty)
    docs, srcs = asyncio.run(search_web("q", max_results=3))
    assert srcs == ["https://n/1", "https://w/1", "https://w/2"]
    assert docs == ["news one", "wiki one", "wiki two"]


async def _async_empty(url):
    return ""


def test_series_limit_keeps_raw_every_period_and_rewritten():
    raw = "chart Samsung's 2025 revenue by quarter"
    qs = build_search_queries(raw, "Samsung 2025 revenue per quarter (rewritten)", ["exp1", "exp2"])
    assert qs[0] == raw
    assert qs[1:5] == period_queries(raw)
    assert "Samsung 2025 revenue per quarter (rewritten)" in qs
    assert len(qs) == 6


# ── Period-table annotation ─────────────────────────────────────────────────
# Background (fifth round): with all four quarterly PDFs in context, the local
# model still assigned full-year totals to quarters and shuffled columns, because
# the period labels sit on a header line above the values with "% of sales"
# cells interleaved. These fixtures are the real layout-extracted lines from
# Samsung's 1Q and 4Q 2025 releases.

from retrieval.web_search import annotate_period_tables

Q1_PAGE = """KRW trillion  1Q24  % of sales  4Q24  % of sales  1Q25  % of sales  Sales and operating profit growth

Sales  71.9  100.0%  75.8  100.0%  79.1  100.0%  Sales
Cost of sales  45.9  63.8%  47.3  62.4%  51.0  64.5%  10% YoY  Operating
  profit
Gross profit  26.0  36.2%  28.5  37.6%  28.1  35.5%"""

Q4_PAGE = """  4Q24  3Q25  % of  4Q25  % of  2024  % of  2025  % of  Sales and operating profit growth  (KRW trillion)
KRW  trillion  sales  sales  sales  sales
Sales  75.8  86.1  100.0%  93.8  100.0%  300.9  100.0%  333.6  100.0%  Sales

Costof sales  47.3  52.6  61.1%  49.6  52.8%  186.6  62.0%  202.2  60.6%  24% YoY  Operating
  93.8  profit"""


def test_annotates_q1_layout_with_interleaved_percent_cells():
    out = annotate_period_tables(Q1_PAGE)
    assert "Sales: Q1 2024=71.9; Q4 2024=75.8; Q1 2025=79.1 (KRW trillion)" in out
    assert "Gross profit: Q1 2024=26.0; Q4 2024=28.5; Q1 2025=28.1" in out
    assert out.endswith(Q1_PAGE), "raw text is preserved after the annotations"


def test_annotates_q4_layout_where_some_columns_lack_percent_cells_and_years_become_fy():
    out = annotate_period_tables(Q4_PAGE)
    assert "Sales: Q4 2024=75.8; Q3 2025=86.1; Q4 2025=93.8; FY2024=300.9; FY2025=333.6 (KRW trillion)" in out
    assert "Costof sales: Q4 2024=47.3; Q3 2025=52.6; Q4 2025=49.6; FY2024=186.6; FY2025=202.2" in out


def test_rows_that_do_not_line_up_are_left_alone():
    page = "1Q25  2Q25  3Q25\nSales  1.0  2.0\nOther  9.9  8.8  7.7  6.6"
    out = annotate_period_tables(page)
    assert "Sales:" not in out and "Other:" not in out
    assert out == page


def test_no_period_header_means_no_change():
    assert annotate_period_tables("just prose with 2 numbers 3.5 and 4.5") == "just prose with 2 numbers 3.5 and 4.5"


def test_pdf_extraction_emits_annotations_for_numeric_pages():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(10, 4))
        fig.text(0.05, 0.8, "KRW trillion      1Q25      2Q25      3Q25", fontsize=11, family="monospace")
        fig.text(0.05, 0.6, "Sales             79.1      74.6      86.1", fontsize=11, family="monospace")
        pdf.savefig(fig); plt.close(fig)
    text = _extract_pdf_text(buf.getvalue())
    assert "Sales: Q1 2025=79.1; Q2 2025=74.6; Q3 2025=86.1" in text


# ── Series straight from the annotated tables ───────────────────────────────
# Background (sixth round): even with unambiguous "Sales: Q1 2025=79.1; ..." lines
# in context, the local model still invented quarterly figures. So for a period
# request the chart series is taken from the annotations deterministically and
# the same figures are appended to the answer.

from retrieval.web_search import series_from_context

# The real annotation lines the extractor produces for Samsung's four 2025 releases.
ANNOT_DOCS = [
    "[TABLE Q1 2024 | Q4 2024 | Q1 2025]\nSales: Q1 2024=71.9; Q4 2024=75.8; Q1 2025=79.1 (KRW trillion)\nOperating profit: Q1 2024=6.6; Q4 2024=6.5; Q1 2025=6.7 (KRW trillion)",
    "[TABLE Q2 2024 | Q1 2025 | Q2 2025]\nSales: Q2 2024=74.1; Q1 2025=79.1; Q2 2025=74.6 (KRW trillion)",
    "[TABLE Q3 2024 | Q2 2025 | Q3 2025]\nSales: Q3 2024=79.1; Q2 2025=74.6; Q3 2025=86.1 (KRW trillion)",
    "[TABLE Q4 2024 | Q3 2025 | Q4 2025 | FY2024 | FY2025]\nSales: Q4 2024=75.8; Q3 2025=86.1; Q4 2025=93.8; FY2024=300.9; FY2025=333.6 (KRW trillion)\nOperating profit: Q4 2024=6.5; Q3 2025=12.2; Q4 2025=20.1; FY2024=32.7; FY2025=43.6 (KRW trillion)",
]


def test_series_builds_the_four_quarters_from_the_filings():
    s = series_from_context(ANNOT_DOCS, "chart Samsung's 2025 revenue by quarter")
    assert s is not None
    assert s["points"] == [("Q1 2025", 79.1), ("Q2 2025", 74.6), ("Q3 2025", 86.1), ("Q4 2025", 93.8)]
    assert s["unit"] == "KRW trillion"
    assert "Samsung's 2025 revenue" in s["title"] and "quarter" in s["title"]
    assert "| Q3 2025 | 86.1 |" in s["table"]


def test_series_never_uses_full_year_totals_as_quarters():
    s = series_from_context(ANNOT_DOCS, "Samsung quarterly revenue 2025")
    assert all(v < 100 for _, v in s["points"])


def test_series_picks_the_requested_metric():
    s = series_from_context(ANNOT_DOCS, "plot Samsung operating profit by quarter 2025")
    assert s["points"] == [("Q1 2025", 6.7), ("Q3 2025", 12.2), ("Q4 2025", 20.1)]  # Q2 OP not in fixtures


def test_series_takes_the_agreeing_value_when_releases_repeat_a_quarter():
    docs = ANNOT_DOCS + ["Sales: Q3 2025=86.1; Q4 2025=93.8 (KRW trillion)", "Sales: Q3 2025=99.9 (KRW trillion)"]
    s = series_from_context(docs, "chart Samsung's 2025 revenue by quarter")
    assert dict(s["points"])["Q3 2025"] == 86.1


def test_series_infers_year_when_query_has_none():
    s = series_from_context(ANNOT_DOCS, "Samsung revenue by quarter")
    assert [k for k, _ in s["points"]] == ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]


def test_series_is_none_for_non_series_or_missing_data():
    assert series_from_context(ANNOT_DOCS, "what is Samsung's 2025 revenue?") is None
    assert series_from_context(["no tables here"], "chart Samsung's 2025 revenue by quarter") is None
    assert series_from_context(["Sales: Q1 2025=79.1 (KRW trillion)"], "chart Samsung's 2025 revenue by quarter") is None


# ── Degraded-search flag ────────────────────────────────────────────────────
# Background (seventh round): with DDG blocked, a run produced a confident
# "no such data exists" answer from headlines + Wikipedia and cached it for an
# hour. The flag lets the orchestrator label such answers and skip the cache.

import retrieval.web_search as ws


def test_degraded_flag_is_set_by_a_challenge_and_expires(monkeypatch):
    monkeypatch.setattr(ws, "_last_challenge_ts", [0.0])
    assert ws.search_is_degraded() is False
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(202, "text/html", text=CHALLENGE_HTML))
    assert ws._search_duckduckgo("anything") == []
    assert ws.search_is_degraded() is True
    assert ws.search_is_degraded(window_s=0) is False


def test_normal_results_do_not_set_the_flag(monkeypatch):
    monkeypatch.setattr(ws, "_last_challenge_ts", [0.0])
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: _FakeResp(200, "text/html", text=RESULTS_HTML))
    assert ws._search_duckduckgo("q")
    assert ws.search_is_degraded() is False


# ── SearXNG tier ────────────────────────────────────────────────────────────
# Background (eighth round): every scrapable engine bot-gates the container's
# address under load. A self-hosted metasearch instance (compose service
# `searxng`) becomes the first tier whenever SEARXNG_URL is set.

from retrieval.web_search import _search_searxng

class _FakeJsonResp(_FakeResp):
    def __init__(self, status, payload):
        super().__init__(status, "application/json")
        self._payload = payload
    def json(self):
        return self._payload

SEARX_PAYLOAD = {"results": [
    {"url": "https://images.samsung.com/x/2025_1Q_conference_eng.pdf", "title": "1Q25", "content": "Sales 79.1", "engines": ["google"]},
    {"url": "", "title": "no url", "content": ""},
    {"url": "https://news.samsung.com/q1", "title": "Q1 results", "content": "…", "engines": ["bing"]},
]}


def test_searxng_tier_is_skipped_when_unset(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    calls = []
    monkeypatch.setattr("retrieval.web_search.requests.get", lambda *a, **k: calls.append(1))
    assert _search_searxng("q") == [] and calls == []


def test_searxng_tier_parses_results_and_drops_empty_urls(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searxng:8080/")
    seen = {}
    def fake_get(url, params=None, **k):
        seen["url"] = url; seen["params"] = params
        return _FakeJsonResp(200, SEARX_PAYLOAD)
    monkeypatch.setattr("retrieval.web_search.requests.get", fake_get)
    r = _search_searxng("Samsung Q1 2025 results", 5)
    assert seen["url"] == "http://searxng:8080/search" and seen["params"]["format"] == "json"
    assert [x["href"] for x in r] == ["https://images.samsung.com/x/2025_1Q_conference_eng.pdf", "https://news.samsung.com/q1"]
    assert r[0]["body"] == "Sales 79.1"


def test_searxng_failure_falls_through_to_duckduckgo(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setattr("retrieval.web_search._search_searxng", lambda q, n: [])
    monkeypatch.setattr("retrieval.web_search._search_duckduckgo", lambda q, n: [{"title": "d", "href": "https://d/1", "body": "ddg body"}])
    monkeypatch.setattr("retrieval.web_search._fetch_and_extract", _async_empty)
    docs, srcs = asyncio.run(search_web("q", max_results=3))
    assert srcs == ["https://d/1"] and docs == ["ddg body"]


def test_searxng_success_means_duckduckgo_is_not_called(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setattr("retrieval.web_search._search_searxng", lambda q, n: [{"title": "s", "href": "https://s/1", "body": "searx body"}])
    called = []
    monkeypatch.setattr("retrieval.web_search._search_duckduckgo", lambda q, n: called.append(1) or [])
    monkeypatch.setattr("retrieval.web_search._fetch_and_extract", _async_empty)
    docs, srcs = asyncio.run(search_web("q", max_results=3))
    assert srcs == ["https://s/1"] and called == []


# ── Filing-first fetch pool for financial requests ──────────────────────────
# Background (ninth round): with metasearch working, the engines ranked Samsung's
# newsroom HTML (which stalls) above its earnings PDFs (fast, and the only source
# of the period tables), and the top-3 cut dropped the PDFs. For a financial
# request the fetch pool is wider, filings go first, and documents that yielded a
# table are kept ahead of the rest.

from retrieval.web_search import _is_filing_url, _FINANCE_INTENT_RE

def test_filing_urls_are_recognised():
    assert _is_filing_url("https://images.samsung.com/x/2025_1Q_conference_eng.pdf")
    assert _is_filing_url("https://www.samsung.com/global/ir/financial-information/")
    assert not _is_filing_url("https://news.samsung.com/global/samsung-announces-first-quarter-2025-results"), "a newsroom headline is not a filing"
    assert not _is_filing_url("https://www.facebook.com/x/posts/samsung-q3-earnings")
    assert not _is_filing_url("https://en.wikipedia.org/wiki/Samsung_Electronics")


def test_finance_intent_regex():
    assert _FINANCE_INTENT_RE.search("Samsung's 2025 revenue Q1 results")
    assert _FINANCE_INTENT_RE.search("chart Samsung's 2025 revenue by quarter")
    assert not _FINANCE_INTENT_RE.search("how tall is the Eiffel Tower")


def _pool_engine(*_a, **_k):
    return [
        {"title": "news", "href": "https://news.example/story", "body": "news snippet"},
        {"title": "wiki", "href": "https://en.wikipedia.org/wiki/X", "body": "wiki snippet"},
        {"title": "blog", "href": "https://blog.example/post", "body": "blog snippet"},
        {"title": "ir", "href": "https://x.example/ir/earnings-release/", "body": "ir snippet"},
        {"title": "pdf", "href": "https://x.example/docs/2025_1Q.pdf", "body": "pdf snippet"},
        {"title": "other", "href": "https://other.example/", "body": "other snippet"},
    ]

_PAGE_TEXT = {
    "https://news.example/story": "",                                   # stalls -> empty
    "https://en.wikipedia.org/wiki/X": "w" * 900,
    "https://blog.example/post": "b" * 1200,
    "https://x.example/ir/earnings-release/": "i" * 700,
    "https://x.example/docs/2025_1Q.pdf": "[TABLE Q1 2025]\nSales: Q1 2025=79.1\n" + "p" * 500,
    "https://other.example/": "o" * 300,
}

async def _fake_fetch(url):
    return _PAGE_TEXT.get(url, "")


def test_financial_query_fetches_wider_pool_and_keeps_table_docs_first(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setattr("retrieval.web_search._search_duckduckgo", _pool_engine)
    fetched = []
    async def counting_fetch(url):
        fetched.append(url); return await _fake_fetch(url)
    monkeypatch.setattr("retrieval.web_search._fetch_and_extract", counting_fetch)
    monkeypatch.setattr("retrieval.web_search._fetch_html", lambda u: "")
    docs, srcs = asyncio.run(search_web("Samsung 2025 revenue Q1 results", max_results=3))
    assert len(fetched) == 6, "financial request fetches the wider pool"
    assert fetched[:3] == ["https://x.example/ir/earnings-release/", "https://x.example/docs/2025_1Q.pdf", "https://news.example/story"], "filings/PDFs are fetched first, then engine order (stable sort)"
    assert srcs[0] == "https://x.example/docs/2025_1Q.pdf", "the document with a period table is kept first"
    assert docs[0].startswith("[TABLE Q1 2025]")
    assert len(docs) == 3


def test_non_financial_query_keeps_engine_order_and_top_n_only(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setattr("retrieval.web_search._search_duckduckgo", _pool_engine)
    fetched = []
    async def counting_fetch(url):
        fetched.append(url); return await _fake_fetch(url)
    monkeypatch.setattr("retrieval.web_search._fetch_and_extract", counting_fetch)
    docs, srcs = asyncio.run(search_web("how tall is the Eiffel Tower", max_results=3))
    assert fetched == ["https://news.example/story", "https://en.wikipedia.org/wiki/X", "https://blog.example/post"]
    assert srcs == fetched, "engine order preserved for ordinary queries"
    assert docs[0] == "news snippet", "a failed fetch falls back to its snippet"



# ── PDF discovery from filing pages, slow-host memory ───────────────────────
# Background (tenth round): the metasearch ranked the (stalling) newsroom above the
# PDFs and the possessive per-period phrasing surfaced no PDF at all, while the IR
# earnings-release page statically links every quarter's PDF.

from retrieval.web_search import discover_period_pdfs, _fetch_html, _slow_hosts, _host_is_slow

IR_HTML = """<html><body>
<a href="//images.samsung.com/x/ir/docs/2026_1Q_conference_eng.pdf">1Q26</a>
<a href="//images.samsung.com/x/ir/docs/2025_1Q_conference_eng.pdf">1Q25</a>
<a href='/x/ir/docs/2025_2Q_conference_eng.pdf?dl=1'>2Q25</a>
<a href="https://images.samsung.com/x/ir/docs/2025_3Q_Interim_Report.pdf">3Q25 interim</a>
<a href="https://images.samsung.com/x/ir/docs/2025_1Q_conference_eng.pdf">dup</a>
<a href="https://images.samsung.com/x/ir/docs/2025_governance_report.pdf">not a period doc</a>
<a href="https://images.samsung.com/x/ir/docs/2025_4Q_conference_eng.pdf">4Q25</a>
</body></html>"""


def test_discover_period_pdfs_filters_by_year_resolves_relative_and_dedupes():
    out = discover_period_pdfs(IR_HTML, "https://www.samsung.com/global/ir/financial-information/earnings-release/", year="2025", limit=10)
    assert out == [
        "https://images.samsung.com/x/ir/docs/2025_1Q_conference_eng.pdf",
        "https://www.samsung.com/x/ir/docs/2025_2Q_conference_eng.pdf?dl=1",
        "https://images.samsung.com/x/ir/docs/2025_3Q_Interim_Report.pdf",
        "https://images.samsung.com/x/ir/docs/2025_4Q_conference_eng.pdf",
    ]


def test_discover_period_pdfs_respects_limit_and_no_year():
    assert len(discover_period_pdfs(IR_HTML, "https://x/", year="", limit=2)) == 2
    assert discover_period_pdfs("", "https://x/", "2025") == []


def test_slow_host_is_remembered_and_skipped(monkeypatch):
    _slow_hosts.clear()
    import requests as rq
    calls = []
    def stalling(*a, **k):
        calls.append(1); raise rq.exceptions.ReadTimeout("stall")
    monkeypatch.setattr("retrieval.web_search.requests.get", stalling)
    assert asyncio.run(_fetch_and_extract("https://news.example/a")) == ""
    assert len(calls) == 1 and _host_is_slow("https://news.example/anything")
    assert asyncio.run(_fetch_and_extract("https://news.example/b")) == ""
    assert len(calls) == 1, "second request to the slow host is skipped without a fetch"
    assert _fetch_html("https://news.example/c") == "" and len(calls) == 1
    assert not _host_is_slow("https://other.example/")
    _slow_hosts.clear()


def test_financial_search_follows_pdf_links_from_filing_pages(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    _slow_hosts.clear()
    monkeypatch.setattr("retrieval.web_search._search_duckduckgo", lambda q, n: [
        {"title": "ir", "href": "https://x.example/ir/earnings-release/", "body": "ir snippet"},
        {"title": "news", "href": "https://news.example/story", "body": "news snippet"},
    ])
    pages = {
        "https://x.example/ir/earnings-release/": "i" * 800,
        "https://news.example/story": "n" * 800,
        "https://cdn.example/docs/2025_1Q_conference_eng.pdf": "[TABLE Q1 2025]\nSales: Q1 2025=79.1\n" + "p" * 400,
    }
    async def fake_fetch(url): return pages.get(url, "")
    monkeypatch.setattr("retrieval.web_search._fetch_and_extract", fake_fetch)
    monkeypatch.setattr("retrieval.web_search._fetch_html", lambda u: '<a href="https://cdn.example/docs/2025_1Q_conference_eng.pdf">q1</a>' if "earnings-release" in u else "")
    docs, srcs = asyncio.run(search_web("Samsung Q1 2025 earnings release pdf", max_results=3))
    assert srcs[0] == "https://cdn.example/docs/2025_1Q_conference_eng.pdf", "the discovered PDF with a table is kept first"
    assert docs[0].startswith("[TABLE Q1 2025]")
