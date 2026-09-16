import json
import logging
import asyncio
import os
import re
import xml.etree.ElementTree as ET
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


_MD_EMPHASIS_RE = re.compile(r"[*_`~]+")


def _clean_query(query: str) -> str:
    """Strip markdown emphasis the history-rewriter likes to add (**Samsung**) and
    collapse whitespace. Search engines treat the asterisks as literal characters."""
    q = _MD_EMPHASIS_RE.sub("", query or "")
    return re.sub(r"\s+", " ", q).strip()


def parse_query_list(raw: str) -> list[str]:
    """
    Turn an LLM's "JSON list of strings" reply into a list. Small local models often
    emit near-JSON (missing commas between items, code fences, trailing prose), so
    fall back to pulling out the quoted strings when strict parsing fails.
    """
    text = re.sub(r"```(?:json)?", "", raw or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return [m.strip() for m in re.findall(r'"([^"\n]{3,200})"', text) if m.strip()]


_PERIOD_PATTERNS = [
    # (regex on the query, period labels to search one by one)
    (re.compile(r"\b(by|per|each|every)\s+quarter\b|\bquarterly\b|\bq1\b.*\bq4\b|\bq[1-4]\s*(?:-|to|through)\s*q[1-4]\b", re.I),
     ["Q1", "Q2", "Q3", "Q4"]),
    (re.compile(r"\b(by|per|each|every)\s+half\b|\bhalf[- ]year(ly)?\b|\bh1\b.*\bh2\b", re.I),
     ["H1", "H2"]),
]
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_SERIES_VERBS_RE = re.compile(r"^\s*(chart|plot|graph|visuali[sz]e|draw|show me|show)\s+", re.I)


_ANNOT_PERIOD = r"(?:(?:Q[1-4]|H[12]) \d{4}|FY\d{4})"
# Values are matched lazily so a trailing "(KRW trillion)" lands in the unit group
# instead of being swallowed into the last value.
_ANNOT_LINE_RE = re.compile(
    r"^([A-Za-z][A-Za-z&/'\- ]{0,40}?):\s+"
    r"(" + _ANNOT_PERIOD + r"=[^;]+?(?:;\s*" + _ANNOT_PERIOD + r"=[^;]+?)*)"
    r"(?:\s+\(([^)]+)\))?\s*$"
)
_METRIC_ALIASES = [
    # (query words, label regex) — first match on the query wins; default is revenue.
    (("operating profit", "operating income", "op margin", "opex"), re.compile(r"^operating (profit|income)", re.I)),
    (("net income", "net profit"), re.compile(r"^net (income|profit)", re.I)),
    (("gross profit", "gross margin"), re.compile(r"^gross profit", re.I)),
    (("cost of sales", "cogs"), re.compile(r"^cost ?of sales", re.I)),
    (("revenue", "sales", "turnover"), re.compile(r"^(net |total |consolidated )?(sales|revenue)s?$", re.I)),
]


def _series_subject(raw_query: str) -> str:
    q = _clean_query(raw_query)
    subject = _SERIES_VERBS_RE.sub("", q)
    for pattern, _ in _PERIOD_PATTERNS:
        subject = pattern.sub("", subject)
    return re.sub(r"\s+", " ", subject).strip(" ,.:;-")


def series_from_context(docs: list[str], raw_query: str) -> dict | None:
    """
    Build the chart series for a period request straight from the annotated source
    tables, so the numbers on the chart are the numbers in the filings — not a
    small model's transcription of them. Returns
        {"title", "kind", "unit", "points": [(label, value), ...], "table": markdown}
    or None when the request is not a series or fewer than two periods were found.
    """
    labels_wanted = period_labels(raw_query)
    if not labels_wanted:
        return None
    q_lower = _clean_query(raw_query).lower()
    metric_re = _METRIC_ALIASES[-1][1]
    for words, rx in _METRIC_ALIASES:
        if any(w in q_lower for w in words):
            metric_re = rx
            break

    # Gather every (period -> [values]) for the metric across all docs.
    found: dict[str, list[float]] = {}
    units: dict[str, int] = {}
    metric_label = None
    for doc in docs:
        for line in doc.splitlines():
            m = _ANNOT_LINE_RE.match(line.strip())
            if not m or not metric_re.search(m.group(1).strip()):
                continue
            metric_label = metric_label or m.group(1).strip()
            if m.group(3):
                units[m.group(3).strip()] = units.get(m.group(3).strip(), 0) + 1
            for pair in m.group(2).split(";"):
                per, _, val = pair.strip().partition("=")
                try:
                    v = float(val.replace(",", "").strip("() "))
                except ValueError:
                    continue
                found.setdefault(per.strip(), []).append(-v if val.strip().startswith("(") else v)
    if not found:
        return None

    year_m = _YEAR_RE.search(q_lower)
    if year_m:
        year = year_m.group(1)
    else:
        # No year in the request: take the year for which the most requested periods
        # are present, latest year on a tie (the newest complete series is the answer).
        years = {p.split()[-1] for p in found if " " in p}
        if not years:
            return None
        year = max(years, key=lambda y: (sum(f"{lab} {y}" in found for lab in labels_wanted), y))

    points = []
    for lab in labels_wanted:
        key = f"{lab} {year}"
        vals = found.get(key)
        if vals:
            best = max(set(vals), key=vals.count)   # agreement across releases wins
            points.append((key, best))
    if len(points) < 2:
        return None

    unit = max(units, key=units.get) if units else ""
    subject = _series_subject(raw_query) or metric_label or "Series"
    title = f"{subject} by {'half' if labels_wanted[0].startswith('H') else 'quarter'}" + (f" ({unit})" if unit else "")
    table = "| Period | " + (metric_label or "Value") + (f" ({unit})" if unit else "") + " |\n|---|---|\n" + \
            "\n".join(f"| {k} | {v:g} |" for k, v in points)
    return {"title": title[:80], "kind": "bar", "unit": unit, "points": points, "table": table}


_FINANCE_METRIC_RE = re.compile(r"\b(revenue|sales|turnover|earnings|profit|income|ebitda|margin|eps|opex|capex)\b", re.I)
_POSSESSIVE_RE = re.compile(r"(\w)['\u2019]s\b")


def period_labels(raw_query: str) -> list[str]:
    """["Q1","Q2","Q3","Q4"], ["H1","H2"], or [] when the request is not a period series."""
    q = _clean_query(raw_query)
    for pattern, periods in _PERIOD_PATTERNS:
        if pattern.search(q):
            return list(periods)
    return []


def _query_subject(raw_query: str, drop_metric: bool) -> str:
    """The entity the per-period searches are about: the request minus the charting
    verb, the period phrase, the year, possessives and (for a financial metric) the
    metric word — "chart Samsung's 2025 revenue by quarter" -> "Samsung"."""
    q = _clean_query(raw_query)
    subject = _SERIES_VERBS_RE.sub("", q)
    for pattern, _ in _PERIOD_PATTERNS:
        subject = pattern.sub("", subject)
    subject = _YEAR_RE.sub("", subject)
    subject = _POSSESSIVE_RE.sub(r"\1", subject)
    if drop_metric:
        subject = _FINANCE_METRIC_RE.sub("", subject)
    return re.sub(r"\s+", " ", subject).strip(" ,.:;-")


def period_queries(raw_query: str) -> list[str]:
    """
    One search per period. Phrased the way the filings are actually found: for a
    financial metric, "<company> Q1 2025 earnings release pdf" — measured to put
    the quarter's own PDF at the top, where "<company>'s 2025 revenue Q1 results"
    returns only news. Non-financial series get "<subject> Q1 2025".
    """
    labels = period_labels(raw_query)
    if not labels:
        return []
    q = _clean_query(raw_query)
    year_m = _YEAR_RE.search(q)
    year = year_m.group(1) if year_m else ""
    finance = bool(_FINANCE_METRIC_RE.search(q))
    subject = _query_subject(raw_query, drop_metric=finance)
    if not subject:
        return []
    out = []
    for lab in labels:
        core = f"{subject} {lab} {year}".strip()
        out.append(f"{core} earnings release pdf" if finance else core)
    return [re.sub(r"\s+", " ", x) for x in out]


def build_search_queries(raw_query: str, rewritten_query: str, expansions: list[str], limit: int = 4) -> list[str]:
    """
    The set of strings actually sent to the search engines, in priority order:
    the user's literal words first, then the history-rewritten form, then the
    model's expansions. Cleaned, de-duplicated (case-insensitive), capped.

    The raw query always goes first because the rewriter can turn a request into a
    question aimed at the user ("What chart would you like to see of ..."), which
    returns nothing from any engine. The rewritten form still runs so follow-ups
    like "what about 2024?" keep the context they need.
    """
    periods = period_queries(raw_query)
    if periods:
        # A series request: the per-period searches are what actually find each
        # quarter's release, so they outrank the model's generic expansions and the
        # limit stretches to fit them (dedupe + the context cap keep the window safe).
        candidates = [raw_query, *periods, rewritten_query, *expansions]
        limit = max(limit, 2 + len(periods))  # raw + periods + rewritten
    else:
        candidates = [raw_query, rewritten_query, *expansions]
    seen, out = set(), []
    for q in candidates:
        q = _clean_query(q)
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
        if len(out) >= limit:
            break
    return out


def _extract_structured_text(soup: BeautifulSoup) -> str:
    """Extract text while preserving table structure as Markdown."""
    # Process tables first
    for table in soup.find_all("table"):
        markdown_table = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
            if cells:
                markdown_table.append("| " + " | ".join(cells) + " |")
        
        # Replace table with its markdown string
        if markdown_table:
            table_text = "\n" + "\n".join(markdown_table) + "\n"
            table.replace_with(table_text)

    # Remove noise
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


class SearchChallenge(Exception):
    """The engine answered with a bot challenge (CAPTCHA) instead of results."""


_DDG_CHALLENGE_MARKERS = ("bots use duckduckgo too", "complete the following challenge", "anomaly-modal")
_ddg_stagger_lock = __import__("threading").Lock()
_ddg_last_call = [0.0]
DDG_MIN_INTERVAL = 0.6  # seconds between DDG requests from this process


def _stagger_ddg():
    """Space DDG requests out. Several truly simultaneous searches per user request
    is exactly the burst that earns the address a CAPTCHA for the next while."""
    import time
    with _ddg_stagger_lock:
        wait = _ddg_last_call[0] + DDG_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _ddg_last_call[0] = time.monotonic()


_last_challenge_ts = [0.0]
DEGRADED_WINDOW_S = 300


def search_is_degraded(window_s: float = DEGRADED_WINDOW_S) -> bool:
    """True if the primary engine served a bot challenge within the last window.
    The orchestrator uses this to label the answer as fallback-sourced and to skip
    caching it — a headlines-and-encyclopedia answer must not be served for an hour
    as if it were a real search result."""
    import time
    return (time.monotonic() - _last_challenge_ts[0]) < window_s


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo HTML search (no API key). Returns [] on a bot challenge — the
    caller falls through to the other tiers — and logs it as what it is, not as
    "no results", so a degraded search is visible in the logs."""
    _stagger_ddg()
    try:
        return _search_duckduckgo_once(query, max_results)
    except SearchChallenge:
        import time
        _last_challenge_ts[0] = time.monotonic()
        logger.warning("DuckDuckGo is serving a bot challenge to this address (rate-limited); using fallback engines.")
        return []


def _search_duckduckgo_once(query: str, max_results: int = 5) -> list[dict]:
    results = []
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=10,
        )
        lowered = resp.text[:20000].lower()
        if resp.status_code == 202 or any(m in lowered for m in _DDG_CHALLENGE_MARKERS):
            raise SearchChallenge("duckduckgo bot challenge")
        soup = BeautifulSoup(resp.text, "html.parser")
        for result in soup.select(".result__body")[:max_results]:
            title_el = result.select_one(".result__title a")
            snippet_el = result.select_one(".result__snippet")
            if title_el and snippet_el:
                href = title_el.get("href", "")
                # DDG uses redirect links — extract real URL
                if "uddg=" in href:
                    from urllib.parse import unquote, urlparse, parse_qs
                    qs = parse_qs(urlparse(href).query)
                    href = unquote(qs.get("uddg", [""])[0])
                results.append({
                    "title": title_el.get_text(strip=True),
                    "href": href,
                    "body": snippet_el.get_text(strip=True),
                })
    except SearchChallenge:
        raise
    except Exception as e:
        logger.warning(f"DuckDuckGo HTML search failed: {e}")
    return results


def _searxng_base() -> str:
    """Base URL of a SearXNG instance, or "" when the tier is disabled. Read at call
    time so tests and a running process can change it without a restart."""
    return os.getenv("SEARXNG_URL", "").strip().rstrip("/")


def _search_searxng(query: str, max_results: int = 5) -> list[dict]:
    """
    Metasearch tier: a self-hosted SearXNG queries the major engines server-side
    and returns JSON. It is the first tier when SEARXNG_URL is set, because
    scraping one engine's HTML from a home address earns a CAPTCHA after a burst.
    """
    base = _searxng_base()
    if not base:
        return []
    try:
        resp = requests.get(
            f"{base}/search",
            params={"q": query, "format": "json", "language": "en", "categories": "general"},
            headers={**HEADERS, "Accept": "application/json"},
            timeout=(5, 20),
        )
        if resp.status_code != 200:
            logger.warning(f"SearXNG returned HTTP {resp.status_code}")
            return []
        out = []
        for r in resp.json().get("results", []):
            url = (r.get("url") or "").strip()
            if not url:
                continue
            out.append({"title": (r.get("title") or "").strip(), "href": url, "body": (r.get("content") or "")[:800]})
            if len(out) >= max_results:
                break
        return out
    except Exception as e:
        logger.warning(f"SearXNG search failed: {e}")
        return []


def _search_wikipedia_rest(query: str, max_results: int = 4) -> list[dict]:
    """Use Wikipedia's REST summary API — more reliable than opensearch."""
    results = []
    try:
        # Step 1: search for page titles via the search API
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        data = resp.json()
        search_hits = data.get("query", {}).get("search", [])

        for hit in search_hits:
            title = hit["title"]
            try:
                ext_resp = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "titles": title,
                        "prop": "extracts",
                        "exsentences": 20,
                        "explaintext": True,
                        "format": "json",
                    },
                    headers=HEADERS,
                    timeout=10,
                )
                pages = ext_resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if extract and len(extract) > 50:
                        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        results.append({
                            "title": title,
                            "href": url,
                            "body": extract[:2500],
                        })
            except Exception as inner_e:
                logger.debug(f"Wikipedia extract failed for {title}: {inner_e}")

    except Exception as e:
        logger.error(f"Wikipedia REST search failed: {e}")
    return results


def _parse_rss_items(xml_bytes: bytes, max_results: int) -> list[dict]:
    """Parse <item> entries from an RSS feed with the stdlib. (bs4's "xml" mode needs
    lxml, which is not installed — that made this fallback crash on every call.)"""
    root = ET.fromstring(xml_bytes)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc = item.findtext("description") or ""
        link = (item.findtext("link") or "").strip()
        if not title or not desc:
            continue
        body = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)
        items.append({"title": title, "href": link, "body": body[:800]})
        if len(items) >= max_results:
            break
    return items


def _search_google_news_rss(query: str, max_results: int = 4) -> list[dict]:
    """Google News RSS as a third-tier fallback — no key needed."""
    results = []
    try:
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(rss_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        results = _parse_rss_items(resp.content, max_results)
    except Exception as e:
        logger.error(f"Google News RSS fallback failed: {e}")
    return results


PDF_MAX_PAGES = 12
PDF_MIN_DIGIT_RATIO = 0.02
PER_DOC_CHARS = 6000


_PERIOD_TOKEN_RE = re.compile(
    r"(?<![\w.])(?:(?P<qa>[1-4])Q\s?(?P<ya>\d{2}|\d{4})|Q(?P<qb>[1-4])\s?(?P<yb>\d{2}|\d{4})"
    r"|FY\s?(?P<yf>\d{2}|\d{4})|H(?P<h>[12])\s?(?P<yh>\d{2}|\d{4})|(?P<yy>20\d{2}))(?![\w.%])"
)
_NUM_TOKEN_RE = re.compile(r"(?<![\w.])-?\(?\d[\d,]*\.?\d*\)?%?(?![\w.])")
_UNIT_RE = re.compile(r"\b(KRW|USD|EUR|GBP|JPY|INR|\$|€|£|₩)\s?(trillion|billion|million|thousand|tn|bn|mn|k)\b", re.I)
_ROW_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z&/'()\- ]{1,40}?)\s{2,}(?=[-(\d])")


def _norm_period(m: "re.Match") -> str:
    def yr(y):
        return y if len(y) == 4 else "20" + y
    if m.group("qa"):
        return f"Q{m.group('qa')} {yr(m.group('ya'))}"
    if m.group("qb"):
        return f"Q{m.group('qb')} {yr(m.group('yb'))}"
    if m.group("yf"):
        return f"FY{yr(m.group('yf'))}"
    if m.group("h"):
        return f"H{m.group('h')} {yr(m.group('yh'))}"
    return f"FY{m.group('yy')}"


def annotate_period_tables(text: str, max_rows: int = 12) -> str:
    """
    Make financial tables unambiguous for a small model. Layout-extracted PDF
    tables put the period labels on a header line ("1Q24  4Q24  1Q25") and the
    values on the rows below ("Sales  71.9  100.0%  75.8  100.0%  79.1  100.0%"),
    often with "% of sales" cells interleaved and sometimes missing for some
    columns. A 3B model cannot align those reliably and will hand a full-year
    total to a quarter. Here the alignment is done deterministically: for every
    header line with 2+ period tokens, each following data row whose plain
    numeric cells match the header count is emitted as
        Sales: Q1 2024=71.9; Q4 2024=75.8; Q1 2025=79.1 (KRW trillion)
    and prepended to the page text. Rows that do not line up are left alone.
    """
    lines = text.splitlines()
    out_notes, i = [], 0
    while i < len(lines):
        header = list(_PERIOD_TOKEN_RE.finditer(lines[i]))
        if len(header) < 2:
            i += 1
            continue
        periods = [_norm_period(m) for m in header]
        unit_m = _UNIT_RE.search(" ".join(lines[max(0, i - 2): i + 3]))
        unit = f" ({unit_m.group(1)} {unit_m.group(2)})".replace("( ", "(") if unit_m else ""
        rows, j = [], i + 1
        while j < len(lines) and j <= i + 40 and len(rows) < max_rows:
            line = lines[j]
            if len(list(_PERIOD_TOKEN_RE.finditer(line))) >= 2:
                break  # next table's header
            lm = _ROW_LABEL_RE.match(line)
            if lm:
                nums = [t for t in _NUM_TOKEN_RE.findall(line) if not t.endswith("%")]
                if len(nums) == len(periods):
                    label = re.sub(r"\s+", " ", lm.group(1)).strip()
                    rows.append(f"{label}: " + "; ".join(f"{p}={n}" for p, n in zip(periods, nums)) + unit)
            j += 1
        if rows:
            out_notes.append("[TABLE " + " | ".join(periods) + "]\n" + "\n".join(rows))
        i = j if rows else i + 1
    if not out_notes:
        return text
    return "\n\n".join(out_notes) + "\n\n" + text


def _extract_pdf_text(data: bytes, max_pages: int = PDF_MAX_PAGES) -> str:
    """
    Text from the first pages of a PDF. Earnings releases, reports and datasheets
    are routinely served as PDFs, and their tables are exactly the numbers a
    "chart X by quarter" request needs. Uses pypdf (already an ingestion dependency).
    Layout mode keeps table columns apart; fall back to plain mode if it fails.
    """
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages[:max_pages]:
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        # Cover, disclaimer and contents pages carry almost no digits; the tables a
        # "by quarter" request needs are dense with them. Skip the former so the
        # per-document cap is spent on the latter.
        digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
        if digit_ratio < PDF_MIN_DIGIT_RATIO:
            continue
        text = re.sub(r"[ \t]{3,}", "  ", text)
        parts.append(annotate_period_tables(text))
    return "\n".join(parts).strip()


_slow_hosts: dict[str, float] = {}
SLOW_HOST_TTL_S = 600.0


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _host_is_slow(url: str) -> bool:
    import time
    return _slow_hosts.get(_host(url), 0.0) > time.monotonic()


def _mark_slow(url: str):
    """A host that timed out is skipped for a while. Some newsrooms stall on every
    request; retrying them on each of six parallel searches costs 8 s each for nothing."""
    import time
    _slow_hosts[_host(url)] = time.monotonic() + SLOW_HOST_TTL_S
    logger.info(f"Skipping slow host for {int(SLOW_HOST_TTL_S)}s: {_host(url)}")


def _fetch_html(url: str) -> str:
    """Raw HTML of a page (for link discovery), "" on any failure."""
    if _host_is_slow(url):
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=(5, 8))
        if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", "").lower():
            return resp.text
    except requests.exceptions.Timeout:
        _mark_slow(url)
    except Exception as e:
        logger.debug(f"Failed to fetch html {url}: {e}")
    return ""


_PDF_HREF_RE = re.compile(r"""href=["']([^"']+?\.pdf(?:\?[^"']*)?)["']""", re.I)
_PERIOD_IN_NAME_RE = re.compile(r"(?i)(?:[1-4]q|q[1-4]|quarter0?[1-4]|h[12]\b|fy|annual|interim|earnings|conference|results)")


def discover_period_pdfs(html: str, base_url: str, year: str = "", limit: int = 4) -> list[str]:
    """
    Period-named PDF links on a filing page. An investor-relations index links every
    quarter's release as a PDF; following those is how an analyst finds them, and it
    does not depend on which page the search engine happened to rank first.
    """
    seen, out = set(), []
    for href in _PDF_HREF_RE.findall(html or ""):
        url = urljoin(base_url, href.strip())
        name = url.rsplit("/", 1)[-1]
        if year and not (year in name or re.search(rf"(?i)(?:[1-4]q|q[1-4]){year[2:]}\b", name)):
            continue
        if not _PERIOD_IN_NAME_RE.search(name):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= limit:
            break
    return out


async def _fetch_and_extract(url: str) -> str:
    """
    Fetch a result URL and return its text: HTML pages via the structured extractor,
    PDFs via pypdf. Fails fast — a host that stalls (some newsrooms do) must not hold
    the whole batch, so the read timeout is short and every failure yields "" so the
    caller falls back to the search snippet.
    """
    if _host_is_slow(url):
        return ""
    try:
        resp = await asyncio.to_thread(requests.get, url, headers=HEADERS, timeout=(5, 8))
        if resp.status_code != 200:
            return ""
        ctype = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
            return await asyncio.to_thread(_extract_pdf_text, resp.content)
        if "text/html" in ctype:
            soup = BeautifulSoup(resp.text, "html.parser")
            return _extract_structured_text(soup)
    except requests.exceptions.Timeout:
        _mark_slow(url)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
    return ""


def merge_search_results(per_query: list[tuple[list[str], list[str]]], max_total_chars: int = 22000) -> tuple[list[str], list[str]]:
    """
    Combine (docs, sources) from several parallel searches into one context:
    drop repeat URLs (the same press release surfaces for every query variant),
    keep first-seen order, and cap the total size so it fits the model's context
    window instead of being silently truncated by the runtime. A single overlong
    document is clipped rather than dropped so the cap never empties the context.
    """
    seen, docs, sources, used = set(), [], [], 0
    for doc_texts, srcs in per_query:
        for text, src in zip(doc_texts, srcs):
            key = (src or "").split("#")[0].rstrip("/").lower()
            if not text or key in seen:
                continue
            room = max_total_chars - used
            if room <= 0:
                return docs, sources
            if len(text) > room:
                text = text[:room]
            seen.add(key)
            docs.append(text)
            sources.append(src)
            used += len(text)
    return docs, sources


FETCH_POOL = 6
_FINANCE_INTENT_RE = re.compile(
    r"\b(revenue|sales|earnings|results|profit|income|ebitda|margin|quarter(?:ly)?|q[1-4]|h[12]|fy\s?\d{2,4}|annual report|10-k|10-q)\b",
    re.I,
)
# Investor-relations sections and filings themselves — not news headlines that merely
# contain "results" or "earnings" (those are the pages that stall or paywall).
_FILING_URL_RE = re.compile(
    r"(\.pdf(?:$|\?)|/ir/|/ir$|investor|earnings-release|financial-information|/financials?/|sec\.gov|/filings?/|/results/)",
    re.I,
)


def _is_filing_url(url: str) -> bool:
    """Investor-relations pages and PDFs: for a financial question these are the
    ground truth, and they are the ones that carry the period tables."""
    return bool(_FILING_URL_RE.search(url or ""))


async def search_web(query: str, max_results: int = 5) -> tuple[list[str], list[str]]:
    """
    Tries DuckDuckGo HTML → Wikipedia REST API → Google News RSS.
    Visits the top links to extract full structured content (tables preserved).
    Returns (doc_texts, sources).
    """
    query = _clean_query(query)
    if not query:
        return [], []

    results, source_label = [], ""
    if _searxng_base():
        results = await asyncio.to_thread(_search_searxng, query, max_results)
        source_label = "SearXNG"
    if not results:
        results = await asyncio.to_thread(_search_duckduckgo, query, max_results)
        source_label = "DuckDuckGo"

    if not results:
        # Wikipedia is encyclopedic and Google News is current; neither alone covers
        # what a web search does, so when DDG is down run both and interleave them
        # rather than stopping at the first tier that returns *something*.
        logger.warning("DuckDuckGo returned nothing; querying Wikipedia and Google News together.")
        wiki, news = await asyncio.gather(
            asyncio.to_thread(_search_wikipedia_rest, query, max_results),
            asyncio.to_thread(_search_google_news_rss, query, max_results),
        )
        results = [r for pair in zip(news, wiki) for r in pair]
        results += news[len(wiki):] + wiki[len(news):]
        source_label = "Google News + Wikipedia"

    if not results:
        logger.error("All web search sources exhausted — no results found.")
        return [], []

    finance = bool(_FINANCE_INTENT_RE.search(query))
    if finance:
        # A financial question: fetch a wider pool, filings and PDFs first (engines
        # rank a newsroom's HTML above its own PDF, and that HTML often stalls), then
        # keep the documents that yielded an annotated period table ahead of the
        # rest, longest next. Only these carry the numbers a "by quarter" chart needs.
        pool = results[: max(FETCH_POOL, max_results)]
        pool = sorted(pool, key=lambda r: 0 if _is_filing_url(r["href"]) else 1)  # stable
    else:
        pool = results[:max_results]

    extracted_texts = await asyncio.gather(*[_fetch_and_extract(r["href"]) for r in pool])

    scored = []
    for r, text in zip(pool, extracted_texts):
        if text and len(text) > 200:
            doc, rank = text[:PER_DOC_CHARS], (0 if "[TABLE " in text else 1)
        else:
            doc, rank = r["body"], 2  # fetch failed or too short: fall back to the snippet
        scored.append((rank, -len(doc), r["href"], doc))
    if finance:
        # Second route to the filings: follow period-named PDF links from the filing
        # pages we did fetch (an IR index lists every quarter's release).
        year_m = _YEAR_RE.search(query)
        year = year_m.group(1) if year_m else ""
        have = {r["href"] for r in pool}
        filing_pages = [r["href"] for r in pool if _is_filing_url(r["href"]) and not r["href"].lower().split("?")[0].endswith(".pdf")][:2]
        discovered: list[str] = []
        for page in filing_pages:
            html = await asyncio.to_thread(_fetch_html, page)
            for pdf in discover_period_pdfs(html, page, year):
                if pdf not in have and pdf not in discovered:
                    discovered.append(pdf)
        discovered = discovered[:4]
        if discovered:
            more = await asyncio.gather(*[_fetch_and_extract(u) for u in discovered])
            for u, text in zip(discovered, more):
                if text and len(text) > 200:
                    scored.append((0 if "[TABLE " in text else 1, -len(text), u, text[:PER_DOC_CHARS]))
            logger.info(f"Discovered {len(discovered)} period PDFs from filing pages for: {query}")
        scored.sort(key=lambda t: (t[0], t[1]))
    keep = scored[:max_results]
    full_docs = [d for _, _, _, d in keep]
    final_sources = [h for _, _, h, _ in keep]

    logger.info(f"Web search ({source_label}) → {len(full_docs)} results for: {query}")
    return full_docs, final_sources
