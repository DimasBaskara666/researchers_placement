"""
scripts/fetch_csrankings_data.py
--------------------------------
Stage-1 (Pilot) data acquisition pipeline.

CSRankings CSV → researcher selection → DBLP papers → Semantic Scholar
abstracts → publications.csv + groups.csv + acquisition_log.json

Usage:
    python -m scripts.fetch_csrankings_data
    python -m scripts.fetch_csrankings_data --stage 1 --max-researchers 50
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ── project imports ───────────────────────────────────────────────────────
# Allow running from repo root: python -m scripts.fetch_csrankings_data
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AREA_VENUES: Dict[str, Set[str]] = {
    "Artificial Intelligence": {"aaai", "ijcai"},
    "Computer Vision": {"cvpr", "iccv", "eccv"},
    "Machine Learning": {"icml", "neurips", "nips"},
    "Natural Language Processing": {"acl", "emnlp", "naacl"},
    "The Web & Information Retrieval": {"www", "sigir"},
    "Computer Graphics": {"siggraph", "siggraph-asia", "tog"},
    "Human-Computer Interaction": {"chi", "ubicomp", "uist"},
    "Robotics": {"icra", "iros", "rss"},
    "Databases": {"sigmod", "vldb", "vldb-j"},
    "Data Mining": {"kdd", "wsdm"},
    "Software Engineering": {"icse", "fse"},
    "Visualization": {"vis", "tvcg"},
}

VENUE_TO_AREA: Dict[str, str] = {
    venue: area for area, venues in AREA_VENUES.items() for venue in venues
}

PILOT_TARGET_AREAS: List[str] = list(AREA_VENUES.keys())
SCALED_TARGET_AREAS: List[str] = PILOT_TARGET_AREAS
PILOT_MAX_RESEARCHERS: int = 500
PILOT_MIN_PAPERS: int = 5
PILOT_YEAR_RANGE: Tuple[int, int] = (2015, 2025)
SCALED_YEAR_RANGE: Tuple[int, int] = (2015, 2025)
DBLP_DELAY_SEC: float = 1.0
S2_DELAY_SEC: float = 1.0
S2_JITTER_SEC: float = 0.5
S2_BATCH_SIZE: int = 100
MIN_ABSTRACT_LENGTH: int = 100

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── dataclasses ───────────────────────────────────────────────────────────
@dataclass
class RawPaper:
    """Paper record fetched from DBLP."""
    title: str
    authors: str          # semicolon-separated
    year: int
    venue: str            # DBLP venue key
    doi: str = ""
    dblp_url: str = ""
    fetched_for: str = "" # researcher name this was fetched for


@dataclass
class EnrichedPaper:
    """Paper with abstract from Semantic Scholar."""
    title: str
    abstract: str
    authors: str
    year: int
    venue: str
    doi: str


@dataclass
class AcquisitionLog:
    """Tracks all metrics for the acquisition run."""
    stage: int = 1
    started_at: str = ""
    completed_at: str = ""
    target_areas: List[str] = field(default_factory=list)
    researchers_selected: int = 0
    researcher_names: List[str] = field(default_factory=list)
    dblp_papers_fetched: int = 0
    dblp_papers_after_dedup: int = 0
    dblp_errors: int = 0
    dblp_failed_researchers: List[str] = field(default_factory=list)
    s2_papers_queried: int = 0
    s2_abstracts_found: int = 0
    s2_abstracts_empty: int = 0
    s2_papers_not_found: int = 0
    s2_errors: int = 0
    s2_rate_limit_retries: int = 0
    s2_rate_limit_failures: int = 0
    s2_failed_batches: int = 0
    s2_failed_title_lookups: int = 0
    s2_title_lookups_attempted: int = 0
    s2_title_abstracts_found: int = 0
    s2_api_key_used: bool = False
    s2_delay_sec: float = 0.0
    s2_jitter_sec: float = 0.0
    s2_rate_limit_events: List[str] = field(default_factory=list)
    papers_after_quality_filter: int = 0
    final_publications: int = 0
    final_researchers: int = 0
    final_groups: int = 0
    group_distribution: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class HttpJsonResult:
    """HTTP JSON response plus retry/failure metadata."""
    success: bool = False
    data: Optional[Any] = None
    status_code: Optional[int] = None
    attempts: int = 0
    rate_limited: bool = False
    rate_limit_events: int = 0
    error: str = ""


class RateLimiter:
    """Simple process-local rate limiter for sequential API requests."""

    def __init__(
        self,
        min_interval_sec: float,
        jitter_sec: float = 0.0,
        clock=time.monotonic,
        sleeper=time.sleep,
        jitter_func=random.uniform,
    ):
        self.min_interval_sec = min_interval_sec
        self.jitter_sec = jitter_sec
        self._clock = clock
        self._sleeper = sleeper
        self._jitter_func = jitter_func
        self._last_request_at: Optional[float] = None

    def wait(self) -> float:
        now = self._clock()
        slept = 0.0
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            remaining = self.min_interval_sec - elapsed
            if remaining > 0:
                self._sleeper(remaining)
                slept = remaining
                now = self._clock()
        if self.jitter_sec > 0:
            jitter = max(0.0, self._jitter_func(0.0, self.jitter_sec))
            if jitter > 0:
                self._sleeper(jitter)
                slept += jitter
                now = self._clock()
        self._last_request_at = now
        return slept


# ── HTTP helpers ──────────────────────────────────────────────────────────
import ssl

def _get_ssl_context() -> ssl.SSLContext:
    """Create SSL context, falling back to unverified on cert issues (Windows)."""
    # Try certifi first (if installed)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    # Try default context with a quick connection test
    ctx = ssl.create_default_context()
    try:
        import urllib.request
        req = urllib.request.Request("https://dblp.org", method="HEAD")
        urllib.request.urlopen(req, timeout=5, context=ctx)
        return ctx
    except Exception:
        pass
    # Fallback: unverified context (safe for public academic APIs)
    log.warning("SSL certificate verification failed. Using unverified context.")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

_SSL_CTX = _get_ssl_context()
_DBLP_RATE_LIMITER = RateLimiter(min_interval_sec=DBLP_DELAY_SEC)
_S2_RATE_LIMITER = RateLimiter(min_interval_sec=S2_DELAY_SEC, jitter_sec=S2_JITTER_SEC)


def _configure_s2_rate_limiter(delay_sec: float, jitter_sec: float) -> None:
    """Update the process-local Semantic Scholar limiter for a run."""
    _S2_RATE_LIMITER.min_interval_sec = max(0.0, delay_sec)
    _S2_RATE_LIMITER.jitter_sec = max(0.0, jitter_sec)


def _retry_after_seconds(headers: Any) -> Optional[float]:
    """Parse Retry-After seconds when an API provides it."""
    if not headers:
        return None
    value = None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _sleep_with_optional_jitter(seconds: float, jitter_sec: float = 0.0) -> float:
    """Sleep for backoff/retry waits; returns the actual requested sleep time."""
    total = max(0.0, seconds)
    if jitter_sec > 0:
        total += random.uniform(0.0, jitter_sec)
    time.sleep(total)
    return total


def _execute_json_request(
    request_factory,
    url: str,
    *,
    delay: float,
    retries: int,
    timeout: float,
    rate_limiter: Optional[RateLimiter],
    retry_jitter_sec: float,
    rate_limit_cap: float,
    rate_limit_factor: float,
    server_error_cap: float,
    server_error_factor: float,
    exception_cap: float,
    exception_factor: float,
    s2_request: bool,
) -> HttpJsonResult:
    result = HttpJsonResult()
    for attempt in range(retries):
        result.attempts = attempt + 1
        try:
            if rate_limiter:
                rate_limiter.wait()
            else:
                time.sleep(delay)
            with urllib.request.urlopen(
                request_factory(), timeout=timeout, context=_SSL_CTX
            ) as resp:
                result.success = True
                result.status_code = resp.status
                result.data = json.loads(resp.read().decode("utf-8"))
                return result
        except urllib.error.HTTPError as e:
            result.status_code = e.code
            if e.code == 429:
                result.rate_limited = True
                result.rate_limit_events += 1
                retry_after = _retry_after_seconds(e.headers)
                wait = (
                    retry_after
                    if retry_after is not None
                    else min(rate_limit_cap, (2 ** attempt) * rate_limit_factor)
                )
                if s2_request:
                    log.warning(f"S2 rate limited (429). Waiting {wait}s...")
                else:
                    log.warning(
                        f"Rate limited (429). Waiting {wait}s... [{url[:80]}]"
                    )
                _sleep_with_optional_jitter(wait, retry_jitter_sec)
            elif e.code in (500, 502, 503):
                wait = min(server_error_cap, (2 ** attempt) * server_error_factor)
                prefix = "S2 " if s2_request else ""
                log.warning(
                    f"{prefix}server error ({e.code}). Retrying in {wait}s..."
                )
                _sleep_with_optional_jitter(wait, retry_jitter_sec)
            else:
                prefix = "S2 " if s2_request else ""
                result.error = f"{prefix}HTTP {e.code}"
                if s2_request:
                    log.error(f"S2 HTTP {e.code}")
                else:
                    log.error(f"HTTP {e.code} for {url[:100]}")
                return result
        except Exception as e:
            result.error = str(e)
            if s2_request:
                log.error(f"S2 request failed: {e}")
            else:
                log.error(f"Request failed: {e} [{url[:80]}]")
            if attempt < retries - 1:
                wait = min(exception_cap, (2 ** attempt) * exception_factor)
                _sleep_with_optional_jitter(wait, retry_jitter_sec)
    if not result.error:
        result.error = "Retries exhausted."
    return result


def _http_get_json(
    url: str,
    delay: float = 1.0,
    retries: int = 5,
    api_key: Optional[str] = None,
    rate_limiter: Optional[RateLimiter] = None,
    retry_jitter_sec: float = 0.0,
) -> HttpJsonResult:
    """GET request with retry metadata."""
    headers = {"User-Agent": "CollabIdentifier/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    return _execute_json_request(
        lambda: urllib.request.Request(url, headers=headers),
        url,
        delay=delay,
        retries=retries,
        timeout=30,
        rate_limiter=rate_limiter,
        retry_jitter_sec=retry_jitter_sec,
        rate_limit_cap=120,
        rate_limit_factor=10,
        server_error_cap=90,
        server_error_factor=5,
        exception_cap=60,
        exception_factor=3,
        s2_request=False,
    )


def _http_post_json(
    url: str,
    data: dict,
    params: Optional[dict] = None,
    delay: float = 1.0,
    retries: int = 3,
    api_key: Optional[str] = None,
    rate_limiter: Optional[RateLimiter] = None,
    retry_jitter_sec: float = S2_JITTER_SEC,
) -> HttpJsonResult:
    """POST JSON request with retry metadata."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    body = json.dumps(data).encode("utf-8")
    headers = {
        "User-Agent": "CollabIdentifier/1.0",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["x-api-key"] = api_key
    return _execute_json_request(
        lambda: urllib.request.Request(
            url, data=body, headers=headers, method="POST"
        ),
        url,
        delay=delay,
        retries=retries,
        timeout=60,
        rate_limiter=rate_limiter,
        retry_jitter_sec=retry_jitter_sec,
        rate_limit_cap=180,
        rate_limit_factor=15,
        server_error_cap=120,
        server_error_factor=10,
        exception_cap=90,
        exception_factor=5,
        s2_request=True,
    )


# ── Step 1: Load CSRankings data ──────────────────────────────────────────
def load_generated_author_info(filepath: Path) -> List[dict]:
    """Load generated-author-info.csv from CSRankings repo."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            area = (row.get("area") or "").strip()
            count_str = (row.get("count") or "0").strip()
            adj_str = (row.get("adjustedcount") or "0").strip()
            year_str = (row.get("year") or "0").strip()
            if not name or not area:
                continue
            try:
                rows.append({
                    "name": name,
                    "dept": (row.get("dept") or "").strip(),
                    "venue": area,
                    "count": float(count_str),
                    "adjustedcount": float(adj_str),
                    "year": int(year_str),
                })
            except ValueError:
                continue
    log.info(f"Loaded {len(rows)} rows from {filepath.name}")
    return rows


# ── Step 2: Select researchers ────────────────────────────────────────────
def select_researchers(
    author_rows: List[dict],
    target_areas: List[str],
    max_researchers: int,
    min_papers: int = 3,
    year_range: Tuple[int, int] = SCALED_YEAR_RANGE,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Select top researchers by adjusted publication count in target areas.

    Returns list of (name, total_score, {area: score}).
    """
    # Get venue codes for target areas
    target_venues: Set[str] = set()
    for area in target_areas:
        target_venues |= AREA_VENUES.get(area, set())

    # Compute per-researcher area scores
    researcher_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    min_year, max_year = year_range
    for row in author_rows:
        if row["year"] < min_year or row["year"] > max_year:
            continue
        if row["venue"] not in target_venues:
            continue
        area = VENUE_TO_AREA.get(row["venue"])
        if area and area in target_areas:
            researcher_scores[row["name"]][area] += row["adjustedcount"]

    # Filter by minimum papers and sort
    candidates = []
    for name, area_scores in researcher_scores.items():
        total = sum(area_scores.values())
        if total >= min_papers:
            candidates.append((name, total, dict(area_scores)))

    candidates.sort(key=lambda x: -x[1])

    if max_researchers > 0:
        candidates = candidates[:max_researchers]

    log.info(
        f"Selected {len(candidates)} researchers from {len(researcher_scores)} "
        f"candidates (min_papers={min_papers})"
    )
    return candidates


# ── Step 3: Fetch DBLP papers ─────────────────────────────────────────────
def _clean_dblp_name(name: str) -> str:
    """Remove DBLP disambiguation suffixes like '0001'."""
    return re.sub(r"\s+\d{4}$", "", name).strip()


def _normalize_title(title: str) -> str:
    """Normalize title for deduplication."""
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


def fetch_dblp_papers(
    researcher_name: str,
    target_venues: Set[str],
    year_range: Tuple[int, int] = SCALED_YEAR_RANGE,
    delay: float = DBLP_DELAY_SEC,
) -> Tuple[List[RawPaper], Optional[str]]:
    """Fetch papers for one researcher from DBLP API."""
    clean_name = _clean_dblp_name(researcher_name)
    # DBLP author search: use underscores for spaces in author query
    query_name = clean_name.replace(" ", "_")
    url = (
        f"https://dblp.org/search/publ/api"
        f"?q=author%3A{urllib.parse.quote(query_name)}%3A"
        f"&format=json&h=1000"
    )

    response = _http_get_json(url, delay=delay, rate_limiter=_DBLP_RATE_LIMITER)
    if not response.success:
        reason = response.error or f"HTTP {response.status_code}" if response.status_code else "unknown error"
        return [], reason
    data = response.data

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    papers: List[RawPaper] = []

    for hit in hits:
        info = hit.get("info", {})
        title = info.get("title", "").strip().rstrip(".")
        year_raw = info.get("year", "0")
        venue_raw = info.get("venue", "")
        doi_raw = info.get("doi", "")

        # DBLP sometimes returns these as lists
        if isinstance(year_raw, list):
            year_raw = year_raw[0] if year_raw else "0"
        if isinstance(doi_raw, list):
            doi_raw = doi_raw[0] if doi_raw else ""
        doi = str(doi_raw).strip() if doi_raw else ""

        if not title:
            continue
        try:
            year = int(str(year_raw))
        except (ValueError, TypeError):
            continue
        min_year, max_year = year_range
        if year < min_year or year > max_year:
            continue

        # Normalize venue to match CSRankings codes
        venue_key = _match_venue(venue_raw, target_venues)
        if not venue_key:
            continue

        # Extract authors
        authors_data = info.get("authors", {}).get("author", [])
        if isinstance(authors_data, dict):
            authors_data = [authors_data]
        author_names = []
        for a in authors_data:
            if isinstance(a, dict):
                author_names.append(a.get("text", ""))
            elif isinstance(a, str):
                author_names.append(a)
        authors_str = "; ".join(n for n in author_names if n)

        papers.append(RawPaper(
            title=title,
            authors=authors_str,
            year=year,
            venue=venue_key,
            doi=doi,
            dblp_url=info.get("url", ""),
            fetched_for=researcher_name,
        ))

    return papers, None


def _match_venue(venue_raw: str, target_venues: Set[str]) -> Optional[str]:
    """Match DBLP venue string to CSRankings venue code."""
    if not venue_raw:
        return None
    # venue_raw can be a string like "ICML" or "NeurIPS" or a list
    if isinstance(venue_raw, list):
        venue_raw = " ".join(str(v) for v in venue_raw)

    venue_lower = venue_raw.lower().strip()

    # Direct mapping of common DBLP venue names to codes
    venue_aliases = {
        "aaai": "aaai",
        "ijcai": "ijcai",
        "cvpr": "cvpr",
        "eccv": "eccv",
        "iccv": "iccv",
        "icml": "icml",
        "neurips": "nips",
        "nips": "nips",
        "iclr": "iclr",
        "kdd": "kdd",
        "acl": "acl",
        "emnlp": "emnlp",
        "naacl": "naacl",
        "sigir": "sigir",
        "www": "www",
        "isca": "isca",
        "micro": "micro",
        "asplos": "asplos",
        "hpca": "hpca",
        "sigcomm": "sigcomm",
        "nsdi": "nsdi",
        "ccs": "ccs",
        "usenix security": "usenixsec",
        "ieee s&p": "oakland",
        "ieee symposium on security and privacy": "oakland",
        "ndss": "ndss",
        "pets": "pets",
        "sigmod": "sigmod",
        "vldb": "vldb",
        "icde": "icde",
        "dac": "dac",
        "iccad": "iccad",
        "mobicom": "mobicom",
        "mobisys": "mobisys",
        "sensys": "sensys",
        "sosp": "sosp",
        "osdi": "osdi",
        "eurosys": "eurosys",
        "icse": "icse",
        "fse": "fse",
        "ase": "ase",
        "issta": "issta",
        "stoc": "stoc",
        "focs": "focs",
        "soda": "soda",
        "crypto": "crypto",
        "eurocrypt": "eurocrypt",
        "lics": "lics",
        "cav": "cav",
        "siggraph": "siggraph",
        "chi": "chiconf",
        "uist": "uist",
        "ubicomp": "ubicomp",
        "icra": "icra",
        "iros": "iros",
        "rss": "rss",
        "sc": "sc",
        "hpdc": "hpdc",
    }

    # Try exact match first
    for alias, code in venue_aliases.items():
        if alias in venue_lower:
            if code in target_venues:
                return code

    return None


def fetch_all_dblp_papers(
    researchers: List[Tuple[str, float, Dict[str, float]]],
    target_areas: List[str],
    checkpoint_dir: Path,
    year_range: Tuple[int, int] = SCALED_YEAR_RANGE,
    delay: float = DBLP_DELAY_SEC,
) -> Tuple[List[RawPaper], Dict[str, Any]]:
    """Fetch papers for all researchers with checkpointing."""
    stats: Dict[str, Any] = {"errors": 0, "failed_researchers": []}
    target_venues: Set[str] = set()
    for area in target_areas:
        target_venues |= AREA_VENUES.get(area, set())

    all_papers: List[RawPaper] = []
    progress_file = checkpoint_dir / "dblp_progress.json"
    papers_file = checkpoint_dir / "dblp_papers.json"

    # Load checkpoint
    completed_names: Set[str] = set()
    failed_researchers: Dict[str, str] = {}
    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
            completed_names = set(progress.get("completed", []))
            failed_raw = progress.get("failed", {})
            if isinstance(failed_raw, dict):
                failed_researchers = {str(k): str(v) for k, v in failed_raw.items()}
            elif isinstance(failed_raw, list):
                failed_researchers = {str(name): "legacy failed checkpoint" for name in failed_raw}
        log.info(f"Resuming DBLP fetch: {len(completed_names)} already done")
        if failed_researchers:
            log.info(f"DBLP retry candidates from checkpoint: {len(failed_researchers)}")

    if papers_file.exists():
        with open(papers_file, "r") as f:
            saved = json.load(f)
            all_papers = [RawPaper(**p) for p in saved]
        log.info(f"Loaded {len(all_papers)} cached DBLP papers")

    remaining = [(n, s, a) for n, s, a in researchers if n not in completed_names]
    log.info(f"DBLP: {len(remaining)} researchers remaining")

    for i, (name, score, areas) in enumerate(remaining):
        log.info(f"  [{i+1}/{len(remaining)}] Fetching DBLP: {name} (score={score:.1f})")
        papers, error = fetch_dblp_papers(name, target_venues, year_range=year_range, delay=delay)
        if error:
            stats["errors"] += 1
            stats["failed_researchers"].append(f"{name}: {error}")
            failed_researchers[name] = error
            log.warning(f"    DBLP fetch failed; will retry on resume: {error}")
            _save_checkpoint(checkpoint_dir, all_papers, completed_names, failed_researchers)
            continue
        all_papers.extend(papers)
        completed_names.add(name)
        failed_researchers.pop(name, None)
        log.info(f"    → {len(papers)} papers found")

        # Checkpoint every 10 researchers
        if (i + 1) % 10 == 0 or i == len(remaining) - 1:
            _save_checkpoint(checkpoint_dir, all_papers, completed_names, failed_researchers)

    _save_checkpoint(checkpoint_dir, all_papers, completed_names, failed_researchers)
    return all_papers, stats


def _save_checkpoint(
    checkpoint_dir: Path,
    papers: List[RawPaper],
    completed: Set[str],
    failed: Optional[Dict[str, str]] = None,
):
    """Save DBLP progress checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / "dblp_papers.json", "w") as f:
        json.dump([p.__dict__ for p in papers], f)
    with open(checkpoint_dir / "dblp_progress.json", "w") as f:
        json.dump({"completed": sorted(completed), "failed": failed or {}, "count": len(papers)}, f)


# ── Step 4: Deduplicate ──────────────────────────────────────────────────
def deduplicate_papers(papers: List[RawPaper]) -> List[RawPaper]:
    """Deduplicate by DOI or (normalized_title, year)."""
    seen_dois: Set[str] = set()
    seen_titles: Set[str] = set()
    unique: List[RawPaper] = []

    for p in papers:
        # Prefer DOI-based dedup
        if p.doi:
            doi_key = p.doi.lower().strip()
            if doi_key in seen_dois:
                continue
            seen_dois.add(doi_key)
        else:
            title_key = _normalize_title(p.title) + f"_{p.year}"
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

        unique.append(p)

    log.info(f"Deduplication: {len(papers)} → {len(unique)} papers")
    return unique


# ── Step 5: Fetch Semantic Scholar abstracts ──────────────────────────────
def fetch_s2_abstracts(
    papers: List[RawPaper],
    checkpoint_dir: Path,
    batch_size: int = S2_BATCH_SIZE,
    delay: float = S2_DELAY_SEC,
    jitter: float = S2_JITTER_SEC,
    api_key: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Batch fetch abstracts from Semantic Scholar.
    Returns dict: paper_key → abstract.
    """
    abstracts_file = checkpoint_dir / "s2_abstracts.json"
    abstracts: Dict[str, str] = {}
    stats: Dict[str, Any] = {
        "errors": 0,
        "rate_limit_retries": 0,
        "rate_limit_failures": 0,
        "failed_batches": 0,
        "failed_title_lookups": 0,
        "title_lookups_attempted": 0,
        "title_abstracts_found": 0,
        "api_key_used": bool(api_key),
        "delay_sec": delay,
        "jitter_sec": jitter,
        "rate_limit_events": [],
    }
    _configure_s2_rate_limiter(delay, jitter)

    if abstracts_file.exists():
        with open(abstracts_file, "r") as f:
            abstracts = json.load(f)
        log.info(f"Loaded {len(abstracts)} cached S2 abstracts")

    # Build lookup keys (DOI preferred, else title hash)
    paper_keys: List[Tuple[int, str, str]] = []  # (index, s2_id, local_key)
    for i, p in enumerate(papers):
        local_key = _paper_key(p)
        if local_key in abstracts:
            continue  # cached successful S2 response, including no-abstract responses
        if p.doi:
            s2_id = f"DOI:{p.doi}"
        else:
            s2_id = f"TITLE:{p.title}"  # fallback - will use title search
        paper_keys.append((i, s2_id, local_key))

    log.info(f"S2: {len(paper_keys)} papers need abstracts ({len(abstracts)} cached)")

    # Batch by DOI papers only (title-based require individual searches)
    doi_keys = [(i, s2_id, lk) for i, s2_id, lk in paper_keys if s2_id.startswith("DOI:")]
    title_keys = [(i, s2_id, lk) for i, s2_id, lk in paper_keys if s2_id.startswith("TITLE:")]

    # Batch DOI lookups
    for batch_start in range(0, len(doi_keys), batch_size):
        batch = doi_keys[batch_start:batch_start + batch_size]
        batch_ids = [s2_id for _, s2_id, _ in batch]

        log.info(
            f"  S2 batch {batch_start//batch_size + 1}/"
            f"{(len(doi_keys)-1)//batch_size + 1}: {len(batch)} papers"
        )

        result = _http_post_json(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            data={"ids": batch_ids},
            params={"fields": "title,abstract"},
            delay=delay,
            retries=5,
            api_key=api_key,
            rate_limiter=_S2_RATE_LIMITER,
            retry_jitter_sec=jitter,
        )

        if result.success and isinstance(result.data, list):
            if result.rate_limit_events:
                stats["rate_limit_retries"] += result.rate_limit_events
                stats["rate_limit_events"].append(
                    f"DOI batch {batch_start//batch_size + 1} recovered after "
                    f"{result.rate_limit_events} rate-limit response(s)"
                )
            for j, item in enumerate(result.data):
                _, _, local_key = batch[j]
                if item and isinstance(item, dict):
                    abstract = (item.get("abstract") or "").strip()
                    abstracts[local_key] = abstract
                else:
                    abstracts[local_key] = ""
        else:
            stats["errors"] += 1
            stats["failed_batches"] += 1
            if result.rate_limited:
                stats["rate_limit_failures"] += 1
                stats["rate_limit_events"].append(
                    f"DOI batch {batch_start//batch_size + 1} failed after "
                    f"{result.attempts} attempt(s)"
                )
            log.warning(
                f"  S2 batch failed after {result.attempts} attempt(s); "
                f"not caching {len(batch)} failed lookup(s)"
            )

        # Checkpoint after each batch
        with open(abstracts_file, "w") as f:
            json.dump(abstracts, f)

    # Individual title lookups for papers without DOI
    # NOTE: Disabled for pilot — too slow without S2 API key (constant 429s).
    # DOI-batch lookups provide sufficient coverage.
    # Re-enable with API key for Stage 2/3.
    if title_keys:
        log.info(
            f"  S2 title search: {len(title_keys)} papers without DOI — "
            f"ENABLED"
        )

    for idx, (_, s2_id, local_key) in enumerate(title_keys, start=1):
        title = s2_id.removeprefix("TITLE:")
        stats["title_lookups_attempted"] += 1
        if idx % 50 == 1 or idx == len(title_keys):
            log.info(f"    S2 title lookup {idx}/{len(title_keys)}")

        result = _search_s2_by_title(title, delay=delay, jitter=jitter, api_key=api_key)
        if result.success:
            if result.rate_limit_events:
                stats["rate_limit_retries"] += result.rate_limit_events
                stats["rate_limit_events"].append(
                    f"Title lookup {idx}/{len(title_keys)} recovered after "
                    f"{result.rate_limit_events} rate-limit response(s)"
                )
            abstract = _extract_title_search_abstract(result.data)
            abstracts[local_key] = abstract
            if abstract:
                stats["title_abstracts_found"] += 1
        else:
            stats["errors"] += 1
            stats["failed_title_lookups"] += 1
            if result.rate_limited:
                stats["rate_limit_failures"] += 1
                stats["rate_limit_events"].append(
                    f"Title lookup {idx}/{len(title_keys)} failed after "
                    f"{result.attempts} attempt(s)"
                )

        if idx % 50 == 0 or idx == len(title_keys):
            with open(abstracts_file, "w") as f:
                json.dump(abstracts, f)

    return abstracts, stats


def _search_s2_by_title(
    title: str,
    delay: float = S2_DELAY_SEC,
    jitter: float = S2_JITTER_SEC,
    api_key: Optional[str] = None,
) -> HttpJsonResult:
    """Search Semantic Scholar by title for papers without DOI."""
    params = urllib.parse.urlencode({
        "query": title,
        "limit": 1,
        "fields": "title,abstract,year,externalIds",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    return _http_get_json(
        url,
        delay=delay,
        retries=3,
        api_key=api_key,
        rate_limiter=_S2_RATE_LIMITER,
        retry_jitter_sec=jitter,
    )


def _extract_title_search_abstract(data: Any) -> str:
    """Extract the top title-search abstract, if present."""
    if not isinstance(data, dict):
        return ""
    candidates = data.get("data") or []
    if not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    return (first.get("abstract") or "").strip()


def _paper_key(paper: RawPaper) -> str:
    """Generate a stable key for a paper."""
    if paper.doi:
        return f"doi:{paper.doi.lower().strip()}"
    norm = _normalize_title(paper.title)
    h = hashlib.md5(f"{norm}_{paper.year}".encode()).hexdigest()[:12]
    return f"title:{h}"


# ── Step 6: Merge and filter ─────────────────────────────────────────────
def merge_and_filter(
    papers: List[RawPaper],
    abstracts: Dict[str, str],
    min_abstract_len: int = MIN_ABSTRACT_LENGTH,
) -> List[EnrichedPaper]:
    """Merge DBLP papers with S2 abstracts and apply quality filters."""
    enriched: List[EnrichedPaper] = []
    stats = {"no_abstract": 0, "short_abstract": 0, "accepted": 0}

    for p in papers:
        key = _paper_key(p)
        abstract = abstracts.get(key, "").strip()

        if not abstract:
            stats["no_abstract"] += 1
            continue
        if len(abstract) < min_abstract_len:
            stats["short_abstract"] += 1
            continue

        enriched.append(EnrichedPaper(
            title=p.title,
            abstract=abstract,
            authors=p.authors,
            year=p.year,
            venue=p.venue,
            doi=p.doi,
        ))
        stats["accepted"] += 1

    log.info(
        f"Quality filter: {stats['accepted']} accepted, "
        f"{stats['no_abstract']} no abstract, "
        f"{stats['short_abstract']} short abstract"
    )
    return enriched


# ── Step 7: Generate output CSVs ─────────────────────────────────────────
def generate_publications_csv(papers: List[EnrichedPaper], output_path: Path):
    """Write publications.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Abstract", "Authors", "Year", "Venue", "DOI"])
        for p in papers:
            writer.writerow([p.title, p.abstract, p.authors, p.year, p.venue, p.doi])
    log.info(f"Wrote {len(papers)} rows to {output_path}")


def generate_groups_csv(
    researchers: List[Tuple[str, float, Dict[str, float]]],
    author_rows: List[dict],
    target_areas: List[str],
    enriched_authors: Set[str],
    output_path: Path,
    year_range: Tuple[int, int],
):
    """
    Write groups.csv: researcher_name, group_name (area-based, multi-row).
    Only include researchers who have at least one enriched paper.
    """
    # Build area membership from generated-author-info rows
    researcher_areas: Dict[str, Set[str]] = defaultdict(set)
    researcher_names = {r[0] for r in researchers}

    min_year, max_year = year_range
    for row in author_rows:
        if row["name"] not in researcher_names:
            continue
        if row["year"] < min_year or row["year"] > max_year:
            continue
        area = VENUE_TO_AREA.get(row["venue"])
        if area and area in target_areas:
            researcher_areas[row["name"]].add(area)

    # Filter to researchers with enriched papers
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    written_researcher_areas: Dict[str, Set[str]] = {}
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["researcher_name", "group_name"])
        for name in sorted(researcher_areas.keys()):
            if name not in enriched_authors:
                continue
            written_researcher_areas[name] = researcher_areas[name].copy()
            for area in sorted(researcher_areas[name]):
                writer.writerow([name, area])
                rows_written += 1

    log.info(f"Wrote {rows_written} rows to {output_path}")
    return written_researcher_areas


# ── Main pipeline ────────────────────────────────────────────────────────
def run_pilot(
    raw_data_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    target_areas: Optional[List[str]] = None,
    year_range: Tuple[int, int] = SCALED_YEAR_RANGE,
    max_researchers: int = PILOT_MAX_RESEARCHERS,
    min_papers: int = PILOT_MIN_PAPERS,
    s2_delay_sec: float = S2_DELAY_SEC,
    s2_jitter_sec: float = S2_JITTER_SEC,
    s2_batch_size: int = S2_BATCH_SIZE,
    s2_api_key: Optional[str] = None,
):
    """Execute Stage 1 (Pilot) acquisition pipeline."""
    acq_log = AcquisitionLog(
        stage=1,
        started_at=datetime.now(timezone.utc).isoformat(),
        target_areas=target_areas or PILOT_TARGET_AREAS,
        s2_delay_sec=s2_delay_sec,
        s2_jitter_sec=s2_jitter_sec,
    )
    target_areas = acq_log.target_areas
    start_time = time.time()
    min_year, max_year = year_range
    log.info(f"Using publication year window: {min_year}-{max_year}")

    # ── Step 1: Load CSRankings data ──────────────────────────────────────
    gai_path = raw_data_dir / "generated-author-info.csv"
    if not gai_path.exists():
        log.error(f"File not found: {gai_path}")
        log.info("Please download generated-author-info.csv from:")
        log.info("  https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/generated-author-info.csv")
        log.info(f"  and place it in {raw_data_dir}")
        return None

    author_rows = load_generated_author_info(gai_path)

    # ── Step 2: Select researchers ────────────────────────────────────────
    researchers = select_researchers(
        author_rows,
        target_areas=target_areas,
        max_researchers=max_researchers,
        min_papers=min_papers,
        year_range=year_range,
    )
    acq_log.researchers_selected = len(researchers)
    acq_log.researcher_names = [r[0] for r in researchers]

    if not researchers:
        log.error("No researchers selected. Check target areas and data.")
        return None

    log.info(f"Selected {len(researchers)} researchers:")
    for name, score, areas in researchers[:10]:
        area_str = ", ".join(f"{a}: {s:.1f}" for a, s in sorted(areas.items(), key=lambda x: -x[1]))
        log.info(f"  {name} (score={score:.1f}) [{area_str}]")
    if len(researchers) > 10:
        log.info(f"  ... and {len(researchers) - 10} more")

    # ── Step 3: Fetch DBLP papers ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase: DBLP Paper Fetch")
    log.info("=" * 60)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_papers, dblp_stats = fetch_all_dblp_papers(
        researchers, target_areas, checkpoint_dir, year_range=year_range, delay=DBLP_DELAY_SEC
    )
    acq_log.dblp_papers_fetched = len(raw_papers)
    acq_log.dblp_errors = int(dblp_stats.get("errors", 0))
    acq_log.dblp_failed_researchers = list(dblp_stats.get("failed_researchers", []))

    # ── Step 4: Deduplicate ───────────────────────────────────────────────
    unique_papers = deduplicate_papers(raw_papers)
    acq_log.dblp_papers_after_dedup = len(unique_papers)

    # ── Step 5: Fetch S2 abstracts ────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase: Semantic Scholar Abstract Fetch")
    log.info("=" * 60)

    abstracts, s2_stats = fetch_s2_abstracts(
        unique_papers, checkpoint_dir,
        batch_size=s2_batch_size,
        delay=s2_delay_sec,
        jitter=s2_jitter_sec,
        api_key=s2_api_key,
    )

    acq_log.s2_papers_queried = len(unique_papers)
    acq_log.s2_abstracts_found = sum(1 for v in abstracts.values() if v.strip())
    acq_log.s2_abstracts_empty = sum(1 for v in abstracts.values() if not v.strip())
    acq_log.s2_errors = int(s2_stats.get("errors", 0))
    acq_log.s2_rate_limit_retries = int(s2_stats.get("rate_limit_retries", 0))
    acq_log.s2_rate_limit_failures = int(s2_stats.get("rate_limit_failures", 0))
    acq_log.s2_failed_batches = int(s2_stats.get("failed_batches", 0))
    acq_log.s2_failed_title_lookups = int(s2_stats.get("failed_title_lookups", 0))
    acq_log.s2_title_lookups_attempted = int(s2_stats.get("title_lookups_attempted", 0))
    acq_log.s2_title_abstracts_found = int(s2_stats.get("title_abstracts_found", 0))
    acq_log.s2_api_key_used = bool(s2_stats.get("api_key_used", False))
    acq_log.s2_rate_limit_events = list(s2_stats.get("rate_limit_events", []))

    # ── Step 6: Merge and filter ──────────────────────────────────────────
    enriched = merge_and_filter(unique_papers, abstracts)
    acq_log.papers_after_quality_filter = len(enriched)

    # ── Step 7: Generate outputs ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase: Generate Output CSVs")
    log.info("=" * 60)

    pubs_path = output_dir / "publications.csv"
    groups_path = output_dir / "groups.csv"

    generate_publications_csv(enriched, pubs_path)

    # Find which researchers appear in enriched papers
    enriched_authors: Set[str] = set()
    for p in enriched:
        for author in p.authors.split(";"):
            enriched_authors.add(author.strip())

    researcher_areas = generate_groups_csv(
        researchers,
        author_rows,
        target_areas,
        enriched_authors,
        groups_path,
        year_range,
    )

    # ── Compute final stats ───────────────────────────────────────────────
    acq_log.final_publications = len(enriched)
    # Count researchers who appear in groups.csv
    groups_researchers = {
        name for name, areas in researcher_areas.items()
        if name in enriched_authors
    }
    acq_log.final_researchers = len(groups_researchers)
    # Count unique groups
    all_groups: Set[str] = set()
    for areas in researcher_areas.values():
        all_groups |= areas
    acq_log.final_groups = len(all_groups)
    # Group distribution
    group_counts: Dict[str, int] = defaultdict(int)
    for name, areas in researcher_areas.items():
        if name in enriched_authors:
            for area in areas:
                group_counts[area] += 1
    acq_log.group_distribution = dict(group_counts)

    acq_log.completed_at = datetime.now(timezone.utc).isoformat()
    acq_log.duration_seconds = round(time.time() - start_time, 1)

    # Save acquisition log
    log_path = output_dir / "acquisition_log.json"
    with open(log_path, "w") as f:
        json.dump(asdict(acq_log), f, indent=2)
    log.info(f"Acquisition log saved to {log_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PILOT DATASET ACQUISITION COMPLETE")
    log.info("=" * 60)
    log.info(f"  DBLP papers fetched:     {acq_log.dblp_papers_fetched}")
    log.info(f"  After deduplication:     {acq_log.dblp_papers_after_dedup}")
    log.info(f"  S2 abstracts found:      {acq_log.s2_abstracts_found}")
    log.info(f"  S2 abstracts empty:      {acq_log.s2_abstracts_empty}")
    log.info(f"  After quality filter:    {acq_log.papers_after_quality_filter}")
    log.info(f"  Final publications:      {acq_log.final_publications}")
    log.info(f"  Final researchers:       {acq_log.final_researchers}")
    log.info(f"  Final research groups:   {acq_log.final_groups}")
    log.info(f"  Group distribution:      {dict(group_counts)}")
    log.info(f"  Duration:                {acq_log.duration_seconds}s")

    return acq_log


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CSRankings data acquisition pipeline")
    parser.add_argument("--stage", type=int, default=1, help="Acquisition stage (1=Pilot)")
    parser.add_argument("--max-researchers", type=int, default=PILOT_MAX_RESEARCHERS)
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw CSRankings CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--target-areas", type=str, default=None, help="Comma-separated CSRankings areas")
    parser.add_argument("--min-papers", type=int, default=PILOT_MIN_PAPERS)
    parser.add_argument("--year-start", type=int, default=SCALED_YEAR_RANGE[0])
    parser.add_argument("--year-end", type=int, default=SCALED_YEAR_RANGE[1])
    parser.add_argument("--s2-api-key", type=str, default=None, help="Semantic Scholar API key")
    parser.add_argument(
        "--s2-delay-sec",
        type=float,
        default=S2_DELAY_SEC,
        help="Minimum delay between Semantic Scholar requests.",
    )
    parser.add_argument(
        "--s2-jitter-sec",
        type=float,
        default=S2_JITTER_SEC,
        help="Random jitter added to Semantic Scholar request waits.",
    )
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=S2_BATCH_SIZE,
        help="Semantic Scholar DOI batch size.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore checkpoints")
    args = parser.parse_args()
    s2_api_key = (
        args.s2_api_key
        or os.getenv("S2_API_KEY")
        or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else project_root / "data" / "raw"
    year_range = (args.year_start, args.year_end)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "pilot_2021_2025"
    )
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else project_root / "data" / "checkpoints" / f"stage{args.stage}_{args.year_start}_{args.year_end}"
    )
    target_areas = (
        [area.strip() for area in args.target_areas.split(",") if area.strip()]
        if args.target_areas
        else SCALED_TARGET_AREAS
    )

    if args.s2_delay_sec < 0:
        parser.error("--s2-delay-sec must be non-negative.")
    if args.s2_jitter_sec < 0:
        parser.error("--s2-jitter-sec must be non-negative.")
    if args.s2_batch_size < 1:
        parser.error("--s2-batch-size must be at least 1.")
    if args.no_cache and checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
        log.info("Cleared checkpoint cache")

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for required input file
    gai_path = raw_dir / "generated-author-info.csv"
    if not gai_path.exists():
        log.info("Downloading generated-author-info.csv from CSRankings GitHub...")
        url = "https://raw.githubusercontent.com/emeryberger/CSrankings/gh-pages/generated-author-info.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CollabIdentifier/1.0"})
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CTX) as resp:
                with open(str(gai_path), "wb") as f_out:
                    f_out.write(resp.read())
            log.info(f"Downloaded to {gai_path}")
        except Exception as e:
            log.error(f"Failed to download: {e}")
            log.info(f"Please manually download from:\n  {url}")
            log.info(f"  and place it at: {gai_path}")
            sys.exit(1)

    result = run_pilot(
        raw_data_dir=raw_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        target_areas=target_areas,
        year_range=year_range,
        max_researchers=args.max_researchers,
        min_papers=args.min_papers,
        s2_delay_sec=args.s2_delay_sec,
        s2_jitter_sec=args.s2_jitter_sec,
        s2_batch_size=args.s2_batch_size,
        s2_api_key=s2_api_key,
    )

    if result:
        log.info("Pipeline completed successfully.")
        sys.exit(0)
    else:
        log.error("Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
