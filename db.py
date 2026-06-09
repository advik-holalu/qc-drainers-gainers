"""Supabase data access layer for the QC Drainers/Gainers dashboard.

app.py imports from this module rather than touching the Supabase client
directly. Credentials are read from st.secrets (Streamlit Cloud) with a
fallback to .env (local development).
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, datetime
from typing import Callable, Optional, Sequence

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client


_TABLE_BY_GRANULARITY = {
    "daily": "daily_data",
    "weekly": "weekly_data",
    "monthly": "monthly_data",
}

# DB snake_case → original display name expected by app.py
_DB_TO_DISPLAY = {
    "platform": "Platform",
    "date": "Date",
    "city": "City",
    "category": "Category",
    "product_id": "Product ID",
    "product_title": "Product Title",
    "grammage": "Grammage",
    "brand": "Brand",
    "item_id": "Item ID",
    "offtake_mrp": "Offtake (MRP)",
    "offtake_sp": "Offtake (SP)",
    "est_category_share": "Est. Category Share",
    "est_category_share_sp": "Est. Category Share (SP)",
    "overall_sov": "Overall SOV",
    "ad_sov": "Ad SOV",
    "wt_osa_pct": "Wt. OSA %",
    "avg_osa_pct": "Avg. OSA %",
    "mrp": "MRP",
    "selling_price": "Selling Price",
    "ptype": "Ptype",
    "variant": "Variant",
}
_DISPLAY_COLUMN_ORDER = list(_DB_TO_DISPLAY.values())
_DISPLAY_TO_DB = {v: k for k, v in _DB_TO_DISPLAY.items()}
_BOOKKEEPING_COLS = ("id", "uploaded_at")
_PAGE_SIZE = 1000  # Supabase REST default; paginate above this
_UPSERT_CONFLICT = "date,platform,city,product_title,ptype,variant"
INSERT_BATCH_SIZE = 5000  # bytes-per-row × 5000 stays well under Supabase's payload cap
_PARALLEL_BATCHES = 5     # max simultaneous in-flight POSTs to the REST API
_UPLOAD_TIMEOUT_SEC = 120


def _table_for(granularity: str) -> str:
    try:
        return _TABLE_BY_GRANULARITY[granularity]
    except KeyError as exc:
        raise ValueError(
            f"Unknown granularity {granularity!r}; expected one of "
            f"{list(_TABLE_BY_GRANULARITY)}"
        ) from exc


def _read_credentials() -> tuple[str, str]:
    url = key = None
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_SECRET_KEY")
    except (FileNotFoundError, KeyError, AttributeError):
        pass

    if not url or not key:
        load_dotenv()
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_SECRET_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not found. Set SUPABASE_URL and "
            "SUPABASE_SECRET_KEY in either .streamlit/secrets.toml (for Streamlit "
            "Cloud) or .env (for local development)."
        )
    return url, key


@st.cache_resource(show_spinner=False)
def _get_client() -> Client:
    url, key = _read_credentials()
    return create_client(url, key)


def _rename_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in _BOOKKEEPING_COLS if c in df.columns], errors="ignore")
    df = df.rename(columns=_DB_TO_DISPLAY)
    ordered = [c for c in _DISPLAY_COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


@st.cache_data(ttl=3600, show_spinner=False)
def get_last_updated(granularity: str) -> Optional[datetime]:
    """Most recent uploaded_at timestamp for the granularity, or None if empty.

    Returns a tz-aware datetime (UTC) when the column carries timezone info,
    otherwise naive. Callers should normalise to UTC before converting.
    """
    table = _table_for(granularity)
    res = (
        _get_client()
        .table(table)
        .select("uploaded_at")
        .order("uploaded_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows or not rows[0].get("uploaded_at"):
        return None
    return pd.to_datetime(rows[0]["uploaded_at"]).to_pydatetime()


def get_row_count(granularity: str) -> int:
    """Live row count for the granularity's table. Not cached on purpose."""
    table = _table_for(granularity)
    res = (
        _get_client()
        .table(table)
        .select("id", count="exact", head=True)
        .execute()
    )
    return int(res.count or 0)


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_dates(granularity: str) -> list[date]:
    """Sorted unique dates present in the table. Empty list if no rows."""
    table = _table_for(granularity)
    rows: list[dict] = []
    start = 0
    while True:
        res = (
            _get_client()
            .table(table)
            .select("date")
            .range(start, start + _PAGE_SIZE - 1)
            .execute()
        )
        chunk = res.data or []
        rows.extend(chunk)
        if len(chunk) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    if not rows:
        return []
    series = pd.to_datetime(pd.Series([r.get("date") for r in rows]), errors="coerce")
    return sorted({d.date() for d in series.dropna()})


@st.cache_data(ttl=3600, show_spinner=False)
def get_data_for_dates(granularity: str, dates: Sequence[date]) -> pd.DataFrame:
    """Fetch all rows whose date is in the given list, returned with display column names.

    Also left-joins the sku_mapping table on item_id so the returned frame has
    an additional `erp_name` column (NULL for unmapped items). The join happens
    in pandas — sku_mapping is small enough that this is simpler and faster
    than a Postgres-side JOIN.
    """
    table = _table_for(granularity)
    if not dates:
        return pd.DataFrame(columns=_DISPLAY_COLUMN_ORDER)
    iso_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    rows: list[dict] = []
    start = 0
    while True:
        res = (
            _get_client()
            .table(table)
            .select("*")
            .in_("date", iso_dates)
            .range(start, start + _PAGE_SIZE - 1)
            .execute()
        )
        chunk = res.data or []
        rows.extend(chunk)
        if len(chunk) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    if not rows:
        return pd.DataFrame(columns=_DISPLAY_COLUMN_ORDER)
    df = _rename_and_clean(pd.DataFrame(rows)).reset_index(drop=True)

    # Left-join SKU mapping on item_id → expose erp_name. Display column name
    # for item_id (after _rename_and_clean) is "Item ID"; align the mapping
    # to that and stringify both sides so int/str mismatches don't lose rows.
    mapping = get_sku_mapping_df()
    if not mapping.empty and "item_id" in mapping.columns and "Item ID" in df.columns:
        mapping_slim = (
            mapping[["item_id", "erp_name"]]
            .rename(columns={"item_id": "Item ID"})
            .dropna(subset=["Item ID"])
        )
        mapping_slim["Item ID"] = mapping_slim["Item ID"].astype(str).str.strip()
        df["Item ID"] = df["Item ID"].astype(str).str.strip()
        df = df.merge(mapping_slim, on="Item ID", how="left")
    else:
        df["erp_name"] = pd.NA
    return df


def _clean_nans(records: list[dict]) -> list[dict]:
    """Replace pandas NaN/NaT with None so Supabase JSON serialisation succeeds."""
    cleaned: list[dict] = []
    for record in records:
        out: dict = {}
        for key, value in record.items():
            if not isinstance(value, (list, dict)) and pd.isna(value):
                out[key] = None
            else:
                out[key] = value
        cleaned.append(out)
    return cleaned


def truncate_table(granularity: str) -> None:
    """Delete every row in the granularity's table. Used only by Replace mode.

    Supabase REST requires a filter on delete; we use a date sentinel that
    can't match any real row to delete-all.
    """
    table = _table_for(granularity)
    _get_client().table(table).delete().neq("date", "1900-01-01").execute()


def get_existing_dates_for_overlap(granularity: str, dates_in_file: list[str]) -> list[str]:
    """Subset of `dates_in_file` that already exists in the table (YYYY-MM-DD).

    Stops as soon as every candidate date has been seen, so this only ever does
    one full scan in the worst case (when none of the dates are present).
    """
    if not dates_in_file:
        return []
    table = _table_for(granularity)
    wanted = set(dates_in_file)
    seen: set[str] = set()
    start = 0
    while True:
        res = (
            _get_client()
            .table(table)
            .select("date")
            .in_("date", dates_in_file)
            .range(start, start + _PAGE_SIZE - 1)
            .execute()
        )
        chunk = res.data or []
        for row in chunk:
            d = row.get("date")
            if d in wanted:
                seen.add(d)
        if seen == wanted or len(chunk) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    return sorted(seen)


async def _post_batch(
    http: httpx.AsyncClient,
    url: str,
    table_name: str,
    api_key: str,
    batch: list[dict],
    on_conflict: Optional[str],
) -> int:
    """POST one batch to the Supabase REST endpoint. Returns rows written."""
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params: dict[str, str] = {}
    if on_conflict:
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
        params["on_conflict"] = on_conflict
    else:
        headers["Prefer"] = "return=minimal"

    res = await http.post(
        f"{url}/rest/v1/{table_name}",
        headers=headers,
        json=batch,
        params=params,
    )
    res.raise_for_status()
    return len(batch)


async def _upload_batches_parallel(
    url: str,
    table_name: str,
    api_key: str,
    batches: list[list[dict]],
    on_conflict: Optional[str],
    on_progress: Optional[Callable[[int, int], None]],
) -> list:
    """Run up to _PARALLEL_BATCHES POSTs concurrently. Each result is either
    the int row-count for that batch or the Exception that batch raised."""
    total = len(batches)
    done = 0
    sem = asyncio.Semaphore(_PARALLEL_BATCHES)

    async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT_SEC) as http:
        async def run_one(batch: list[dict]):
            nonlocal done
            async with sem:
                try:
                    n = await _post_batch(http, url, table_name, api_key, batch, on_conflict)
                except Exception as exc:  # noqa: BLE001
                    n = exc
            done += 1
            if on_progress:
                try:
                    on_progress(done, total)
                except Exception:  # noqa: BLE001 — never let UI callback kill the upload
                    pass
            return n

        return await asyncio.gather(*[run_one(b) for b in batches])


def insert_data(
    granularity: str,
    df: pd.DataFrame,
    mode: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Write rows to the granularity table.

    df has display-name columns with Date already in YYYY-MM-DD string form.
    mode is 'append' (upsert on the natural key) or 'replace' (truncate + insert).
    on_progress(batches_done, total_batches) is called after each batch settles.
    Dashboard caches are cleared on success so the next dashboard view sees fresh data.
    """
    table = _table_for(granularity)
    mode = mode.lower()
    if mode not in ("append", "replace"):
        raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")

    # Convert display column names → DB snake_case and keep only known columns
    df = df.rename(columns=_DISPLAY_TO_DB)
    keep = [c for c in _DISPLAY_TO_DB.values() if c in df.columns]
    df = df[keep].copy()

    # Normalize "0", "", and whitespace in ptype/variant to None so they collapse
    # with already-NULL rows under the (date, platform, city, product_title, ptype, variant)
    # unique constraint. Without this, "0" and NULL coexist and re-uploads explode duplicates.
    for col in ("ptype", "variant"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: None if (pd.isna(v) or str(v).strip() in ("", "0")) else v
            )

    records = _clean_nans(df.to_dict(orient="records"))

    if mode == "replace":
        truncate_table(granularity)

    if not records:
        return {
            "success": True,
            "rows_inserted": 0,
            "batches_succeeded": 0,
            "batches_failed": 0,
            "errors": [],
        }

    batches = [
        records[i : i + INSERT_BATCH_SIZE]
        for i in range(0, len(records), INSERT_BATCH_SIZE)
    ]
    on_conflict = _UPSERT_CONFLICT if mode == "append" else None

    url, api_key = _read_credentials()
    results = asyncio.run(
        _upload_batches_parallel(url, table, api_key, batches, on_conflict, on_progress)
    )

    rows_inserted = sum(r for r in results if isinstance(r, int))
    errors = [f"Batch {i + 1}: {r}" for i, r in enumerate(results) if isinstance(r, Exception)]
    batches_succeeded = sum(1 for r in results if isinstance(r, int))
    batches_failed = len(errors)

    if batches_succeeded > 0:
        st.cache_data.clear()

    return {
        "success": batches_failed == 0,
        "rows_inserted": rows_inserted,
        "batches_succeeded": batches_succeeded,
        "batches_failed": batches_failed,
        "errors": errors,
    }


# ── SKU mapping table ────────────────────────────────────────────────────────
_SKU_MAPPING_TABLE = "sku_mapping"
_SKU_MAPPING_COLS = ("item_id", "platform_item_name", "erp_name", "platform")


@st.cache_data(ttl=3600, show_spinner=False)
def get_sku_mapping_count() -> int:
    """Row count of sku_mapping. Cached for 1h; cache is cleared on replace."""
    res = (
        _get_client()
        .table(_SKU_MAPPING_TABLE)
        .select("item_id", count="exact", head=True)
        .execute()
    )
    return int(res.count or 0)


@st.cache_data(ttl=3600, show_spinner=False)
def get_sku_mapping_df() -> pd.DataFrame:
    """Full sku_mapping table as a DataFrame. Cached for 1h; cleared on replace."""
    client = _get_client()
    rows: list[dict] = []
    start = 0
    while True:
        res = (
            client
            .table(_SKU_MAPPING_TABLE)
            .select("*")
            .range(start, start + _PAGE_SIZE - 1)
            .execute()
        )
        chunk = res.data or []
        rows.extend(chunk)
        if len(chunk) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    if not rows:
        return pd.DataFrame(columns=list(_SKU_MAPPING_COLS) + ["updated_at"])
    return pd.DataFrame(rows)


def replace_sku_mapping(df: pd.DataFrame) -> dict:
    """Truncate sku_mapping and insert the given rows via the async parallel
    batch path. df must have columns (item_id, platform_item_name, erp_name,
    platform) — extras ignored, missing columns raise.

    Returns {"success": bool, "rows_inserted": int} on success or
    {"success": False, "error": "..."} on failure.
    """
    missing = [c for c in _SKU_MAPPING_COLS if c not in df.columns]
    if missing:
        return {"success": False, "error": f"Missing columns: {missing}"}

    try:
        df = df[list(_SKU_MAPPING_COLS)].copy()
        records = _clean_nans(df.to_dict(orient="records"))

        client = _get_client()
        # Truncate. Sentinel filter that can't match any real item_id, since
        # PostgREST delete requires a filter expression.
        client.table(_SKU_MAPPING_TABLE).delete().neq(
            "item_id", "__never_match_sentinel__"
        ).execute()

        if not records:
            st.cache_data.clear()
            return {"success": True, "rows_inserted": 0}

        batches = [
            records[i : i + INSERT_BATCH_SIZE]
            for i in range(0, len(records), INSERT_BATCH_SIZE)
        ]
        url, api_key = _read_credentials()
        results = asyncio.run(
            _upload_batches_parallel(
                url, _SKU_MAPPING_TABLE, api_key, batches,
                on_conflict=None, on_progress=None,
            )
        )

        rows_inserted = sum(r for r in results if isinstance(r, int))
        errors = [str(r) for r in results if isinstance(r, Exception)]
        if errors:
            return {
                "success": False,
                "error": f"{len(errors)} batch(es) failed: " + "; ".join(errors[:3]),
                "rows_inserted": rows_inserted,
            }

        st.cache_data.clear()
        return {"success": True, "rows_inserted": rows_inserted}
    except Exception as exc:  # noqa: BLE001 — surface message to caller
        return {"success": False, "error": str(exc)}


# ── Focus / New SKU lists (small, single-column TEXT-PK tables) ──────────────
def _read_sku_list(table_name: str) -> list[str]:
    """Read a single-column TEXT-PK table, return sorted erp_names. The
    caching decorator is on the public wrappers (get_focus_skus / get_new_skus)
    so a single helper here doesn't accidentally share cache keys across tables.
    """
    client = _get_client()
    rows: list[dict] = []
    start = 0
    while True:
        res = (
            client
            .table(table_name)
            .select("erp_name")
            .range(start, start + _PAGE_SIZE - 1)
            .execute()
        )
        chunk = res.data or []
        rows.extend(chunk)
        if len(chunk) < _PAGE_SIZE:
            break
        start += _PAGE_SIZE
    names = [
        str(r["erp_name"]).strip()
        for r in rows
        if r.get("erp_name") and str(r["erp_name"]).strip()
    ]
    return sorted(set(names))


def _replace_sku_list(table_name: str, erp_names: list[str]) -> dict:
    """Truncate the given table then insert one row per non-empty erp_name.
    Clears cache_data on success so downstream reads pick up the new state."""
    try:
        cleaned = sorted({
            str(n).strip() for n in (erp_names or []) if n and str(n).strip()
        })
        client = _get_client()
        # Sentinel filter that can't match any real row, since PostgREST delete
        # requires a filter expression.
        client.table(table_name).delete().neq(
            "erp_name", "__never_match_sentinel__"
        ).execute()
        if cleaned:
            client.table(table_name).insert(
                [{"erp_name": n} for n in cleaned]
            ).execute()
        st.cache_data.clear()
        return {"success": True, "count": len(cleaned)}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}


@st.cache_data(ttl=3600, show_spinner=False)
def get_focus_skus() -> list[str]:
    """Sorted list of erp_names currently marked as Focus. Cached 1h; cleared on save."""
    return _read_sku_list("focus_skus")


@st.cache_data(ttl=3600, show_spinner=False)
def get_new_skus() -> list[str]:
    """Sorted list of erp_names currently marked as New. Cached 1h; cleared on save."""
    return _read_sku_list("new_skus")


def replace_focus_skus(erp_names: list[str]) -> dict:
    """Replace focus_skus table contents. Returns {'success': bool, 'count': N}
    on success or {'success': False, 'error': str} on failure."""
    return _replace_sku_list("focus_skus", erp_names)


def replace_new_skus(erp_names: list[str]) -> dict:
    """Replace new_skus table contents. Returns {'success': bool, 'count': N}
    on success or {'success': False, 'error': str} on failure."""
    return _replace_sku_list("new_skus", erp_names)
