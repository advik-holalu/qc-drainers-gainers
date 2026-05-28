from datetime import date, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

import db


# ── Column renaming (display → internal snake_case) ─────────────────────────
# Display name (from db.py) → internal snake_case used downstream in this file.
# Name-based: tolerates any column order coming out of get_data_for_dates.
_DISPLAY_TO_INTERNAL = {
    "Platform": "Platform",
    "Date": "Date",
    "City": "City",
    "Category": "Category",
    "Product ID": "Product_ID",
    "Product Title": "Product_Title",
    "Grammage": "Grammage",
    "Brand": "Brand",
    "Item ID": "Item_ID",
    "Offtake (MRP)": "Offtake_MRP",
    "Offtake (SP)": "Offtake_SP",
    "Est. Category Share": "MS_MRP",
    "Est. Category Share (SP)": "MS_SP",
    "Overall SOV": "Overall_SOV",
    "Ad SOV": "Ad_SOV",
    "Wt. OSA %": "Wt_OSA",
    "Avg. OSA %": "Avg_OSA",
    "MRP": "MRP_price",
    "Selling Price": "Selling_Price",
    "Ptype": "Ptype",
    "Variant": "Variant",
}

KEY_COLS = ["City", "Ptype", "Variant", "Product_Title", "Platform", "Category"]
METRIC_COLS = ["Offtake_MRP", "MS_MRP", "Wt_OSA", "Ad_SOV", "Overall_SOV", "Selling_Price"]
FILTER_COLS = ["Platform", "Category", "Ptype", "Variant"]

# Subtle row backgrounds for the top-50% offtake contributors. Slightly lifted
# tones that read across both light and dark modes.
ROW_BG_DRAINERS = "#4a2828"
ROW_BG_GAINERS = "#28402a"

# IST = UTC+5:30 (team is in India). uploaded_at is stored UTC in Supabase.
_IST = timezone(timedelta(hours=5, minutes=30))


# ── Cleanup helpers ──────────────────────────────────────────────────────────
def _uncategorized(v):
    if pd.isna(v):
        return "Uncategorized"
    s = str(v).strip()
    if s in ("", "0"):
        return "Uncategorized"
    return str(v)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Name-based rename (display → internal) + date parse + Uncategorized normalisation."""
    df = df.copy()
    df = df.rename(columns=_DISPLAY_TO_INTERNAL)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    for col in ("Ptype", "Variant"):
        if col in df.columns:
            df[col] = df[col].apply(_uncategorized)
    return df.dropna(subset=["Date"]).reset_index(drop=True)


# ── Display formatting ───────────────────────────────────────────────────────
NA = "—"


def fp(v):
    return NA if pd.isna(v) else f"{v:.1f}%"


def fpd(v):
    if pd.isna(v):
        return NA
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


def fr(v):
    return NA if pd.isna(v) else f"₹{v:,.0f}"


def frd(v):
    if pd.isna(v):
        return NA
    if v > 0:
        return f"+₹{v:,.0f}"
    if v < 0:
        return f"-₹{abs(v):,.0f}"
    return "₹0"


def fra(v):
    return NA if pd.isna(v) else f"₹{v:,.1f}"


def frad(v):
    if pd.isna(v):
        return NA
    if v > 0:
        return f"+₹{v:,.1f}"
    if v < 0:
        return f"-₹{abs(v):,.1f}"
    return "₹0.0"


# SKU truncation strategy: leave the full Product Title in the underlying data
# (so CSV downloads include it). At render time we hand Streamlit a column_config
# with width="medium" on the SKU column — st.dataframe natively shows "..." when
# the column is narrower than the cell content, and the user can drag to widen.
# We rely on Streamlit matching the SKU column by its leaf name; with MultiIndex
# columns Streamlit flattens the lookup to the leaf in column_config.

_FMT = {
    ("MS", "Prev"): fp,          ("MS", "Current"): fp,
    ("MS", "PP Δ"): fpd,         ("MS", "% Δ"): fpd,
    ("Offtake (MRP)", "Current"): fr, ("Offtake (MRP)", "Prev"): fr,
    ("Offtake (MRP)", "Abs Δ"): frd, ("Offtake (MRP)", "% Δ"): fpd,
    ("OSA", "Prev"): fp,         ("OSA", "Current"): fp,   ("OSA", "Δ pp"): fpd,
    ("Ad SOV", "Prev"): fp,      ("Ad SOV", "Current"): fp, ("Ad SOV", "Δ pp"): fpd,
    ("Overall SOV", "Prev"): fp, ("Overall SOV", "Current"): fp, ("Overall SOV", "Δ pp"): fpd,
    ("ASP", "Prev"): fra,        ("ASP", "Current"): fra,  ("ASP", "Δ abs"): frad,
}

_DELTA_COLS = [
    ("MS", "PP Δ"), ("MS", "% Δ"),
    ("Offtake (MRP)", "Abs Δ"), ("Offtake (MRP)", "% Δ"),
    ("OSA", "Δ pp"),
    ("Ad SOV", "Δ pp"),
    ("Overall SOV", "Δ pp"),
    ("ASP", "Δ abs"),
]


def _make_display(data: pd.DataFrame) -> pd.DataFrame:
    cols = {
        ("", "City"): data["City"].values,
        ("", "Ptype"): data["Ptype"].values,
        ("", "Variant"): data["Variant"].values,
        ("", "SKU"): data["Product_Title"].values,
        ("", "Platform"): data["Platform"].values,
        ("", "Category"): data["Category"].values,
        ("MS", "Prev"): data["MS_MRP_p"].values,
        ("MS", "Current"): data["MS_MRP_c"].values,
        ("MS", "PP Δ"): data["MS_pp"].values,
        ("MS", "% Δ"): data["MS_pct"].values,
        ("Offtake (MRP)", "Current"): data["Offtake_MRP_c"].values,
        ("Offtake (MRP)", "Prev"): data["Offtake_MRP_p"].values,
        ("Offtake (MRP)", "Abs Δ"): data["Oft_abs"].values,
        ("Offtake (MRP)", "% Δ"): data["Oft_pct"].values,
        ("OSA", "Prev"): data["Wt_OSA_p"].values,
        ("OSA", "Current"): data["Wt_OSA_c"].values,
        ("OSA", "Δ pp"): data["OSA_d"].values,
        ("Ad SOV", "Prev"): data["Ad_SOV_p"].values,
        ("Ad SOV", "Current"): data["Ad_SOV_c"].values,
        ("Ad SOV", "Δ pp"): data["AdSOV_d"].values,
        ("Overall SOV", "Prev"): data["Overall_SOV_p"].values,
        ("Overall SOV", "Current"): data["Overall_SOV_c"].values,
        ("Overall SOV", "Δ pp"): data["OvSOV_d"].values,
        ("ASP", "Prev"): data["Selling_Price_p"].values,
        ("ASP", "Current"): data["Selling_Price_c"].values,
        ("ASP", "Δ abs"): data["ASP_d"].values,
    }
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def _color_deltas(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for col in _DELTA_COLS:
        if col in data.columns:
            styles[col] = data[col].apply(
                lambda v: ""
                if pd.isna(v) or v == 0
                else ("color: green; font-weight: 600" if v > 0 else "color: red; font-weight: 600")
            )
    return styles


def _row_bg_factory(mask: list[bool], color: str):
    css = f"background-color: {color}"

    def styler(row):
        return [css if mask[row.name] else "" for _ in row]

    return styler


def _style(data: pd.DataFrame, row_mask: list[bool] | None = None, row_color: str | None = None):
    display = _make_display(data)
    styler = (
        display.style
        .format(_FMT, na_rep=NA)
        .apply(_color_deltas, axis=None)
    )
    if row_mask and row_color and any(row_mask):
        styler = styler.apply(_row_bg_factory(row_mask, row_color), axis=1)
    return styler


# ── Pipeline helpers ─────────────────────────────────────────────────────────
def _top50_mask(magnitudes: pd.Series) -> list[bool]:
    """Return a positional boolean mask marking rows that cumulatively
    contribute to the first 50% of total `magnitudes` (sorted descending
    by caller). The row that crosses the 50% threshold is included.
    Rows with zero or negative magnitude are never marked — they don't
    contribute to growth/decline (callers clip them to 0 before passing).
    """
    if len(magnitudes) == 0:
        return []
    total = float(magnitudes.sum())
    if total <= 0:
        return [False] * len(magnitudes)
    cum = magnitudes.cumsum()
    prior = cum - magnitudes
    return ((magnitudes > 0) & (prior < 0.5 * total)).tolist()


def _classify_issue(ms_c, ms_p, oft_c, oft_p) -> str:
    """Label for the MZ expander. Eligibility requires non-NULL MS and Offtake
    on both sides; this classifier reports which side(s) are missing."""
    parts: list[str] = []
    ms_c_miss = pd.isna(ms_c)
    ms_p_miss = pd.isna(ms_p)
    if ms_c_miss and ms_p_miss:
        parts.append("MS missing in both")
    elif ms_c_miss:
        parts.append("MS missing in current")
    elif ms_p_miss:
        parts.append("MS missing in previous")

    oft_c_miss = pd.isna(oft_c)
    oft_p_miss = pd.isna(oft_p)
    if oft_c_miss and oft_p_miss:
        parts.append("Offtake missing in both")
    elif oft_c_miss:
        parts.append("Offtake missing in current")
    elif oft_p_miss:
        parts.append("Offtake missing in previous")

    return "; ".join(parts) if parts else ""


def _flat_csv_columns(multi_cols) -> list[str]:
    return [f"{grp} {sub}".strip() if grp else sub for grp, sub in multi_cols]


def _drainers_gainers_csv_bytes(data: pd.DataFrame) -> bytes:
    """CSV with raw, unformatted values matching the user's filtered+sorted view."""
    display = _make_display(data)
    flat = display.copy()
    flat.columns = _flat_csv_columns(flat.columns)
    return flat.to_csv(index=False).encode("utf-8")


def _build_mz_view(missing_zero: pd.DataFrame) -> pd.DataFrame:
    if missing_zero.empty:
        return pd.DataFrame(
            columns=[
                "City", "Ptype", "Variant", "SKU", "Platform", "Category",
                "Offtake (MRP) This Week", "Offtake (MRP) Previous Week", "Issue",
            ]
        )
    issues = [
        _classify_issue(msc, msp, oftc, oftp)
        for msc, msp, oftc, oftp in zip(
            missing_zero["MS_MRP_c"],
            missing_zero["MS_MRP_p"],
            missing_zero["Offtake_MRP_c"],
            missing_zero["Offtake_MRP_p"],
        )
    ]
    view = pd.DataFrame({
        "City": missing_zero["City"].values,
        "Ptype": missing_zero["Ptype"].values,
        "Variant": missing_zero["Variant"].values,
        "SKU": missing_zero["Product_Title"].values,
        "Platform": missing_zero["Platform"].values,
        "Category": missing_zero["Category"].values,
        "Offtake (MRP) This Week": missing_zero["Offtake_MRP_c"].values,
        "Offtake (MRP) Previous Week": missing_zero["Offtake_MRP_p"].values,
        "Issue": issues,
    })
    return view.sort_values(["Issue", "SKU"], kind="stable").reset_index(drop=True)


def _format_last_updated(dt) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_IST).strftime("%d %b %Y, %H:%M") + " IST"


def _format_week_label(d: date) -> str:
    """Format a weekly snapshot date as DD/MM/YYYY with leading zeros."""
    return d.strftime("%d/%m/%Y")


def _format_month_label(d: date) -> str:
    """Format a monthly snapshot date as 'April 2026'."""
    return d.strftime("%B %Y")


def _snap_to_available(picked, available: list[date]) -> date | None:
    """Return the available date closest to picked (None if list is empty)."""
    if not available:
        return None
    if picked in available:
        return picked
    return min(available, key=lambda d: abs((d - picked).days))


# ── Main render ──────────────────────────────────────────────────────────────
def render_dashboard() -> None:
    # ── Page header ──────────────────────────────────────────────────────────
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.title("QC Drainers/Gainers Dashboard")
    with refresh_col:
        st.write("")  # vertical alignment with the title
        if st.button("🔄 Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    # ── Granularity toggle ───────────────────────────────────────────────────
    granularity = st.radio(
        "Data granularity",
        options=["Daily", "Weekly", "Monthly"],
        index=1,
        horizontal=True,
    )
    gran_key = granularity.lower()

    if db.get_row_count(gran_key) == 0:
        st.info(
            f"📭 No {granularity} data uploaded yet. Switch to a different granularity "
            f"or upload {granularity} data via the Upload page."
        )
        st.stop()

    # ── Data last updated ────────────────────────────────────────────────────
    last_updated = db.get_last_updated(gran_key)
    if last_updated is not None:
        st.caption(f"Data last updated: {_format_last_updated(last_updated)}")

    # ── Period pickers (one per period, widget type depends on granularity) ──
    available_dates: list[date] = db.get_available_dates(gran_key)
    if len(available_dates) < 2:
        st.warning(
            f"Need at least 2 {granularity.lower()} periods uploaded to compare. "
            f"Currently only {len(available_dates)} available."
        )
        st.stop()

    options_desc = sorted(available_dates, reverse=True)
    this_default = options_desc[0]
    prev_default = options_desc[1]

    col1, col2 = st.columns(2)

    if granularity == "Daily":
        with col1:
            picked_this = st.date_input(
                "This Period",
                value=this_default,
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="this_period_date",
            )
        with col2:
            picked_prev = st.date_input(
                "Previous Period",
                value=prev_default,
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="prev_period_date",
            )
        this_period = _snap_to_available(picked_this, available_dates)
        prev_period = _snap_to_available(picked_prev, available_dates)
        if this_period != picked_this and this_period is not None:
            st.caption(f"Snapped This Period to nearest available date: {this_period.strftime('%d/%m/%Y')}")
        if prev_period != picked_prev and prev_period is not None:
            st.caption(f"Snapped Previous Period to nearest available date: {prev_period.strftime('%d/%m/%Y')}")

    elif granularity == "Weekly":
        # Defensive: drop any session-state value that's no longer in options
        for k, default in [("this_period_week", this_default), ("prev_period_week", prev_default)]:
            if k in st.session_state and st.session_state[k] not in options_desc:
                st.session_state[k] = default
        with col1:
            this_period = st.selectbox(
                "This Period",
                options=options_desc,
                format_func=_format_week_label,
                index=0 if "this_period_week" not in st.session_state else None,
                key="this_period_week",
            )
        with col2:
            prev_period = st.selectbox(
                "Previous Period",
                options=options_desc,
                format_func=_format_week_label,
                index=1 if "prev_period_week" not in st.session_state else None,
                key="prev_period_week",
            )

    else:  # Monthly
        for k, default in [("this_period_month", this_default), ("prev_period_month", prev_default)]:
            if k in st.session_state and st.session_state[k] not in options_desc:
                st.session_state[k] = default
        with col1:
            this_period = st.selectbox(
                "This Period",
                options=options_desc,
                format_func=_format_month_label,
                index=0 if "this_period_month" not in st.session_state else None,
                key="this_period_month",
            )
        with col2:
            prev_period = st.selectbox(
                "Previous Period",
                options=options_desc,
                format_func=_format_month_label,
                index=1 if "prev_period_month" not in st.session_state else None,
                key="prev_period_month",
            )

    # Format caption + period strings per granularity
    if granularity == "Daily":
        this_label = this_period.strftime("%d/%m/%Y")
        prev_label = prev_period.strftime("%d/%m/%Y")
        csv_suffix = f"{this_period.strftime('%Y-%m-%d')}_vs_{prev_period.strftime('%Y-%m-%d')}"
    elif granularity == "Weekly":
        this_label = _format_week_label(this_period)
        prev_label = _format_week_label(prev_period)
        csv_suffix = f"week_{this_period.strftime('%Y-%m-%d')}_vs_{prev_period.strftime('%Y-%m-%d')}"
    else:
        this_label = _format_month_label(this_period)
        prev_label = _format_month_label(prev_period)
        csv_suffix = f"month_{this_period.strftime('%Y-%m')}_vs_{prev_period.strftime('%Y-%m')}"

    st.caption(
        f"Comparing {this_label} (This Period) vs {prev_label} (Previous Period) — "
        "deltas are This Period minus Previous Period"
    )

    if this_period == prev_period:
        st.warning(
            "⚠️ You've selected the same period for both. "
            "Choose different periods to see deltas."
        )
        st.stop()

    this_period_dt = pd.Timestamp(this_period)
    prev_period_dt = pd.Timestamp(prev_period)

    if prev_period_dt > this_period_dt:
        st.warning(
            f"⚠️ Heads up: your Previous Period ({prev_label}) is later than "
            f"This Period ({this_label}). Deltas will read 'backwards' — the "
            "biggest drainers are actually gainers in real-life time order, "
            "and vice versa."
        )

    # ── Fetch the two snapshots from Supabase ────────────────────────────────
    period_df = _prepare(db.get_data_for_dates(gran_key, [prev_period, this_period]))

    # ── Cascading sub-filters ────────────────────────────────────────────────
    for _col in FILTER_COLS:
        st.session_state.setdefault(f"flt_{_col}", [])

    def _options_for(col: str) -> list:
        d = period_df
        for other in FILTER_COLS:
            if other == col:
                continue
            sel = st.session_state.get(f"flt_{other}", [])
            if sel:
                d = d[d[other].isin(sel)]
        return sorted(d[col].dropna().astype(str).unique().tolist())

    for _col in FILTER_COLS:
        opts = _options_for(_col)
        key = f"flt_{_col}"
        st.session_state[key] = [v for v in st.session_state[key] if v in opts]

    with st.expander("Filters (optional)", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.multiselect("Platform", _options_for("Platform"), key="flt_Platform")
        with fc2:
            st.multiselect("Category", _options_for("Category"), key="flt_Category")
        with fc3:
            st.multiselect("Ptype", _options_for("Ptype"), key="flt_Ptype")
        with fc4:
            st.multiselect("Variant", _options_for("Variant"), key="flt_Variant")

        if st.button("Clear all filters"):
            for _col in FILTER_COLS:
                st.session_state[f"flt_{_col}"] = []
            st.rerun()

    filtered = period_df
    for _col in FILTER_COLS:
        sel = st.session_state.get(f"flt_{_col}", [])
        if sel:
            filtered = filtered[filtered[_col].astype(str).isin(sel)]

    curr_all = filtered[filtered["Date"] == this_period_dt].copy()
    prev_all = filtered[filtered["Date"] == prev_period_dt].copy()

    if curr_all.empty and prev_all.empty:
        st.info("No data for selected range.")
        st.stop()

    # ── Build merged dataset (OUTER so missing-side rows survive) ────────────
    curr_sel = curr_all[KEY_COLS + METRIC_COLS].drop_duplicates(subset=KEY_COLS).copy()
    prev_sel = prev_all[KEY_COLS + METRIC_COLS].drop_duplicates(subset=KEY_COLS).copy()
    merged_all = curr_sel.merge(prev_sel, on=KEY_COLS, how="outer", suffixes=("_c", "_p"))

    def _f(col: str) -> pd.Series:
        return merged_all[col].fillna(0)

    def _pct_chg(c: str, p: str) -> np.ndarray:
        d = _f(c) - _f(p)
        return np.where(_f(p) != 0, d / _f(p) * 100, np.nan)

    merged_all["MS_pp"] = _f("MS_MRP_c") - _f("MS_MRP_p")
    merged_all["MS_pct"] = _pct_chg("MS_MRP_c", "MS_MRP_p")
    merged_all["Oft_abs"] = _f("Offtake_MRP_c") - _f("Offtake_MRP_p")
    merged_all["Oft_pct"] = _pct_chg("Offtake_MRP_c", "Offtake_MRP_p")
    merged_all["OSA_d"] = _f("Wt_OSA_c") - _f("Wt_OSA_p")
    merged_all["AdSOV_d"] = _f("Ad_SOV_c") - _f("Ad_SOV_p")
    merged_all["OvSOV_d"] = _f("Overall_SOV_c") - _f("Overall_SOV_p")
    merged_all["ASP_d"] = _f("Selling_Price_c") - _f("Selling_Price_p")

    # ── Split: eligible (non-NULL MS AND Offtake on both sides) vs MZ ────────
    # Eligibility is purely about NULL: a value of 0 is valid (real zero) and
    # stays in the main tables. The split into Drainers/Gainers below is by
    # MS_pp sign, not Offtake — a row that lost MS while growing offtake is a
    # Drainer (we lost share even while growing — the category outgrew us).
    ms_c = merged_all["MS_MRP_c"]
    ms_p = merged_all["MS_MRP_p"]
    oft_c = merged_all["Offtake_MRP_c"]
    oft_p = merged_all["Offtake_MRP_p"]
    mz_mask = ms_c.isna() | ms_p.isna() | oft_c.isna() | oft_p.isna()
    clean = merged_all[~mz_mask].copy()
    missing_zero = merged_all[mz_mask].copy()

    # Top-80% of current offtake, computed within the cleaned, filtered subset.
    clean_sorted = clean.sort_values("Offtake_MRP_c", ascending=False)
    total_oft = clean_sorted["Offtake_MRP_c"].sum()
    if total_oft > 0:
        cumshare = clean_sorted["Offtake_MRP_c"].cumsum() / total_oft
        top80 = clean_sorted[cumshare <= 0.80]
    else:
        top80 = clean_sorted

    # Split BY MS_pp sign; SORT BY Offtake abs Δ (these are intentionally
    # independent — see comment above the eligibility check).
    drainers = (
        top80[top80["MS_pp"] < 0]
        .sort_values(["Oft_abs", "Product_Title"], ascending=[True, True])
        .reset_index(drop=True)
    )
    gainers = (
        top80[top80["MS_pp"] >= 0]
        .sort_values(["Oft_abs", "Product_Title"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # Top-50% cumulative offtake-contribution mask (per table).
    # Phase 8A split means each table can contain rows with the "wrong" Oft_abs
    # sign (e.g. a MS-Gainer whose offtake actually declined). Clip those to 0
    # so they don't poison the total and never get highlighted as contributors.
    drainer_mask = _top50_mask((-drainers["Oft_abs"]).clip(lower=0))
    gainer_mask = _top50_mask(gainers["Oft_abs"].clip(lower=0))

    # csv_suffix is computed once with the picker block, granularity-aware

    # ── Render Drainers ──────────────────────────────────────────────────────
    st.markdown("### 🔻 Drainers — Where we're losing Market Share")
    st.caption(
        f"Top 80% of current offtake only · {len(drainers):,} row(s) · "
        "rows highlighted red contribute to the first 50% of total offtake decline"
    )
    if drainers.empty:
        st.info("No drainers for the selected period.")
    else:
        st.dataframe(
            _style(drainers, row_mask=drainer_mask, row_color=ROW_BG_DRAINERS),
            width="stretch",
            hide_index=True,
            column_config={
                "SKU": st.column_config.TextColumn(width="medium"),
            },
        )
        st.download_button(
            "📥 Download as CSV",
            data=_drainers_gainers_csv_bytes(drainers),
            file_name=f"drainers_{csv_suffix}.csv",
            mime="text/csv",
            key="drainers_csv_btn",
        )

    # ── Render Gainers ───────────────────────────────────────────────────────
    st.markdown("### 🔺 Gainers — Where we're gaining Market Share")
    st.caption(
        f"Top 80% of current offtake only · {len(gainers):,} row(s) · "
        "rows highlighted green contribute to the first 50% of total offtake growth"
    )
    if gainers.empty:
        st.info("No gainers for the selected period.")
    else:
        st.dataframe(
            _style(gainers, row_mask=gainer_mask, row_color=ROW_BG_GAINERS),
            width="stretch",
            hide_index=True,
            column_config={
                "SKU": st.column_config.TextColumn(width="medium"),
            },
        )
        st.download_button(
            "📥 Download as CSV",
            data=_drainers_gainers_csv_bytes(gainers),
            file_name=f"gainers_{csv_suffix}.csv",
            mime="text/csv",
            key="gainers_csv_btn",
        )

    # ── Render missing/zero offtake expander ─────────────────────────────────
    mz_view = _build_mz_view(missing_zero)
    with st.expander(f"📭 SKUs with missing or zero offtake ({len(mz_view):,})", expanded=False):
        if mz_view.empty:
            st.info("No SKUs with missing or zero offtake in the current view.")
        else:
            mz_styler = mz_view.style.format(
                {
                    "Offtake (MRP) This Week": fr,
                    "Offtake (MRP) Previous Week": fr,
                },
                na_rep=NA,
            )
            st.dataframe(
                mz_styler,
                width="stretch",
                hide_index=True,
                column_config={
                    "SKU": st.column_config.TextColumn(width="medium"),
                },
            )
            st.download_button(
                "📥 Download as CSV",
                data=mz_view.to_csv(index=False).encode("utf-8"),
                file_name=f"missing_zero_offtake_{csv_suffix}.csv",
                mime="text/csv",
                key="mz_csv_btn",
            )
