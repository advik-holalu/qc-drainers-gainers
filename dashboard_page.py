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
FILTER_COLS = ["Platform", "City", "Category", "Ptype", "Variant"]

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
    "MS Prev": fp,             "MS Current": fp,
    "MS PP Δ": fpd,            "MS % Δ": fpd,
    "Offtake Current": fr,     "Offtake Prev": fr,
    "Offtake Abs Δ": frd,      "Offtake % Δ": fpd,
    "OSA Prev": fp,            "OSA Current": fp,        "OSA PP Δ": fpd,
    "Ad SOV Prev": fp,         "Ad SOV Current": fp,     "Ad SOV PP Δ": fpd,
    "Overall SOV Prev": fp,    "Overall SOV Current": fp, "Overall SOV PP Δ": fpd,
    "ASP Prev": fra,           "ASP Current": fra,       "ASP Abs Δ": frad,
}

_DELTA_COLS = [
    "MS PP Δ", "MS % Δ",
    "Offtake Abs Δ", "Offtake % Δ",
    "OSA PP Δ",
    "Ad SOV PP Δ",
    "Overall SOV PP Δ",
    "ASP Abs Δ",
]


def _resolve_sku_column(data: pd.DataFrame) -> pd.Series:
    """Return the Series of strings to display in the SKU column.

    Honours the sidebar toggle:
      • ERP Name      → erp_name, with Product_Title as fallback for NULLs.
      • Platform Name → Product_Title (the GobbleCube name).
    """
    mode = st.session_state.get("sku_display_mode", "ERP Name")
    title = data["Product_Title"]
    if mode == "ERP Name" and "erp_name" in data.columns:
        erp = data["erp_name"]
        return erp.where(erp.notna(), title)
    return title


def _make_display(data: pd.DataFrame) -> pd.DataFrame:
    # Flat single-row headers (no MultiIndex). Identifier order: SKU, City,
    # Platform first (these are pinned-left at render time via column_config
    # to stay frozen while the user scrolls the metric columns horizontally).
    cols = {
        "SKU": _resolve_sku_column(data).values,
        "City": data["City"].values,
        "Platform": data["Platform"].values,
        "Ptype": data["Ptype"].values,
        "Variant": data["Variant"].values,
        "Category": data["Category"].values,
        "MS Prev": data["MS_MRP_p"].values,
        "MS Current": data["MS_MRP_c"].values,
        "MS PP Δ": data["MS_pp"].values,
        "MS % Δ": data["MS_pct"].values,
        "Offtake Current": data["Offtake_MRP_c"].values,
        "Offtake Prev": data["Offtake_MRP_p"].values,
        "Offtake Abs Δ": data["Oft_abs"].values,
        "Offtake % Δ": data["Oft_pct"].values,
        "OSA Prev": data["Wt_OSA_p"].values,
        "OSA Current": data["Wt_OSA_c"].values,
        "OSA PP Δ": data["OSA_d"].values,
        "Ad SOV Prev": data["Ad_SOV_p"].values,
        "Ad SOV Current": data["Ad_SOV_c"].values,
        "Ad SOV PP Δ": data["AdSOV_d"].values,
        "Overall SOV Prev": data["Overall_SOV_p"].values,
        "Overall SOV Current": data["Overall_SOV_c"].values,
        "Overall SOV PP Δ": data["OvSOV_d"].values,
        "ASP Prev": data["Selling_Price_p"].values,
        "ASP Current": data["Selling_Price_c"].values,
        "ASP Abs Δ": data["ASP_d"].values,
    }
    return pd.DataFrame(cols)


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


def _drainers_gainers_csv_bytes(data: pd.DataFrame) -> bytes:
    """CSV with raw, unformatted values matching the user's filtered+sorted view.
    Always exports BOTH Product (ERP) and Product (Platform) at the front,
    regardless of the on-screen toggle, so analysts have full info offline."""
    platform_series = data["Product_Title"]
    if "erp_name" in data.columns:
        erp_series = data["erp_name"].where(data["erp_name"].notna(), platform_series)
    else:
        erp_series = platform_series
    display = _make_display(data).copy()
    if "SKU" in display.columns:
        display = display.drop(columns=["SKU"])
    display.insert(0, "Product (Platform)", platform_series.values)
    display.insert(0, "Product (ERP)", erp_series.values)
    return display.to_csv(index=False).encode("utf-8")


def _build_mz_view(missing_zero: pd.DataFrame) -> pd.DataFrame:
    if missing_zero.empty:
        return pd.DataFrame(
            columns=[
                "SKU", "City", "Platform", "Ptype", "Variant", "Category",
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
        "SKU": missing_zero["Product_Title"].values,
        "City": missing_zero["City"].values,
        "Platform": missing_zero["Platform"].values,
        "Ptype": missing_zero["Ptype"].values,
        "Variant": missing_zero["Variant"].values,
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
    """Format a weekly snapshot date as 'Mon DD, YYYY' (e.g. 'Apr 27, 2026')."""
    return d.strftime("%b %d, %Y")


def _format_month_label(d: date) -> str:
    """Format a monthly snapshot date as 'April 2026'."""
    return d.strftime("%B %Y")


def _render_sku_list_section(
    *,
    label: str,
    saved: list[str],
    universe: list[str],
    multiselect_key: str,
    save_button_key: str,
    save_fn,
) -> None:
    """Render a 'Focus SKUs' / 'New SKUs' sidebar section: header + multiselect
    + dirty-aware Save button. `save_fn` is db.replace_focus_skus or
    db.replace_new_skus. The multiselect's `default` only takes effect on first
    render; after that, session_state[multiselect_key] drives the widget. Dirty
    detection compares sorted lists so order doesn't matter.
    """
    st.sidebar.markdown(f"**{label}**")
    if not universe and not saved:
        st.sidebar.caption(
            "No ERP names available — add SKU mappings via "
            "Upload Data → Update SKU Mapping"
        )
        return

    # Defensive: extend options so any saved or in-session selections that
    # aren't currently in the universe (e.g. the mapping was edited later)
    # still appear in the dropdown rather than crashing the widget.
    existing = st.session_state.get(multiselect_key, saved)
    options = sorted(set(universe) | set(saved) | set(existing))

    current = st.sidebar.multiselect(
        label,
        options=options,
        default=saved,
        key=multiselect_key,
        label_visibility="collapsed",
    )
    dirty = sorted(current) != sorted(saved)
    if dirty:
        if st.sidebar.button(
            "● Save Changes",
            type="primary",
            key=save_button_key,
            use_container_width=True,
        ):
            result = save_fn(current)
            if result.get("success"):
                st.toast(f"Saved {result.get('count', 0)} {label}", icon="✅")
                st.rerun()
            else:
                st.sidebar.error(f"Save failed: {result.get('error', 'unknown')}")
        st.sidebar.caption("Unsaved changes")
    else:
        st.sidebar.button(
            "Saved ✓",
            disabled=True,
            key=save_button_key,
            use_container_width=True,
        )


def _snap_to_available(picked, available: list[date]) -> date | None:
    """Return the available date closest to picked (None if list is empty)."""
    if not available:
        return None
    if picked in available:
        return picked
    return min(available, key=lambda d: abs((d - picked).days))


# ── Main render ──────────────────────────────────────────────────────────────
def render_dashboard() -> None:
    # ── Page header (main body) ──────────────────────────────────────────────
    title_col, refresh_col = st.columns([6, 1])
    with title_col:
        st.title("QC Drainers/Gainers Dashboard")
    with refresh_col:
        st.write("")  # vertical alignment with the title
        if st.button("🔄 Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    # The "Comparing X vs Y" panel is pinned at the very top of the sidebar so
    # it's always visible, but its values depend on the date pickers below.
    # We render a placeholder first, then write into it after the pickers
    # resolve their selected periods.
    comparing_slot = st.sidebar.empty()

    # ── SKU display toggle ──────────────────────────────────────────────────
    # Decoupled widget key: persistent `sku_display_mode` survives unmounting
    # when the user navigates away from the dashboard (Phase 7 pattern).
    _sku_opts = ["ERP Name", "Platform Name"]
    st.session_state.setdefault("sku_display_mode", _sku_opts[0])
    _current_mode = st.session_state.sku_display_mode
    _mode_idx = _sku_opts.index(_current_mode) if _current_mode in _sku_opts else 0
    st.session_state.sku_display_mode = st.sidebar.radio(
        "Display SKU as",
        options=_sku_opts,
        index=_mode_idx,
        horizontal=True,
        key="sku_display_mode_widget",
    )

    st.sidebar.markdown("---")

    # ── Granularity toggle (sidebar) ─────────────────────────────────────────
    granularity = st.sidebar.radio(
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

    # ── Data last updated (main body, below title) ───────────────────────────
    last_updated = db.get_last_updated(gran_key)
    if last_updated is not None:
        st.caption(f"Data last updated: {_format_last_updated(last_updated)}")

    # ── Period pickers (sidebar; widget type depends on granularity) ─────────
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

    if granularity == "Daily":
        picked_this = st.sidebar.date_input(
            "This Period",
            value=this_default,
            min_value=available_dates[0],
            max_value=available_dates[-1],
            key="this_period_date",
        )
        picked_prev = st.sidebar.date_input(
            "Previous Period",
            value=prev_default,
            min_value=available_dates[0],
            max_value=available_dates[-1],
            key="prev_period_date",
        )
        this_period = _snap_to_available(picked_this, available_dates)
        prev_period = _snap_to_available(picked_prev, available_dates)
        if this_period != picked_this and this_period is not None:
            st.sidebar.caption(f"Snapped This Period to nearest available date: {this_period.strftime('%d/%m/%Y')}")
        if prev_period != picked_prev and prev_period is not None:
            st.sidebar.caption(f"Snapped Previous Period to nearest available date: {prev_period.strftime('%d/%m/%Y')}")

    elif granularity == "Weekly":
        # Defensive: drop any session-state value that's no longer in options
        for k, default in [("this_period_week", this_default), ("prev_period_week", prev_default)]:
            if k in st.session_state and st.session_state[k] not in options_desc:
                st.session_state[k] = default
        this_period = st.sidebar.selectbox(
            "This Period",
            options=options_desc,
            format_func=_format_week_label,
            index=0 if "this_period_week" not in st.session_state else None,
            key="this_period_week",
        )
        prev_period = st.sidebar.selectbox(
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
        this_period = st.sidebar.selectbox(
            "This Period",
            options=options_desc,
            format_func=_format_month_label,
            index=0 if "this_period_month" not in st.session_state else None,
            key="this_period_month",
        )
        prev_period = st.sidebar.selectbox(
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

    # Fill the pinned "Comparing" slot at the top of the sidebar now that we
    # know the resolved periods. Styled card: subdued labels, emphasised values,
    # subtle border + tint, compact vertical footprint.
    comparing_slot.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 4px;
            line-height: 1.4;
        ">
          <div style="font-size: 0.78em; letter-spacing: 0.05em; opacity: 0.65; margin-bottom: 8px;">📊 COMPARING</div>
          <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
            <span style="font-size: 0.82em; opacity: 0.6;">This</span>
            <span style="font-weight: 600; font-size: 1.0em;">{this_label}</span>
          </div>
          <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span style="font-size: 0.82em; opacity: 0.6;">Previous</span>
            <span style="font-weight: 600; font-size: 1.0em;">{prev_label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
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

    # ── Cascading sub-filters (sidebar) ──────────────────────────────────────
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

    st.sidebar.divider()
    st.sidebar.markdown("**Filters**")
    st.sidebar.multiselect("Platform", _options_for("Platform"), key="flt_Platform")
    st.sidebar.multiselect("City", _options_for("City"), key="flt_City")
    st.sidebar.multiselect("Category", _options_for("Category"), key="flt_Category")
    st.sidebar.multiselect("Ptype", _options_for("Ptype"), key="flt_Ptype")
    st.sidebar.multiselect("Variant", _options_for("Variant"), key="flt_Variant")

    if st.sidebar.button("Clear all filters"):
        for _col in FILTER_COLS:
            st.session_state[f"flt_{_col}"] = []
        st.rerun()

    # ── Focus / New SKU lists (sidebar) ──────────────────────────────────────
    # Saved state — drives both the "Show only" filter and the dirty detection
    # on the multiselects. We read these BEFORE rendering the multiselects so
    # the filter uses what's saved in Supabase, not the unsaved pending state.
    saved_focus = db.get_focus_skus()
    saved_new = db.get_new_skus()

    # Universe of valid ERP names from sku_mapping. Defensive: strip whitespace,
    # drop NaN/empty, deduplicate, sort.
    _mapping_df = db.get_sku_mapping_df()
    if _mapping_df.empty or "erp_name" not in _mapping_df.columns:
        erp_universe: list[str] = []
    else:
        erp_universe = sorted({
            str(n).strip()
            for n in _mapping_df["erp_name"].dropna()
            if str(n).strip()
        })

    st.sidebar.divider()
    st.sidebar.markdown("**Show only**")
    st.sidebar.checkbox("Focus SKUs", key="show_only_focus_widget")
    st.sidebar.checkbox("New SKUs", key="show_only_new_widget")

    _render_sku_list_section(
        label="Focus SKUs",
        saved=saved_focus,
        universe=erp_universe,
        multiselect_key="focus_skus_multiselect",
        save_button_key="focus_save_btn",
        save_fn=db.replace_focus_skus,
    )

    st.sidebar.markdown("")  # visual breath between the two sections

    _render_sku_list_section(
        label="New SKUs",
        saved=saved_new,
        universe=erp_universe,
        multiselect_key="new_skus_multiselect",
        save_button_key="new_save_btn",
        save_fn=db.replace_new_skus,
    )

    # ── Apply filters to the dataframe ───────────────────────────────────────
    filtered = period_df
    for _col in FILTER_COLS:
        sel = st.session_state.get(f"flt_{_col}", [])
        if sel:
            filtered = filtered[filtered[_col].astype(str).isin(sel)]

    # Show-only filter: applied AFTER the regular filters and BEFORE the split
    # so the top-80%/top-50% logic runs on the focus/new subset. Filter uses the
    # SAVED list (not the pending multiselect state) — unsaved changes don't
    # affect the visible tables until the user clicks Save.
    show_focus = bool(st.session_state.get("show_only_focus_widget", False))
    show_new = bool(st.session_state.get("show_only_new_widget", False))
    if show_focus or show_new:
        allowed: set[str] = set()
        if show_focus:
            allowed |= set(saved_focus)
        if show_new:
            allowed |= set(saved_new)
        if "erp_name" in filtered.columns:
            filtered = filtered[filtered["erp_name"].isin(allowed)]
        else:
            filtered = filtered.iloc[0:0]  # no erp_name → can't satisfy filter

    curr_all = filtered[filtered["Date"] == this_period_dt].copy()
    prev_all = filtered[filtered["Date"] == prev_period_dt].copy()

    if curr_all.empty and prev_all.empty:
        st.info("No data for selected range.")
        st.stop()

    # ── Build merged dataset (OUTER so missing-side rows survive) ────────────
    # Pass-through columns (not metrics, not merge keys) that we want to keep
    # available on the merged frame for display + unmapped-count tracking.
    PASS_THROUGH = [c for c in ("Item_ID", "erp_name") if c in filtered.columns]
    curr_sel = (
        curr_all[KEY_COLS + METRIC_COLS + PASS_THROUGH]
        .drop_duplicates(subset=KEY_COLS).copy()
    )
    prev_sel = (
        prev_all[KEY_COLS + METRIC_COLS + PASS_THROUGH]
        .drop_duplicates(subset=KEY_COLS).copy()
    )
    merged_all = curr_sel.merge(prev_sel, on=KEY_COLS, how="outer", suffixes=("_c", "_p"))

    # Collapse the suffixed pass-through columns (e.g. Item_ID_c / Item_ID_p)
    # back to a single column, preferring current; fall back to previous so
    # rows that only exist in prev still have their identifiers populated.
    for col in PASS_THROUGH:
        c, p = f"{col}_c", f"{col}_p"
        if c in merged_all.columns and p in merged_all.columns:
            merged_all[col] = merged_all[c].combine_first(merged_all[p])
            merged_all = merged_all.drop(columns=[c, p])

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

    def _unmapped_caption(df: pd.DataFrame) -> str | None:
        """Caption string with distinct unmapped item_id count, or None to hide.
        Hidden when the toggle is on Platform Name or there's nothing to flag."""
        if st.session_state.get("sku_display_mode", "ERP Name") != "ERP Name":
            return None
        if "erp_name" not in df.columns:
            return None
        unmapped = df[df["erp_name"].isna()]
        if "Item_ID" in df.columns:
            n = unmapped["Item_ID"].nunique(dropna=True)
        else:
            n = unmapped["Product_Title"].nunique(dropna=True)
        if n == 0:
            return None
        return (
            f"{n} SKUs in this view are unmapped · update mapping in "
            "Upload Data → Update SKU Mapping"
        )

    # csv_suffix is computed once with the picker block, granularity-aware

    # ── Render Drainers ──────────────────────────────────────────────────────
    st.markdown("### 🔻 Drainers — Where we're losing Market Share")
    st.markdown(
        f"**Total: {len(drainers):,} rows** · Top 80% of current offtake · "
        "Red rows = first 50% of total offtake decline"
    )
    _drainer_unmapped_msg = _unmapped_caption(drainers)
    if _drainer_unmapped_msg:
        st.caption(_drainer_unmapped_msg)
    if drainers.empty:
        st.info("No drainers for the selected period.")
    else:
        st.dataframe(
            _style(drainers, row_mask=drainer_mask, row_color=ROW_BG_DRAINERS),
            width="stretch",
            hide_index=True,
            column_config={
                "SKU": st.column_config.TextColumn("SKU", pinned="left", width="medium"),
                "City": st.column_config.TextColumn("City", pinned="left"),
                "Platform": st.column_config.TextColumn("Platform", pinned="left"),
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
    st.markdown(
        f"**Total: {len(gainers):,} rows** · Top 80% of current offtake · "
        "Green rows = first 50% of total offtake growth"
    )
    _gainer_unmapped_msg = _unmapped_caption(gainers)
    if _gainer_unmapped_msg:
        st.caption(_gainer_unmapped_msg)
    if gainers.empty:
        st.info("No gainers for the selected period.")
    else:
        st.dataframe(
            _style(gainers, row_mask=gainer_mask, row_color=ROW_BG_GAINERS),
            width="stretch",
            hide_index=True,
            column_config={
                "SKU": st.column_config.TextColumn("SKU", pinned="left", width="medium"),
                "City": st.column_config.TextColumn("City", pinned="left"),
                "Platform": st.column_config.TextColumn("Platform", pinned="left"),
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
