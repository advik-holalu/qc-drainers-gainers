from __future__ import annotations

import re

import pandas as pd
import streamlit as st

import db


# Canonical column names expected in CSV uploads from GobbleCube.
# Column matching is fuzzy: case, spaces, underscores, hyphens, dots,
# and parentheses are ignored. So 'P Type', 'p_type', 'PTYPE' all match 'Ptype'.
# Column order in the CSV doesn't matter — columns are reordered to match
# this list before processing.
# Extra columns in the CSV are silently dropped.
CANONICAL_COLUMNS = [
    "Platform", "Date", "City", "Category", "Product ID",
    "Product Title", "Grammage", "Brand", "Item ID",
    "Offtake (MRP)", "Offtake (SP)", "Est. Category Share",
    "Est. Category Share (SP)", "Overall SOV", "Ad SOV",
    "Wt. OSA %", "Avg. OSA %", "MRP", "Selling Price",
    "Ptype", "Variant",
]
REQUIRED_NON_NULL = ["Platform", "Date", "City", "Product Title"]
DEDUP_KEYS_DISPLAY = ["Date", "Platform", "City", "Product Title", "Ptype", "Variant"]
GRANULARITY_OPTIONS = ["Daily", "Weekly", "Monthly"]
MODE_OPTIONS = ["Append", "Replace"]


def _normalize_col_name(name: str) -> str:
    """Lowercase + strip whitespace / underscores / hyphens / dots / parentheses
    for fuzzy column-name matching."""
    return re.sub(r"[\s_\-\.\(\)]+", "", str(name).strip().lower())


def _build_column_rename_map(
    df_columns: list[str], canonical_columns: list[str]
) -> tuple[dict[str, str], list[str]]:
    """For each canonical name, find the column in the file whose normalized
    form matches. Returns (rename_map, missing_canonical)."""
    normalized_df = {_normalize_col_name(c): c for c in df_columns}
    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for canonical in canonical_columns:
        key = _normalize_col_name(canonical)
        if key in normalized_df:
            rename_map[normalized_df[key]] = canonical
        else:
            missing.append(canonical)
    return rename_map, missing


# ── state helpers ────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "upload_authed": False,
        "upload_step": "config",
        "upload_file_df": None,
        "upload_validation": None,
        "upload_result": None,
        "upload_granularity": "Weekly",
        "upload_mode": "Append",
        "upload_file_widget_counter": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _reset_upload_state() -> None:
    st.session_state.upload_step = "config"
    st.session_state.upload_file_df = None
    st.session_state.upload_validation = None
    st.session_state.upload_result = None
    # Bump the counter so the file_uploader widget gets a fresh key and clears.
    st.session_state.upload_file_widget_counter += 1


# ── validation ───────────────────────────────────────────────────────────────
def _validate_file(uploaded_file) -> dict:
    result: dict = {"valid": False, "errors": [], "warnings": [], "df": None, "stats": {}}

    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"Could not parse CSV: {exc}")
        return result

    rows_in_file = len(df)

    # Fuzzy column matching — tolerate case / spacing / underscore / hyphen / dot
    # / paren variations and any column order coming out of GobbleCube exports.
    rename_map, missing = _build_column_rename_map(df.columns.tolist(), CANONICAL_COLUMNS)
    if missing:
        result["errors"].append(
            "Missing required columns (or recognizable variants): " + ", ".join(missing)
        )
        return result
    df = df.rename(columns=rename_map)

    null_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_NON_NULL}
    if any(n > 0 for n in null_counts.values()):
        bad = ", ".join(f"{c} ({n} null)" for c, n in null_counts.items() if n > 0)
        result["errors"].append(f"Null values in required fields: {bad}")
        return result

    try:
        parsed = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="raise")
        df["Date"] = parsed.dt.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        bad_mask = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce").isna()
        examples = df.loc[bad_mask, "Date"].astype(str).head(5).tolist()
        result["errors"].append(
            f"Could not parse Date column as M/D/YY. First failing values: {examples}"
        )
        return result

    # Restrict to canonical columns in canonical order — drops any extras silently
    # and gives downstream code a guaranteed column layout.
    df = df[CANONICAL_COLUMNS].copy()

    # Normalize "0", "", whitespace in Ptype/Variant to None BEFORE dedup so that
    # rows differing only on these (with one side "0" and the other NULL) collapse
    # together — matching the DB's UNIQUE NULLS NOT DISTINCT constraint.
    for col in ("Ptype", "Variant"):
        df[col] = df[col].apply(
            lambda v: None if (pd.isna(v) or str(v).strip() in ("", "0")) else v
        )

    before = len(df)
    df["_completeness"] = df.notnull().sum(axis=1)
    df = df.sort_values("_completeness", ascending=False, kind="stable")
    df = df.drop_duplicates(subset=DEDUP_KEYS_DISPLAY, keep="first")
    df = df.drop(columns=["_completeness"]).reset_index(drop=True)
    after = len(df)
    duplicates_removed = before - after

    unique_dates = sorted(df["Date"].astype(str).unique().tolist())
    platforms = sorted(df["Platform"].dropna().astype(str).unique().tolist())
    cities_count = int(df["City"].dropna().astype(str).nunique())

    result["valid"] = True
    result["df"] = df
    result["stats"] = {
        "rows_in_file": rows_in_file,
        "rows_after_dedup": after,
        "duplicates_removed": duplicates_removed,
        "date_range": (unique_dates[0], unique_dates[-1]) if unique_dates else ("—", "—"),
        "unique_dates": unique_dates,
        "platforms": platforms,
        "cities_count": cities_count,
    }
    if duplicates_removed > 0:
        result["warnings"].append(
            f"{duplicates_removed:,} duplicate rows were removed before upload."
        )
    return result


# ── step renderers ───────────────────────────────────────────────────────────
def _render_password_gate() -> None:
    st.title("📤 Upload Data")
    st.write("This page is password-protected.")
    st.text_input("Password", type="password", key="upload_pw_input")
    if st.button("Login", key="upload_login_btn"):
        try:
            expected = st.secrets["UPLOAD_PASSWORD"]
        except (FileNotFoundError, KeyError):
            st.error(
                "UPLOAD_PASSWORD is not configured. Add it to .streamlit/secrets.toml."
            )
            return
        if st.session_state.upload_pw_input == expected:
            st.session_state.upload_authed = True
            st.rerun()
        else:
            st.error("Incorrect password")


def _render_authed_header() -> None:
    title_col, logout_col = st.columns([6, 1])
    with title_col:
        st.title("📤 Upload Data")
    with logout_col:
        st.write("")
        if st.button("Logout", key="upload_logout_btn", use_container_width=True):
            st.session_state.upload_authed = False
            _reset_upload_state()
            st.rerun()



def _render_config_step() -> None:
    # CRITICAL: the widget's `key=` is intentionally DIFFERENT from our
    # persistent storage key. Streamlit clears widget state when the widget
    # is not rendered (e.g., when we switch to the preview/uploading step),
    # which previously caused upload_granularity to be wiped between preview
    # and uploading — _init_state's setdefault then re-defaulted to "Weekly",
    # silently routing Daily uploads to weekly_data.
    # Pattern: read persistent → set widget index → on user click, sync back.
    current_gran = st.session_state.upload_granularity
    gran_idx = (
        GRANULARITY_OPTIONS.index(current_gran)
        if current_gran in GRANULARITY_OPTIONS
        else 1  # Weekly fallback
    )
    new_gran = st.radio(
        "Granularity",
        options=GRANULARITY_OPTIONS,
        index=gran_idx,
        horizontal=True,
        key="upload_granularity_widget",  # transient; not relied on for persistence
    )
    st.session_state.upload_granularity = new_gran

    current_mode = st.session_state.upload_mode
    mode_idx = (
        MODE_OPTIONS.index(current_mode)
        if current_mode in MODE_OPTIONS
        else 0
    )
    new_mode = st.radio(
        "Mode",
        options=MODE_OPTIONS,
        index=mode_idx,
        horizontal=True,
        help=(
            "Append adds new rows or updates existing ones. "
            "Replace wipes the entire table first — use with extreme caution."
        ),
        key="upload_mode_widget",
    )
    st.session_state.upload_mode = new_mode

    gran = st.session_state.upload_granularity
    target_table = f"{gran.lower()}_data"
    uploaded = st.file_uploader(
        f"Upload CSV file from GobbleCube  →  destination: {target_table}",
        type=["csv"],
        key=f"upload_file_widget_{st.session_state.upload_file_widget_counter}",
    )

    if uploaded is None:
        return

    with st.spinner("Validating file..."):
        validation = _validate_file(uploaded)
    st.session_state.upload_validation = validation

    if validation["valid"]:
        st.session_state.upload_file_df = validation["df"]
        st.session_state.upload_step = "preview"
        st.rerun()
    else:
        for err in validation["errors"]:
            st.error(err)


def _render_preview_step() -> None:
    st.subheader("Preview")
    validation = st.session_state.upload_validation
    stats = validation["stats"]
    gran = st.session_state.upload_granularity
    mode = st.session_state.upload_mode
    

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows in file", f"{stats['rows_in_file']:,}")
    c2.metric(
        "Rows after dedup",
        f"{stats['rows_after_dedup']:,}",
        delta=(
            f"-{stats['duplicates_removed']:,} duplicates"
            if stats["duplicates_removed"] > 0
            else None
        ),
        delta_color="off",
    )
    c3.metric("Unique dates", str(len(stats["unique_dates"])))
    c4.metric("Platforms", str(len(stats["platforms"])))

    date_lo, date_hi = stats["date_range"]
    target_table = f"{gran.lower()}_data"
    st.markdown(
        f"""
        - **Date range:** `{date_lo}` to `{date_hi}`
        - **Cities covered:** {stats['cities_count']}
        - **Granularity:** {gran}
        - **Mode:** {mode}
        - **Target table:** `{target_table}`
        """
    )

    if stats["duplicates_removed"] > 0:
        st.warning(
            f"⚠️ {stats['duplicates_removed']:,} duplicate rows were removed before upload."
        )

    if mode == "Append":
        try:
            overlap = db.get_existing_dates_for_overlap(gran.lower(), stats["unique_dates"])
        except Exception as exc:  # noqa: BLE001
            overlap = []
            st.warning(f"Could not check overlap with existing data: {exc}")
        if overlap:
            st.warning(
                "These dates already have data in the database. The upload will "
                "UPDATE existing rows with new values from your file:\n\n"
                + "\n".join(f"- `{d}`" for d in overlap)
            )

    st.markdown("**Sample (first 10 rows after cleanup):**")
    st.dataframe(st.session_state.upload_file_df.head(10), use_container_width=True)

    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", key="preview_cancel_btn", use_container_width=True):
            _reset_upload_state()
            st.rerun()
    with confirm_col:
        if st.button(
            f"Confirm and Upload to {target_table}",
            key="preview_confirm_btn",
            use_container_width=True,
            type="primary",
        ):
            if mode == "Replace":
                st.session_state.upload_step = "confirming_replace"
            else:
                st.session_state.upload_step = "uploading"
            st.rerun()


def _render_confirming_replace_step() -> None:
    gran = st.session_state.upload_granularity
    table_name = f"{gran.lower()}_data"
    phrase = f"REPLACE {gran.upper()} DATA"

    st.error(
        f"""
        ⚠️ **DANGER ZONE**

        You are about to **REPLACE** the entire **{gran}** table.

        This will permanently delete all existing rows in `{table_name}` and
        replace them with the contents of your file.

        **This action cannot be undone.**

        To proceed, type:  `{phrase}`
        """
    )

    st.text_input("Type confirmation phrase:", key="replace_phrase_input")

    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", key="replace_cancel_btn", use_container_width=True):
            st.session_state.upload_step = "preview"
            st.rerun()
    with confirm_col:
        if st.button(
            "Yes, Replace All Data",
            key="replace_yes_btn",
            use_container_width=True,
            type="primary",
        ):
            if st.session_state.replace_phrase_input == phrase:
                st.session_state.upload_step = "uploading"
                st.rerun()
            else:
                st.error(f"Phrase doesn't match. Type exactly: {phrase}")


def _render_uploading_step() -> None:
    st.subheader("Uploading…")
    gran = st.session_state.upload_granularity
    mode = st.session_state.upload_mode
    df = st.session_state.upload_file_df

    total_rows = len(df)
    progress_bar = st.progress(0.0, text=f"Uploading {total_rows:,} rows…")

    def on_progress(done_batches: int, total_batches: int) -> None:
        if total_batches <= 0:
            return
        frac = done_batches / total_batches
        approx_rows = min(int(frac * total_rows), total_rows)
        progress_bar.progress(
            frac,
            text=f"Uploaded ~{approx_rows:,} / {total_rows:,} rows "
                 f"({done_batches}/{total_batches} batches)",
        )

    try:
        result = db.insert_data(gran.lower(), df, mode.lower(), on_progress=on_progress)
    except Exception as exc:  # noqa: BLE001
        result = {
            "success": False,
            "rows_inserted": 0,
            "batches_succeeded": 0,
            "batches_failed": 1,
            "errors": [f"Upload failed: {exc}"],
        }

    progress_bar.empty()
    st.session_state.upload_result = result
    st.session_state.upload_step = "success"
    st.rerun()


def _render_success_step() -> None:
    result = st.session_state.upload_result
    gran = st.session_state.upload_granularity
    mode = st.session_state.upload_mode
    table_name = f"{gran.lower()}_data"

    if result["success"]:
        st.success(
            f"✅ Successfully uploaded {result['rows_inserted']:,} rows to {table_name}"
        )
    else:
        st.error(
            f"Upload finished with {result['batches_failed']} failed batch(es). "
            f"{result['rows_inserted']:,} rows were inserted."
        )

    st.markdown(
        f"""
        - **Rows inserted:** {result['rows_inserted']:,}
        - **Batches succeeded:** {result['batches_succeeded']}
        - **Mode:** {mode}
        - **Granularity:** {gran}
        """
    )

    if result["errors"]:
        with st.expander("Errors"):
            for err in result["errors"]:
                st.code(err)

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "📊 View in Dashboard",
            key="success_view_dashboard_btn",
            use_container_width=True,
        ):
            _reset_upload_state()
            st.session_state.page = "dashboard"
            st.rerun()
    with c2:
        if st.button(
            "📤 Upload another file",
            key="success_upload_another_btn",
            use_container_width=True,
        ):
            _reset_upload_state()
            st.rerun()


# ── public entrypoint ────────────────────────────────────────────────────────
def render_upload() -> None:
    _init_state()

    if not st.session_state.upload_authed:
        _render_password_gate()
        return

    _render_authed_header()

    step = st.session_state.upload_step
    if step == "config":
        _render_config_step()
    elif step == "preview":
        _render_preview_step()
    elif step == "confirming_replace":
        _render_confirming_replace_step()
    elif step == "uploading":
        _render_uploading_step()
    elif step == "success":
        _render_success_step()
    else:
        st.warning(f"Unknown upload step: {step}. Resetting.")
        _reset_upload_state()
        st.rerun()
