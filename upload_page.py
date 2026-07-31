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
# Ptype/Variant are no longer used by the dashboard (Phase 13). We still let
# the file provide them for backward compatibility, but if they're absent from
# the file we insert them as NULL rather than rejecting the upload.
OPTIONAL_COLUMNS = {"Ptype", "Variant"}
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
MAPPING_EXPECTED_COLUMNS = ["item_id", "Platform Item Name", "ERP Name", "Platform"]


def _init_state() -> None:
    defaults = {
        # Data upload flow (Tab 1)
        "upload_authed": False,
        "upload_step": "config",
        "upload_file_df": None,
        "upload_validation": None,
        "upload_result": None,
        "upload_granularity": "Weekly",
        "upload_mode": "Append",
        "upload_file_widget_counter": 0,
        # SKU mapping flow (Tab 2)
        "mapping_step": "config",
        "mapping_file_df": None,
        "mapping_result": None,
        "mapping_file_widget_counter": 0,
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


def _reset_mapping_state() -> None:
    st.session_state.mapping_step = "config"
    st.session_state.mapping_file_df = None
    st.session_state.mapping_result = None
    st.session_state.mapping_file_widget_counter += 1


def _validate_mapping_file(uploaded_file):
    """Returns (df_or_None, errors, warnings). df returned only if no errors."""
    errors: list[str] = []
    warnings: list[str] = []

    name = (uploaded_file.name or "").lower()
    try:
        uploaded_file.seek(0)
        if name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            errors.append("File must be .csv or .xlsx.")
            return None, errors, warnings
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Could not parse file: {exc}")
        return None, errors, warnings

    # Column count + exact-match (case-sensitive) header check
    if len(df.columns) != 4:
        errors.append(
            f"File must have exactly 4 columns; got {len(df.columns)}: {df.columns.tolist()}"
        )
        return None, errors, warnings
    actual = df.columns.tolist()
    if actual != MAPPING_EXPECTED_COLUMNS:
        errors.append(
            f"Column headers must be exactly {MAPPING_EXPECTED_COLUMNS} (case-sensitive); "
            f"got {actual}"
        )
        return None, errors, warnings

    # Normalize values
    df["item_id"] = df["item_id"].astype(str).str.strip()
    for col in ("Platform Item Name", "ERP Name", "Platform"):
        df[col] = df[col].astype(str).str.strip()
    # After astype(str) NaN becomes the literal "nan" — collapse both that and
    # the empty string to None for nullable columns so they store as SQL NULL.
    for col in ("Platform Item Name", "Platform"):
        df[col] = df[col].where(~df[col].isin(("", "nan", "NaN", "None")), None)

    # Identify bad rows. Show row numbers as 1-indexed including header (so row 2
    # = first data row, matching what Excel/spreadsheet users expect).
    empty_id = df["item_id"].isin(("", "nan", "NaN", "None")) | df["item_id"].isna()
    if empty_id.any():
        rows = [int(i) + 2 for i in df.index[empty_id].tolist()][:15]
        errors.append(f"Empty item_id at row(s): {rows}")

    empty_erp = df["ERP Name"].isin(("", "nan", "NaN", "None")) | df["ERP Name"].isna()
    if empty_erp.any():
        rows = [int(i) + 2 for i in df.index[empty_erp].tolist()][:15]
        errors.append(f"Empty ERP Name at row(s): {rows}")

    dup_mask = df.duplicated(subset=["item_id"], keep=False) & ~empty_id
    if dup_mask.any():
        dup_ids = df.loc[dup_mask, "item_id"].unique().tolist()[:10]
        errors.append(f"Duplicate item_id(s): {dup_ids}")

    if errors:
        return None, errors, warnings

    # Warnings (non-blocking)
    blank_platform = int(df["Platform"].isna().sum())
    if blank_platform > 0:
        warnings.append(f"{blank_platform} row(s) have empty Platform")
    blank_pname = int(df["Platform Item Name"].isna().sum())
    if blank_pname > 0:
        warnings.append(f"{blank_pname} row(s) have empty Platform Item Name")

    # Rename to DB column names for downstream insert
    db_df = df.rename(
        columns={
            "Platform Item Name": "platform_item_name",
            "ERP Name": "erp_name",
            "Platform": "platform",
        }
    )
    return db_df, errors, warnings


# ── SKU mapping flow (Tab 2) ─────────────────────────────────────────────────
def _render_mapping_config_step() -> None:
    st.subheader("🏷️ Update SKU Mapping")

    count = db.get_sku_mapping_count()
    if count > 0:
        df = db.get_sku_mapping_df()
        if "updated_at" in df.columns and not df.empty:
            try:
                latest = pd.to_datetime(df["updated_at"]).max()
                st.caption(
                    f"Currently mapped: **{count:,} SKUs** · "
                    f"Last updated: {latest.strftime('%d %b %Y, %H:%M')} UTC"
                )
            except Exception:  # noqa: BLE001
                st.caption(f"Currently mapped: **{count:,} SKUs**")
        else:
            st.caption(f"Currently mapped: **{count:,} SKUs**")
    else:
        st.caption("Currently mapped: **0 SKUs** (no mapping uploaded yet)")

    st.info(
        "Uploading a new file will **REPLACE** the entire mapping. The dashboard "
        "will immediately use the new ERP names for matching `item_id`s."
    )

    with st.expander("💡 Expected format"):
        st.markdown(
            """
            The file must have **exactly 4 columns** with these case-sensitive
            headers (any order in the file, but the headers must match exactly):

            | Column | Header | Notes |
            |---|---|---|
            | A | `item_id` | Primary key, required, no duplicates |
            | B | `Platform Item Name` | Optional |
            | C | `ERP Name` | Required |
            | D | `Platform` | Optional |

            Accepted file types: `.csv` or `.xlsx`.
            """
        )

    uploaded = st.file_uploader(
        "Upload SKU mapping file (.csv or .xlsx)",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key=f"mapping_file_widget_{st.session_state.mapping_file_widget_counter}",
    )
    if uploaded is None:
        return

    with st.spinner("Validating file…"):
        df, errors, warnings = _validate_mapping_file(uploaded)

    if errors:
        for err in errors:
            st.error(err)
        return

    # Stash warnings on the df so the preview step can show them too
    st.session_state.mapping_file_df = df
    st.session_state.mapping_warnings = warnings
    st.session_state.mapping_step = "preview"
    st.rerun()


def _render_mapping_preview_step() -> None:
    st.subheader("📋 Review SKU Mapping")
    df = st.session_state.mapping_file_df
    existing = db.get_sku_mapping_count()
    st.markdown(
        f"**{len(df):,} new mappings** will replace **{existing:,} existing mappings**"
    )

    for w in st.session_state.get("mapping_warnings", []) or []:
        st.warning(f"⚠️ {w}")

    st.dataframe(df.head(10), width="stretch", hide_index=True)

    st.error(
        "⚠️ This will **DELETE** the current mapping and replace it with the "
        "uploaded file. **This cannot be undone.**"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Cancel", key="mapping_cancel_btn", use_container_width=True):
            _reset_mapping_state()
            st.rerun()
    with c2:
        if st.button(
            "Confirm and Replace",
            key="mapping_confirm_btn",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.mapping_step = "uploading"
            st.rerun()


def _render_mapping_uploading_step() -> None:
    st.subheader("Replacing SKU mapping…")
    df = st.session_state.mapping_file_df
    with st.spinner("Writing to database…"):
        try:
            result = db.replace_sku_mapping(df)
        except Exception as exc:  # noqa: BLE001
            result = {"success": False, "error": f"Upload failed: {exc}"}
    st.session_state.mapping_result = result
    st.session_state.mapping_step = "success"
    st.rerun()


def _render_mapping_success_step() -> None:
    result = st.session_state.mapping_result or {}
    if result.get("success"):
        n = int(result.get("rows_inserted", 0))
        st.success(f"✅ SKU mapping updated")
        st.markdown(
            f"""
            - **Mappings inserted:** {n:,}
            - **Table:** `sku_mapping`
            - **Mode:** Replace (truncate + insert)
            """
        )
    else:
        st.error(f"Upload failed: {result.get('error', 'Unknown error')}")
        if "rows_inserted" in result:
            st.caption(f"{result['rows_inserted']:,} rows were inserted before failure.")

    if st.button("Upload another mapping", key="mapping_another_btn", use_container_width=True):
        _reset_mapping_state()
        st.rerun()


def _render_mapping_flow() -> None:
    step = st.session_state.mapping_step
    if step == "config":
        _render_mapping_config_step()
    elif step == "preview":
        _render_mapping_preview_step()
    elif step == "uploading":
        _render_mapping_uploading_step()
    elif step == "success":
        _render_mapping_success_step()
    else:
        st.warning(f"Unknown mapping step: {step}. Resetting.")
        _reset_mapping_state()
        st.rerun()


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
    required_missing = [c for c in missing if c not in OPTIONAL_COLUMNS]
    optional_missing = [c for c in missing if c in OPTIONAL_COLUMNS]
    if required_missing:
        result["errors"].append(
            "Missing required columns (or recognizable variants): "
            + ", ".join(required_missing)
        )
        return result
    df = df.rename(columns=rename_map)

    # Backfill any absent optional columns as None so downstream code can
    # treat all canonical columns as present (they just insert as SQL NULL).
    for col in optional_missing:
        df[col] = None
    if optional_missing:
        if len(optional_missing) == 2:
            note = "Ptype and Variant columns not found in file — proceeding without them"
        else:
            note = f"{optional_missing[0]} column not found in file — proceeding without it"
        result["warnings"].append(note)

    null_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_NON_NULL}
    if any(n > 0 for n in null_counts.values()):
        bad = ", ".join(f"{c} ({n} null)" for c, n in null_counts.items() if n > 0)
        result["errors"].append(f"Null values in required fields: {bad}")
        return result

    # Try each known format in order; use the first one that parses every row
    # cleanly. Format order matters: M/D/YY first (original GobbleCube), then
    # YYYY-MM-DD (new Report Builder), then DD-MM-YYYY (Excel-corrupted).
    _DATE_FORMATS = {
        "%m/%d/%y": "M/D/YY",
        "%Y-%m-%d": "YYYY-MM-DD",
        "%d-%m-%Y": "DD-MM-YYYY",
    }
    parsed = None
    for fmt in _DATE_FORMATS:
        candidate = pd.to_datetime(df["Date"], format=fmt, errors="coerce")
        if candidate.notna().all():
            parsed = candidate
            break
    if parsed is None:
        examples = df["Date"].astype(str).head(5).tolist()
        result["errors"].append(
            f"Could not parse Date column. Tried formats: "
            f"{', '.join(_DATE_FORMATS.values())}. "
            f"First 5 values from your file: {examples}"
        )
        return result
    df["Date"] = parsed.dt.strftime("%Y-%m-%d")

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


def _render_data_flow() -> None:
    """Tab 1 — existing GobbleCube data upload flow, untouched."""
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


# ── public entrypoint ────────────────────────────────────────────────────────
def render_upload() -> None:
    _init_state()

    if not st.session_state.upload_authed:
        _render_password_gate()
        return

    _render_authed_header()

    # Freshness banner sits ABOVE the tabs so it's visible on both tabs.
    # Content built by db.format_freshness_markdown() — same source of truth
    # as the dashboard sidebar's freshness box (identical content + formatting).
    st.info(db.format_freshness_markdown())

    tab_data, tab_mapping = st.tabs(["📤 Upload Data", "🏷️ Update SKU Mapping"])
    with tab_data:
        _render_data_flow()
    with tab_mapping:
        _render_mapping_flow()
