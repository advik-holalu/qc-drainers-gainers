# GO DESi MS Tracker

Weekly Market Share analysis dashboard for GO DESi across Blinkit, Zepto, and Instamart.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL printed in the terminal, upload your Market Share Excel file, then pick a **Chosen Week** and a **Compare Week** to see Drainers and Gainers.

## Notes

- Only the **Master** sheet is read from the uploaded Excel file.
- The two tables are restricted to the **top 80% of current-week Offtake (MRP)** before computing deltas.
- Percentage-point metrics (MS, OSA, SOV) are displayed as-is with a `%` suffix. If your source data stores these as decimals (e.g. `0.05` for 5 %), values will appear small — let me know and I'll add a ×100 multiplier.
- `—` in a Prev or Current cell means that combination didn't exist in that week; its value is treated as 0 for delta calculations.
