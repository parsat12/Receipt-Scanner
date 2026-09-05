# Receipt Scanner

Live camera receipt scanner that detects a receipt in frame, captures a cleaned image, sends it to AI for structured extraction, and appends the results to an Excel workbook.

## Output Format

Rows are written to `receipt_log.xlsx` with the columns from the provided formatting reference:

| Date | Vendor | Amount | CC # | COA | Location |
| --- | --- | ---: | --- | --- | --- |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your_api_key_here"
```

## Run

```powershell
python scan.py
```

Useful options:

```powershell
python scan.py --dry-run
python scan.py --camera 1
python scan.py --output receipts.xlsx
python scan.py --min-sharpness 180
python scan.py --backend dshow --width 0 --height 0
```

Controls while the camera window is open:

- `q`: quit
- `c`: force capture when a receipt outline is visible

The camera overlay shows `Focus: score/min OK` when the receipt is sharp enough for automatic capture. If images are still blurry, hold the receipt still, improve lighting, move the receipt farther from the webcam, or raise/lower `--min-sharpness` depending on what score you see on screen.

If OpenCV prints MSMF camera warnings or stops reading frames, run:

```powershell
python scan.py --backend dshow --width 0 --height 0
```

If the receipt is visible but not outlined, move it closer or lower the minimum area:

```powershell
python scan.py --min-area-ratio 0.03
```

After each AI extraction, the terminal shows the proposed row. Press Enter to accept it, type `s` to skip it, or paste corrected JSON to write the corrected row.

```json
{"date":"2025-01-02","vendor":"Costco","amount":65.15,"cc_number":"5121","coa":"auto","location":"Fullerton"}
```

Captured receipt images are saved in `captures/`. Raw AI responses are saved in `ai_logs/`. The Excel workbook is created automatically if it does not already exist.

To append without review:

```powershell
python scan.py --no-review
```
