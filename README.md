# TriCab RFQ Backend

FastAPI service implementing the three-stage RFQ pipeline:
1. **Claude** — normalises free-text RFQ into a structured spec
2. **T5** — predicts product code from structured spec  
3. **tpro_ext** — validates code exists; assigns confidence tier

## Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars
export ANTHROPIC_API_KEY=sk-ant-...
export MODEL_DIR=./model          # path to your T5 model files
export DATA_DIR=./data            # path to CSV data files

# Download model from Google Drive (first time only)
export MODEL_DRIVE_URL="https://drive.google.com/drive/folders/..."
python download_model.py

# Start server
uvicorn main:app --reload --port 8000
```

## Endpoints

**POST /generate**
```json
{ "text": "3C 2.5mm flex control PVC black" }
```
Returns product code, confidence, segment breakdown.

**GET /health**
Returns status and loaded data counts.

## Deploy to Render

1. Push this folder to a GitHub repo
2. Create a new Web Service on Render pointing to the repo
3. Set environment variables in Render dashboard:
   - `ANTHROPIC_API_KEY`
   - `MODEL_DRIVE_URL` — Google Drive folder share link for T5 model
4. Deploy

Render will run `pip install -r requirements.txt && python download_model.py`
then start the server automatically.

## Data files (data/)

| File | Description |
|---|---|
| `tpro_ext.csv` | 18,994 product codes, fully exploded into segments |
| `cable_specs.csv` | 9,817 variant rows for technical validation |
| `descript.csv` | Style name lookup |
| `style_index.json` | Pre-built index of valid options per style (generated at startup if missing) |

## Confidence tiers

| Tier | Meaning |
|---|---|
| HIGH | Exact match found in tpro_ext |
| MEDIUM | Style + nc + xa valid in cable_specs (custom/non-standard variant) |
| LOW | Style recognised but nc/xa outside known range (engineering review needed) |
| NONE | Style not recognised |

## Updating the data

When the TriCab database changes, re-export the three tables and replace the CSV files.
The style_index.json is built automatically on startup from tpro_ext.csv.
