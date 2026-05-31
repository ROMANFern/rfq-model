"""
TriCab RFQ Backend
==================
Three-stage pipeline:
  1. Claude  — normalise free-text RFQ into structured spec
  2. T5      — predict product code from structured spec
  3. tpro_ext — validate code exists; assign confidence tier
"""

import os, json, re, logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import torch
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR      = os.getenv("MODEL_DIR", "./model")
DATA_DIR       = os.getenv("DATA_DIR",  "./data")
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 32

# ── Global state (loaded once at startup) ─────────────────────────────────────

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading data and model...")

    # Data tables
    tpro  = pd.read_csv(f"{DATA_DIR}/tpro_ext.csv")
    specs = pd.read_csv(f"{DATA_DIR}/cable_specs.csv")

    # Product code lookup set — O(1) existence checks
    state["product_codes"] = set(tpro["product_code"].values)

    # Style index — valid options per style
    with open(f"{DATA_DIR}/style_index.json") as f:
        state["style_index"] = json.load(f)

    # Style descriptions
    desc = pd.read_csv(f"{DATA_DIR}/descript.csv")
    state["desc_map"] = dict(zip(desc["desc_code"], desc["desc_name"]))

    # cable_specs for fuzzy validation (style + nc + xa)
    state["specs"] = specs

    # T5 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading T5 from {MODEL_DIR} on {device.upper()}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    state["tokenizer"] = tokenizer
    state["model"]     = model
    state["device"]    = device

    # Anthropic client
    state["anthropic"] = Anthropic()

    log.info(f"Ready — {len(state['product_codes'])} product codes loaded, "
             f"{len(state['style_index'])} styles indexed")
    yield
    state.clear()

app = FastAPI(title="TriCab RFQ API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    text: str

class SegmentDetail(BaseModel):
    position:   int
    key:        str
    label:      str
    value:      str
    display:    str
    confidence: str
    note:       str

class GenerateResponse(BaseModel):
    product_code:       str
    overall_confidence: str
    variant_exists:     bool
    matched_style:      str
    style_description:  str
    segments:           list[SegmentDetail]
    notes:              list[str]
    missing:            list[str]

# ── Stage 1: Claude normalisation ─────────────────────────────────────────────

STYLE_INDEX_SUMMARY = json.dumps(
    {k: {"desc": v["description"], "conductors": v["conductors"],
         "core_types": v["core_types"], "armour_chars": v["armour_chars"]}
     for k, v in {}. items()},  # filled at runtime below
)

def build_system_prompt(style_index: dict, desc_map: dict) -> str:
    # Compact style list for prompt — style: description
    style_lines = "\n".join(
        f"  {k}: {v['description']}"
        for k, v in sorted(style_index.items())
        if v["description"]
    )

    return f"""You are a cable product code expert for TriCab, an Australian cable manufacturer.
Convert a customer RFQ description into a structured JSON spec matching TriCab's product coding system.

## Product code format
STYLE[SHEATH]-CONFIG_CORE ARMOUR CONDUCTOR/NO_CORE CORE_TYPE CROSS_SECTION COLOUR

Where:
- STYLE: 2-3 char series code (e.g. KL, BV, C1, LS)
- SHEATH: optional 1 char (e.g. A, C, M) appended to style when present
- CONFIG_CORE: 2 chars — core colour configuration (e.g. PA, CA, XX, PV, DA)
- ARMOUR: 1 char — X=none, T=SWA, S=screen/braid, G=galvanised, B=bronze, N=armoured
- CONDUCTOR: N=plain copper, D=tinned copper, A=aluminium, K=special
- NO_CORE: integer number of cores/pairs/triads
- CORE_TYPE: C=conductor, P=pair, T=triad, A=armoured core
- CROSS_SECTION: mm² (e.g. 1.5, 2.5, 35, 120) — use .75 not 0.75 for sub-1 values
- COLOUR: 2-char jacket colour — BK=Black OR=Orange GY=Grey WH=White BL=Blue
  RE=Red YE=Yellow EA=Green/Yellow(Earth) BR=Brown GR=Green PU=Purple DB=Dark Blue

## CONFIG_CORE first character meaning
P = Power (single core / building wire)
C = Control / multi-core
D = DC / marine single core
M = Marine multi-core
N = NZ-specific
X = No config / single insulated (LS switchboard)

## Available styles
{style_lines}

## Rules
- Pick the single best-matching style
- If jacket colour not stated: default BK, mark MEDIUM confidence
- If config_core not stated: use PA for power/single, CA for multi-core control,
  DA for marine single, XX for LS switchboard — mark MEDIUM
- overall_confidence = lowest individual segment confidence
- variant_exists: leave as null — the backend validates this
- Confidence: HIGH=exact known value, MEDIUM=inferred/assumed, LOW=uncertain, NONE=invalid

Return ONLY valid JSON, no markdown:
{{
  "matched_style": "KL",
  "product_code": "KL-PAXN/1C2.5OR",
  "overall_confidence": "HIGH",
  "variant_exists": null,
  "segments": [
    {{"position":1,"key":"style_code","label":"Style","value":"KL",
      "display":"Flexible Rubber X-HF-110 0.6/1kV 110C","confidence":"HIGH","note":""}},
    {{"position":2,"key":"config_core","label":"Core Colour Config","value":"PA",
      "display":"Standard power colour config","confidence":"HIGH","note":""}},
    {{"position":3,"key":"armour","label":"Armour / Screen","value":"X",
      "display":"None","confidence":"HIGH","note":""}},
    {{"position":4,"key":"conductor","label":"Conductor","value":"N",
      "display":"Plain annealed copper","confidence":"HIGH","note":""}},
    {{"position":5,"key":"no_core","label":"No. of Cores","value":"1",
      "display":"1 conductor","confidence":"HIGH","note":""}},
    {{"position":6,"key":"core_type","label":"Core Type","value":"C",
      "display":"Conductor","confidence":"HIGH","note":""}},
    {{"position":7,"key":"cross_section","label":"Cross Section","value":"2.5",
      "display":"2.5 mm²","confidence":"HIGH","note":""}},
    {{"position":8,"key":"colour","label":"Jacket Colour","value":"OR",
      "display":"Orange","confidence":"HIGH","note":""}}
  ],
  "notes": [],
  "missing": []
}}"""


def stage1_normalise(rfq_text: str) -> dict:
    prompt = build_system_prompt(state["style_index"], state["desc_map"])
    resp = state["anthropic"].messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=prompt,
        messages=[{"role": "user", "content": rfq_text}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


# ── Stage 2: T5 prediction ────────────────────────────────────────────────────

def stage2_predict(structured_text: str) -> str:
    tok   = state["tokenizer"]
    model = state["model"]
    dev   = state["device"]

    inp = tok(
        "generate product code: " + structured_text,
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=inp.input_ids.to(dev),
            attention_mask=inp.attention_mask.to(dev),
            max_length=MAX_TARGET_LEN,
            num_beams=5,
            early_stopping=True,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()


# ── Stage 3: tpro_ext validation ──────────────────────────────────────────────

def stage3_validate(claude_result: dict, t5_code: str) -> dict:
    """
    Determine final product code and confidence tier:
      HIGH   — exact match in tpro_ext
      MEDIUM — style + nc + xa exist in cable_specs (valid custom config)
      LOW    — style exists but nc/xa outside known range
      NONE   — style not recognised
    """
    product_codes = state["product_codes"]
    style_index   = state["style_index"]
    specs         = state["specs"]

    # Prefer T5 prediction if it exists in tpro_ext, else fall back to Claude's
    final_code   = None
    variant_exists = False

    for candidate in [t5_code, claude_result.get("product_code", "")]:
        if candidate and candidate in product_codes:
            final_code     = candidate
            variant_exists = True
            break

    if not final_code:
        # Neither matched — use Claude's code as the best available
        final_code = claude_result.get("product_code", t5_code or "")

    # Extract style from final code
    matched_style = final_code.split("-")[0] if "-" in final_code else ""
    # Strip sheath suffix to get base style for index lookup
    base_style = matched_style[:2] if len(matched_style) > 2 else matched_style

    style_desc = state["desc_map"].get(matched_style, state["desc_map"].get(base_style, ""))

    # Determine confidence
    if variant_exists:
        confidence = "HIGH"
    else:
        # Check cable_specs for style + nc + xa
        try:
            # Parse nc and xa from the code
            after_dash = final_code.split("-")[1] if "-" in final_code else ""
            after_slash = after_dash.split("/")[1] if "/" in after_dash else ""
            # Extract digits at start for nc, then letter, then rest for xa
            nc_match = re.match(r"(\d+)([A-Za-z])([\d.]+)", after_slash)
            if nc_match:
                nc = int(nc_match.group(1))
                xa = nc_match.group(3)
                # Normalise xa: .75 → 0.75 for specs lookup
                if xa.startswith("."):
                    xa = "0" + xa

                style_specs = specs[
                    (specs["cable_style"].str.startswith(base_style)) &
                    (specs["nc"] == nc) &
                    (specs["xa"] == xa)
                ]
                if len(style_specs) > 0:
                    confidence = "MEDIUM"
                elif base_style in style_index:
                    confidence = "LOW"
                else:
                    confidence = "NONE"
            else:
                confidence = "LOW" if base_style in style_index else "NONE"
        except Exception:
            confidence = "LOW"

    # Override Claude's overall_confidence with validated value
    result = dict(claude_result)
    result["product_code"]       = final_code
    result["overall_confidence"] = confidence
    result["variant_exists"]     = variant_exists
    result["matched_style"]      = matched_style
    result["style_description"]  = style_desc

    # Note if T5 and Claude disagreed
    claude_code = claude_result.get("product_code", "")
    if t5_code and claude_code and t5_code != claude_code:
        result.setdefault("notes", [])
        result["notes"].append(
            f"T5 predicted {t5_code!r}; Claude suggested {claude_code!r}. "
            f"{'T5 matched tpro_ext.' if t5_code in product_codes else 'Claude code used as neither matched.'}"
        )

    return result


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        # Stage 1 — Claude normalisation
        log.info(f"Stage 1 — normalising: {req.text[:80]}")
        claude_result = stage1_normalise(req.text)

        # Stage 2 — T5 prediction (use Claude's normalised text as input)
        # Build a structured description string from Claude's segments
        seg_map = {s["key"]: s["value"] for s in claude_result.get("segments", [])}
        structured = (
            f"{seg_map.get('no_core','?')}{seg_map.get('core_type','C')} "
            f"{seg_map.get('cross_section','?')}mm2 "
            f"{seg_map.get('conductor','N')} "
            f"{seg_map.get('style_code','')} "
            f"{seg_map.get('armour','X')} "
            f"{seg_map.get('colour','BK')}"
        )
        log.info(f"Stage 2 — T5 input: {structured}")
        t5_code = stage2_predict(structured)
        log.info(f"Stage 2 — T5 output: {t5_code}")

        # Stage 3 — validation
        final = stage3_validate(claude_result, t5_code)

        return GenerateResponse(
            product_code=final.get("product_code", ""),
            overall_confidence=final.get("overall_confidence", "NONE"),
            variant_exists=final.get("variant_exists", False),
            matched_style=final.get("matched_style", ""),
            style_description=final.get("style_description", ""),
            segments=[SegmentDetail(**s) for s in final.get("segments", [])],
            notes=final.get("notes", []),
            missing=final.get("missing", []),
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Claude returned invalid JSON: {e}")
    except Exception as e:
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "product_codes": len(state.get("product_codes", [])),
        "styles": len(state.get("style_index", {})),
        "model_loaded": "model" in state,
    }
