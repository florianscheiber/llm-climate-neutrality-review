# llm-climate-neutrality-review

## repo layout

```
scripts/     all pipeline scripts (run them from the repo root)
prompts/     the LLM prompt text files the scripts load
docs/        method write-ups (pipeline.html is the overview)
run-stage3.py   one entry point that chains the stage-3 scripts in order
```

Run every script from the repo root (`python3 scripts/<name>.py ...`), not from
inside `scripts/` -- the data paths (`inputs/`, `outputs/`, `intermediate/`) are
relative to the working directory.

## stage1: abstract screening
1. create key at openrouter.ai (or contact Florian Scheiber for key)
2. put following line in terminal in project folder: 
LLM_ROUTER_API_KEY=<openrouter-key> python3 scripts/stage1-abstract-screening.py inputs/reference_list_papers/reference_list_paper_stage1.csv outputs-screening/outputs-abstract-screening/<name of output csv here> --provider router --model <modelname>
--> abstracts are screened
NB: available <modelname> can be found in openrouter_available_models.rtf

## stage1: compare abstract results with Cadima
1. open scripts/stage1-compare_abstract_results_cadima_llm.py
2. change LLM_RUN_OUTPUT and CADIMA_STAGE according to your analysis
3. comparision_categories.json and overlap_info.json are created as well as overview output in Console

## stage 2: fulltext screening
1. use key (see abstract screening) for 
LLM_ROUTER_API_KEY=<dein_key> LLM_DEBUG=0 python3 scripts/stage2-fulltext-screening.py inputs/fulltext_pdf/<directory of pdf set here> outputs-screening/outputs-fulltext-screening/<name of output csv here> output-1-fulltext-test1.csv --provider router --model <modelname>

LLM_DEBUG=1: for testing (puts out first n characters of paper)

## stage 3 extraction (The terminal commands to start each pipeline are documented at the top of the respective script file.)

**Run the whole stage-3c pipeline with one script:**

```
export LLM_ROUTER_API_KEY=<key>
python3 run-stage3.py                    # the whole pipeline: markdown -> 01 ... 09
python3 run-stage3.py --dry-run          # preview the commands only
python3 run-stage3.py --to harmonize     # stop after the harmonized tables (skip 05-09)
python3 run-stage3.py --from sector-balance   # just re-run the 05-09 analyses
python3 run-stage3.py --from merge       # resume from a given step
```

`run-stage3.py` (repo root) chains, in order,
`scripts/pdf-to-markdown-extraction.py` then
`scripts/01-...` through `scripts/10-...`:
`01` data-extraction, `02` candidates-extraction, `03` merge-extractions,
`04` harmonize, then the exploratory analyses `05` sector-balance,
`06` carbon-price, `07` mitigation-contribution, `08` electricity-price,
`09` capex-by-technology, `10` macro-economic-cost (each writes
`outputs/analysis/*.csv`, `figures/*.png` and a `docs/*.html`).
Full workflow overview: [docs/pipeline.html](docs/pipeline.html).

### stage 3a
Stage 3a uses vision-based extraction: the PDFs are rendered as images and the model reads those page images together with the prompt. This was the first version of the extraction pipeline. (`scripts/stage3a-data-extraction.py`)

### stage 3b
After testing, image-based recognition turned out to be not 100% reliable for the structured data we need. Stage 3b therefore uses only PDF text plus machine-readable tables extracted with PyMuPDF. Images are not numerically extracted. Potentially relevant figures and tables that cannot be extracted reliably are written to `candidate.txt` for manual review. (`scripts/stage3b-data-extraction.py`)

### stage 3c
Stage 3c combines both. Steps (numbered by call order in `scripts/`):

1. `pdf-to-markdown-extraction.py` converts each PDF to Markdown (text + reconstructed tables + figure captions) and rasterises the figure regions.
2. `01-stage3c-data-extraction.py` sends the Markdown plus the figure images to the model and extracts `mitigation` / `costs` / `emissions` rows, writing figures/tables it could not read to `<id>-candidates.txt`.
3. `02-stage3c-candidates-extraction.py` re-visits those `<id>-candidates.txt` figures: it locates each in the source PDF by its caption, renders the **full page(s)** at 300 DPI (the first pass only had cropped low-res regions), and re-runs a focused prompt (`prompts/stage3c_candidates_prompt.txt`) passing the rows the first pass already extracted so they are not duplicated. Structurally un-digitisable items (spatial maps, Sankey diagrams, violin plots) are skipped unless `--all`.
4. `03-stage3c-merge-extractions.py` concatenates the per-paper CSVs from both passes and joins Title/Year/DOI.
5. `04-stage3c-harmonize.py` standardises units/currency and writes `netzero-flags.csv`.

```
# candidates second pass, standalone:
python3 scripts/02-stage3c-candidates-extraction.py --dry-run          # locate figures, write plan.csv, no API calls
LLM_ROUTER_API_KEY=<key> python3 scripts/02-stage3c-candidates-extraction.py
LLM_ROUTER_API_KEY=<key> python3 scripts/02-stage3c-candidates-extraction.py --papers 287,436

# fold the second pass into the merged dataset (adds an extraction_pass column):
python3 scripts/03-stage3c-merge-extractions.py outputs/output-stage3c outputs/output-stage3c \
    --extra-input-dir outputs/output-stage3c-candidates
```

## INPUTS AND OUTPUTS
... are generally stored in https://bokuit.sharepoint.com/sites/PRE-ReviewClimateNeutrality
INPUT: General/papers/included-after-stage-3-100: main + pdf supplements of all 100 papers, that were graded eligible by reviewers after stage 3.
OUTPUT: General/llm/output


