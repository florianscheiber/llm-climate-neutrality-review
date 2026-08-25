# llm-climate-neutrality-review


## stage1: abstract screening
1. create key at openrouter.ai (or contact Florian Scheiber for key)
2. put following line in terminal in project folder: 
LLM_ROUTER_API_KEY=<openrouter-key> python3 abstract-screening.py inputs/reference_list_papers/reference_list_paper_stage1.csv outputs-screening/outputs-abstract-screening/<name of output csv here> --provider router --model <modelname>
--> abstracts are screened
NB: available <modelname> can be found in openrouter_available_models.rtf

## stage1: compare abstract results with Cadima
1. open compare_abstract_results_cadima_llm
2. change LLM_RUN_OUTPUT and CADIMA_STAGE according to your analysis
3. comparision_categories.json and overlap_info.json are created as well as overview output in Console

## stage 2: fulltext screening
1. use key (see abstract screening) for 
LLM_ROUTER_API_KEY=<dein_key> LLM_DEBUG=0 python3 fulltext-screening.py inputs/fulltext_pdf/<directory of pdf set here> outputs-screening/outputs-fulltext-screening/<name of output csv here> output-1-fulltext-test1.csv --provider router --model <modelname>

LLM_DEBUG=1: for testing (puts out first n characters of paper)

## stage 3 extraction (The terminal commands to start each pipeline are documented at the top of the respective script file.)

### stage 3a
Stage 3a uses vision-based extraction: the PDFs are rendered as images and the model reads those page images together with the prompt. This was the first version of the extraction pipeline.

### stage 3b
After testing, image-based recognition turned out to be not 100% reliable for the structured data we need. Stage 3b therefore uses only PDF text plus machine-readable tables extracted with PyMuPDF. Images are not numerically extracted. Potentially relevant figures and tables that cannot be extracted reliably are written to `candidate.txt` for manual review.

## INPUTS AND OUTPUTS
... are generally stored in https://bokuit.sharepoint.com/sites/PRE-ReviewClimateNeutrality
INPUT: General/papers/included-after-stage-3-100: main + pdf supplements of all 100 papers, that were graded eligible by reviewers after stage 3.
OUTPUT: General/llm/output


