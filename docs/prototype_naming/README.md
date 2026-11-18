# Auxiliary prototype naming: prompts and records

[中文说明](README_zh-CN.md) · [Entry points](../ENTRY_POINTS.md)

This guide covers the workflow: training reference -> CLIP candidate ranking ->
GPT-4o visual selection -> auxiliary name. The CLIP-ranking step is implemented
in `tools/naming/run_shape_naming_pilot.py`. GPT-4o selection uses the supplied
prompt and is run separately; the CLIP tool does not submit a GPT-4o request.
Run-record templates are provided under [templates/](templates/).

## Procedure

1. Freeze the prototype bank and reference mapping. Retain the representative's
   full training image with the region marked, its RGB crop and its contour.
   If additional cluster members are shown, record their selection rule.
2. Prepare the candidate vocabulary and term sources. Distinguish dataset
   annotations from author-proposed part/geometry phrases.
3. Encode the RGB crop using CLIP's image encoder, and each prompted candidate
   using its text encoder. L2-normalize the embeddings, rank cosine similarities
   and retain the Top-3. Record the vocabulary, text template, checkpoint hash
   and preprocessing. Break ties by vocabulary order. Scores are not calibrated
   probabilities.
4. Fill in [prompts/vlm_selection.txt](prompts/vlm_selection.txt). Supply the
   Top-3 and the three image views to GPT-4o for visual selection, concise
   refinement or abstention. Save the exact prompt and original response.
5. Save the name beside the prototype ID, linked to its reference and records.
   Automatic selection does not require manual annotation. Optional human
   inspection is recorded separately from the automatic output.

The prompt uses observable reference content. It does not use the final class,
contribution sign or prediction change to select a name. A reference label does
not establish whole-cluster semantic purity or test-image localization. Names
do not enter training, prediction or attribution arithmetic.

## Files to retain with each run

Copy [templates/](templates/) into a fresh directory such as
`outputs/prototype_naming/run_001/`. Fill fields using actual run information.
Retain all attempted prototypes, including abstentions.

| File | Format | Content |
|---|---|---|
| `run_metadata.json` | JSON | Actual service, model ID/version if exposed, access time, settings, CLIP details and hashes |
| `reference_manifest.csv` | UTF-8 CSV | Bank/prototype IDs, training image/crop/contour keys, paths, hashes and selection rule |
| `candidate_terms.csv` | UTF-8 CSV | Complete vocabulary, term sources and vocabulary version |
| `clip_rankings.csv` | UTF-8 CSV | Per-reference ranking, similarity, term and view; at least Top-3, preferably the full ranking |
| `prompts/Pxxxx.txt` | UTF-8 text | Exact submitted prompt with placeholders replaced |
| `responses/Pxxxx.json` | JSON | Original API response, or verbatim output exported from the interface used |
| `parsed/Pxxxx.json` | JSON | Parsed selection/abstention and supporting observation |
| `naming_summary.csv` | UTF-8 CSV | Final label, prototype ID, chosen candidate and links to records |

The parsed-output template does not replace the raw response. If the interface
only exposes `GPT-4o`, record that name, the service and actual access date; leave
an unavailable backend snapshot unset. Record parameters only when set or
exposed. Templates are populated when each naming run is executed.

Publish the prompt, vocabulary, ranking/final-output records and metadata with
the code or in a linked artifact archive. Dataset images can remain external;
retain source keys, hashes and retrieval instructions. Keep credentials out of
records. Each final name links to its own run records through the prototype and
reference identifiers.
