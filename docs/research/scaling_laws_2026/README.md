# Scaling-Law / Efficiency Research — Index

This directory holds Aurelius's **derived** scaling-law research notes and the
metadata index for the papers reviewed in the 2026-06 closer pass.

The raw third-party paper text dumps that used to live here
(`<arxiv-id>v<N>.txt`) were **removed from the repository** — they were vendored
arXiv full-text dumps (~37k lines of third-party content) that are re-downloadable
and do not belong in a code repo. Nothing in `src/`, `tests/`, `scripts/`,
`pyproject.toml`, the `Makefile` or CI referenced them.

## What is still here

| File | Role |
|---|---|
| `inventory.md` | Human-readable index: title, arXiv id, size, relevance-keyword hits, abstract snippets |
| `inventory.json` | Machine-readable index (one record per paper: title, arXiv id, abstract) |
| `AURELIUS_SCALING_RESEARCH_SUMMARY.md` | Summary of the pass and its concrete implications |
| `CLOSER_RESEARCH_TO_AURELIUS.md` | Corrected closer synthesis (referenced from several `docs/*.md` correction notices) |
| `ACDT_SCALING_MECHANISMS.md` | Mechanism notes feeding the ACDT spec |
| `aurelius_config_scaling_audit.json` | Config-vs-scaling-law audit output |

## Re-fetching the source papers

Paper ids are in the `arXiv` column of `inventory.md` / the `arxiv` field of
`inventory.json`. Re-fetch on demand into a scratch directory (do not commit):

```bash
mkdir -p /tmp/scaling_papers
for id in $(python3 -c "import json;print(' '.join(r['arxiv'] for r in json.load(open('docs/research/scaling_laws_2026/inventory.json'))))" | tr ' ' '\n' | sort -u); do
  curl -sL "https://arxiv.org/abs/${id}" -o "/tmp/scaling_papers/${id}.abs.html"
done
```

Unique arXiv ids covered by the index (20):

```
1701.06538v1  2602.09234v2  2604.18002v1  2605.29548v1  2606.20785v1
2606.21911v1  2606.22878v1  2606.22932v1  2606.22938v1  2606.23086v1
2606.23595v1  2606.23670v1  2606.24775v1  2606.24983v1  2606.25170v1
2606.25415v1  2606.25494v1  2606.25674v1  2606.25971v1
```

Two sources in the original set were not arXiv preprints and are therefore not
re-fetchable by id:

- **Tapered Language Models** — same work as `2606.23670v1`; the removed file was a
  locally annotated PDF text extraction (`Tapered_Language_Models-with-annotations.txt`).
- **Lilian Weng, "Scaling Laws"** — blog page `lilianweng.github.io/posts/2020-08-31-scaling-laws/`
  (the removed `lilian_weng_scaling_laws.html` / `.txt` were a saved copy of it).

`inventory.json` retains the `txt`/`pdf` fields from the original run pointing at
the now-removed local paths; they are historical provenance, not live paths.
The `Source: <id>.txt:<lines>` citations inside `CLOSER_RESEARCH_TO_AURELIUS.md`
refer to those same removed dumps — re-fetch the paper by id if you need to check
a quoted passage.
