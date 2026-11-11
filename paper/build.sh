#!/usr/bin/env bash
# Rebuild every number and table from the raw logs, redraw the plots, then the PDF.
set -euo pipefail
cd "$(dirname "$0")"
PY_DATA="${PY_DATA:-$( [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3 )}"
PY_PLOT="${PY_PLOT:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || ( [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3 ) )}"
"$PY_DATA" scripts/build_data.py > generated/build.log
"$PY_PLOT" scripts/make_figures.py
tectonic -X compile main.tex --keep-logs --keep-intermediates
# ICDE 2027: at most 12 pages, excluding references and the AI-content acknowledgment.
page=$(sed -n 's/^\\newlabel{lastcontent}{{[^}]*}{\([0-9]*\)}.*/\1/p' main.aux | head -1)
echo "content ends on page ${page:-unknown} (limit 12); PDF has $(pdfinfo main.pdf | sed -n 's/^Pages: *//p') pages"
if [ "${page:-99}" -gt 12 ]; then echo "WARNING: content exceeds the 12-page limit" >&2; fi
"$PY_DATA" scripts/page_room.py
# Comparative sentences of the paper that the current data no longer support.
echo "claim checks: $(grep -c '^claim check ok' generated/claim_checks.txt) ok, $(grep -c '^claim check REVIEW' generated/claim_checks.txt) to review"
grep '^claim check REVIEW' generated/claim_checks.txt || true
