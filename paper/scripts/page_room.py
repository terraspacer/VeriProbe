#!/usr/bin/env python3
"""Report how many text lines are free on page 12 before the content would spill to page 13.

ICDE 2027 allows 12 pages excluding the references and the AI-content
acknowledgment.  The content ends where the acknowledgment heading starts; when
that heading is already on page 13, the content ends at the last line of the
right column of page 12.
"""
import re
import subprocess
import sys
from pathlib import Path

pdf = Path(__file__).resolve().parents[1] / "main.pdf"
html = subprocess.run(["pdftotext", "-f", "12", "-l", "12", "-bbox-layout", str(pdf), "-"],
                      capture_output=True, text=True, check=True).stdout
words = [(float(x0), float(y0), float(y1), w) for x0, y0, y1, w in
         re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)">(.*?)</word>', html)]
BOTTOM, LINE = 738.0, 12.0     # text block bottom and line height of IEEEtran in points
heading = [w for w in words if "CKNOWLEDGMENT" in w[3].upper()]
if heading:
    x, y, _, _ = heading[0]
    column = "right" if x >= 300 else "left"
    free = (BOTTOM - y) / LINE + (0 if column == "right" else 55)
    print(f"room on page 12: about {free:.0f} lines (acknowledgment heading in the {column} column at y={y:.0f})")
    sys.exit(0)
right = [w for w in words if w[0] >= 300]
last = max((w[2] for w in right), default=0.0)
free = (BOTTOM - last) / LINE
print(f"room on page 12: about {free:.1f} lines (the acknowledgment starts on page 13; "
      f"the right column of page 12 ends at y={last:.0f} of {BOTTOM:.0f})")
