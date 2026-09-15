"""Sonuç tablolarını ek bağımlılık olmadan Markdown'a yazma."""
from __future__ import annotations

import pandas as pd


def write_markdown_table(df: pd.DataFrame, path, floatfmt: str = "{:.4f}") -> None:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
