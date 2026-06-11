from pathlib import Path
from html import escape
import subprocess
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[1]


def markdown_to_html(text: str, title: str, deluxe: bool = False) -> str:
    lines = text.splitlines()
    parts: list[str] = []
    in_ul = False
    in_table = False
    in_code = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            parts.append("</ul>")
            in_ul = False

    def close_table() -> None:
        nonlocal in_table, table_rows
        if in_table and table_rows:
            header = table_rows[0]
            body = table_rows[1:]
            parts.append('<table class="grid">')
            parts.append("<thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in header) + "</tr></thead>")
            if body:
                parts.append("<tbody>")
                for row in body:
                    parts.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
                parts.append("</tbody>")
            parts.append("</table>")
        in_table = False
        table_rows = []

    def close_code() -> None:
        nonlocal in_code, code_lines
        if in_code:
            parts.append("<pre><code>" + escape("\n".join(code_lines)) + "</code></pre>")
            in_code = False
            code_lines = []

    def is_table_separator(line: str) -> bool:
        compact = line.replace("|", "").replace(" ", "")
        return compact != "" and set(compact) <= {"-", ":"}

    def row_to_cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            close_ul()
            close_table()
            if in_code:
                close_code()
            else:
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if "|" in stripped:
            close_ul()
            if not in_table:
                in_table = True
                table_rows = []
            if not is_table_separator(stripped):
                table_rows.append(row_to_cells(stripped))
            continue
        else:
            close_table()

        if stripped == "":
            close_ul()
            parts.append('<p class="spacer"></p>')
            continue

        if stripped.startswith("### "):
            close_ul()
            parts.append(f"<h3>{escape(stripped[4:])}</h3>")
            continue

        if stripped.startswith("## "):
            close_ul()
            parts.append(f"<h2>{escape(stripped[3:])}</h2>")
            continue

        if stripped.startswith("# "):
            close_ul()
            parts.append(f"<h1>{escape(stripped[2:])}</h1>")
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{escape(stripped[2:])}</li>")
            continue

        close_ul()
        parts.append(f"<p>{escape(stripped)}</p>")

    close_ul()
    close_table()
    close_code()

    body = "\n".join(parts)

    cover = ""
    if deluxe:
        cover = f"""
<section class=\"cover\">
  <div class=\"cover-box\">
    <div class=\"cover-kicker\">Documento Académico de Sustentación</div>
    <h1 class=\"cover-title\">GPT-4o-mini aplicado al proyecto Enfócate Más</h1>
    <p class=\"cover-subtitle\">Análisis técnico-académico del modelo de Inteligencia Artificial que da solución al chatbot administrativo 24/7 para más de 3.000 familias</p>
    <div class=\"cover-meta\">
      <p><strong>Proyecto:</strong> Chatbot Administrativo Enfócate Más</p>
      <p><strong>Institución analizada:</strong> Enfócate Más</p>
      <p><strong>Tecnologías del sistema:</strong> LangGraph, RAG, GPT-4o-mini, FastAPI, ChromaDB, OpenAI Embeddings, Twilio WhatsApp</p>
      <p><strong>Tipo de documento:</strong> Versión final de lujo para sustentación universitaria</p>
      <p><strong>Fecha de generación:</strong> 2026-06-01</p>
    </div>
  </div>
</section>
<div class=\"pagebreak\"></div>
<section class=\"toc\">
  <h2>Guía de Lectura</h2>
  <p>Este documento presenta el problema institucional de Enfócate Más, la justificación de GPT-4o-mini como modelo elegido, la arquitectura Transformer que lo sustenta, el detalle matemático de atención y generación autoregresiva, su integración real con RAG y LangGraph, y las métricas obtenidas dentro del proyecto.</p>
  <p>La versión actual fue diseñada como soporte formal para exposición académica y defensa ante jurado.</p>
</section>
<div class=\"pagebreak\"></div>
"""

    return f"""<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Helvetica, Arial, sans-serif; margin: 40px; color: #1f2937; line-height: 1.5; }}
    h1 {{ text-align: center; color: #0f172a; font-size: 28px; margin-bottom: 18px; }}
    h2 {{ color: #0b3b8c; font-size: 20px; border-bottom: 2px solid #dbeafe; padding-bottom: 4px; margin-top: 26px; }}
    h3 {{ color: #14532d; font-size: 15px; margin-top: 18px; }}
    p {{ margin: 7px 0; text-align: justify; }}
    ul {{ margin: 8px 0 12px 22px; }}
    li {{ margin: 4px 0; }}
    pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 6px; white-space: pre-wrap; font-size: 10px; }}
    code {{ font-family: Menlo, Consolas, monospace; }}
    table.grid {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px 0; font-size: 10.5px; }}
    table.grid th, table.grid td {{ border: 1px solid #cbd5e1; padding: 6px 8px; vertical-align: top; }}
    table.grid th {{ background: #e0f2fe; text-align: left; }}
    table.grid tr:nth-child(even) td {{ background: #f8fafc; }}
    .spacer {{ margin: 8px 0; }}
    .pagebreak {{ page-break-before: always; }}
    .cover {{ min-height: 90vh; display: flex; align-items: center; justify-content: center; }}
    .cover-box {{ border: 2px solid #1d4ed8; padding: 36px; border-radius: 12px; background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%); }}
    .cover-kicker {{ text-transform: uppercase; letter-spacing: 1.2px; font-size: 12px; color: #1d4ed8; text-align: center; margin-bottom: 18px; font-weight: bold; }}
    .cover-title {{ font-size: 30px; line-height: 1.2; margin-bottom: 16px; }}
    .cover-subtitle {{ text-align: center; font-size: 14px; color: #334155; margin-bottom: 24px; }}
    .cover-meta {{ border-top: 1px solid #cbd5e1; padding-top: 16px; font-size: 12px; }}
    .cover-meta p {{ text-align: left; margin: 6px 0; }}
    .toc p {{ font-size: 12px; }}
  </style>
</head>
<body>
{cover}
{body}
</body>
</html>
"""


def export(md_path: Path, output_path: Path, deluxe: bool = False) -> None:
    content = md_path.read_text(encoding="utf-8")
    html = markdown_to_html(content, md_path.stem, deluxe=deluxe)
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "document.html"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run(
            ["textutil", "-convert", "docx", str(html_path), "-output", str(output_path)],
            check=True,
        )


def main() -> None:
    md_path = ROOT / "docs" / "GPT4o-mini_EnfocateMas_Academico.md"
    deluxe = "--deluxe" in sys.argv
    if deluxe:
        output_path = ROOT / "GPT4o-mini_EnfocateMas_Academico_Deluxe.docx"
    else:
        output_path = ROOT / "GPT4o-mini_EnfocateMas_Academico.docx"
    export(md_path, output_path, deluxe=deluxe)
    print(output_path)


if __name__ == "__main__":
    main()
