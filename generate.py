from pathlib import Path
import re
import subprocess
import unicodedata

ROOT = Path(__file__).parent

INDEX_MD = ROOT / "index.md"
ABSTRACT_MD = ROOT / "abstract.md"
APPENDIX_MD = ROOT / "appendix.md"
OUT = ROOT / "article.tex"  

PREAMBLE = r"""
\documentclass[12pt]{article} 
\usepackage[utf8]{inputenc} 
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb} 
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{cite}
\usepackage[font=small,labelfont=bf]{caption}

\usepackage{graphicx}
\usepackage[
  font=small,
  labelfont=bf, 
  margin=0.5cm
]{caption}
 

\setlength{\textfloatsep}{1.2em}
\setlength{\floatsep}{1em}
\setlength{\intextsep}{1em}

\geometry{a4paper, top=3.5cm, bottom=3.5cm, left=3.5cm, right=3.5cm} 
\setlength{\parindent}{0pt}
\setlength{\parskip}{1em}

\title{Canonical Machines}
\author{Eric Hermosis \\ \texttt{eric.hermosis@gmail.com}}
\date{\today}

\begin{document}
\maketitle
"""

POSTAMBLE = r"""
\newpage
\bibliographystyle{unsrt}
\bibliography{references}
\end{document}
"""

# ----------------- Utilities -----------------
def sanitize_unicode(text: str) -> str:
    return text.replace("\u200B", "")

def sanitize_latex(s: str) -> str:
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, rep in replacements.items():
        s = s.replace(char, rep)
    return s

def sanitize_label(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = re.sub(r'[^0-9a-zA-Z_-]', '', s)
    return s

def strip_title(text: str) -> str:
    return re.sub(r"^\s*#\s+.*\n+", "", text)

def strip_citation_section(text: str) -> str:
    return re.sub(r"\n##\s+Citation[\s\S]*$", "", text)

# ----------------- Markdown → LaTeX -----------------

def convert_math(text: str):
    equations = []
    def repl(m):
        eq = m.group(1).strip()
        token = f"@@EQ{len(equations)}@@"
        equations.append(eq)
        return token
    text = re.sub(r"\$\$(.*?)\$\$", repl, text, flags=re.S)
    return text, equations

def restore_math(text: str, equations):
    for i, eq in enumerate(equations):
        text = text.replace(f"@@EQ{i}@@", "\\begin{equation}\n" + eq + "\n\\end{equation}")
    return text

def convert_sections(text: str) -> str:
    levels = [("####", r"\subsubsection*{"), ("###", r"\subsection*{"), ("##", r"\section*{"), ("#", r"\section*{")]
    for markdown, latex in levels:
        text = re.sub(
            rf"^\s*{re.escape(markdown)}\s+(.*)$",
            lambda m: latex + sanitize_latex(m.group(1)) + "}",
            text, flags=re.M)
    return text


def convert_lists(text: str) -> str:
    lines = text.splitlines()
    out = []
    in_list = False

    for line in lines:
        if re.match(r"^\s*-\s+", line):
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"  \item " + line.lstrip("- ").strip())

        elif in_list and line.strip() == "":
            continue

        else:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(line)

    if in_list:
        out.append(r"\end{itemize}")

    return "\n".join(out)


def convert_inline_formatting(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\\emph{\1}", text)
    text = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\\emph{\1}", text)
    text = re.sub(r"`(.*?)`", r"\\texttt{\1}", text)
    return text

def convert_citations(text: str) -> str:
    return re.sub(r"\[@([^\]]+)\]", lambda m: m.group(0) if m.group(1).startswith("fig:") else f"\\cite{{{m.group(1)}}}", text)

def convert_fig_refs(text: str) -> str:
    return re.sub(r"\[@fig:([^\]]+)\]", r"\\ref{fig:\1}", text)

# ----------- CHANGED IMAGE PARSER ------------

def convert_images(text: str):
    lines = text.splitlines()
    out = []
    i = 0

    img_pattern = re.compile(r"!\[\{#([^\}]+)\}\]\((.*?)\)")

    while i < len(lines):
        line = lines[i].strip()
        m = img_pattern.match(line)

        if m:
            label = m.group(1)
            path = m.group(2)

            caption = ""
            j = i + 1

            while j < len(lines):
                if lines[j].strip():
                    caption = lines[j].strip()
                    break
                j += 1

            out.append(r"\begin{figure}[h]")
            out.append(r"  \centering")
            out.append(rf"  \includegraphics[width=1.0\textwidth]{{{path}}}")
            out.append(rf"  \caption{{{caption}}}")
            out.append(rf"  \label{{{label}}}")
            out.append(r"\end{figure}")

            i = j + 1
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)

# ---------------------------------------------

def convert_tables(text: str) -> str:
    def md_table_to_latex(table_md, caption="", label=""):
        lines = table_md.strip().splitlines()
        if len(lines) < 2:
            return table_md

        headers = [h.strip() for h in lines[0].split("|")[1:-1]]
        col_format = " | ".join(["l"] * len(headers))

        latex = [
            r"\begin{table}[h]",
            r"  \centering",
            f"  \\begin{{tabular}}{{{col_format}}}",
            "  \\hline"
        ]

        latex.append(" & ".join(headers) + " \\\\ \\hline")

        for row in lines[2:]:
            cells = [c.strip() for c in row.split("|")[1:-1]]

            processed_cells = []
            for cell in cells:
                if cell.startswith("$") and cell.endswith("$"):
                    processed_cells.append(cell)
                else:
                    processed_cells.append(sanitize_latex(cell))

            latex.append("  " + " & ".join(processed_cells) + " \\\\")

        latex.append("  \\hline\n  \\end{tabular}")

        if caption:
            latex.append(f"  \\caption{{{caption}}}")

        if label:
            latex.append(f"  \\label{{table:{label}}}")

        latex.append(r"\end{table}")

        return "\n".join(latex)

    def table_repl(match):
        table_md = match.group(1)

        caption_label_match = re.search(r"<!--table:([^\s]+)\s*-->\s*(.*)", table_md, re.S)

        caption, label = "", ""

        if caption_label_match:
            label = sanitize_label(caption_label_match.group(1))
            caption = caption_label_match.group(2).strip()

            table_md = re.sub(r"<!--table:[^\s]+-->.*", "", table_md, flags=re.S)

        return md_table_to_latex(table_md, caption=caption, label=label)

    table_pattern = r"((?:\|.*\|\n)+(?:<!--table:[^\n]+-->.*)?)"

    return re.sub(table_pattern, table_repl, text)

def convert_table_refs(text: str) -> str:
    return re.sub(r"\[@tab:([^\]]+)\]", r"\\ref{table:\1}", text)

def markdown_to_latex(text: str) -> str:
    text = convert_fig_refs(text)
    text = convert_table_refs(text)
    text = convert_citations(text)
    text = convert_inline_formatting(text)
    text, equations = convert_math(text)
    text = convert_sections(text)
    text = convert_lists(text)
    text = convert_tables(text)
    text = convert_images(text)
    text = restore_math(text, equations)
    return text

def convert_appendix(text: str) -> str:
    lines, out = text.splitlines(), []
    first_section = True

    for line in lines:

        m_sec = re.match(r"^##\s+(.*)$", line)

        if m_sec and first_section:
            out.append(rf"\section*{{Appendix: {sanitize_latex(m_sec.group(1))}}}")
            first_section = False
            continue

        m_sub = re.match(r"^###\s+(.*)$", line)

        if m_sub:
            out.append(rf"\subsection*{{{sanitize_latex(m_sub.group(1))}}}")
            continue

        out.append(line)

    return markdown_to_latex("\n".join(out))

# ----------------- Compile -----------------

def compile():

    abstract = sanitize_unicode(ABSTRACT_MD.read_text(encoding="utf-8").strip())

    body = sanitize_unicode(INDEX_MD.read_text(encoding="utf-8"))

    body = strip_title(body)
    body = strip_citation_section(body)

    abstract_tex = markdown_to_latex(abstract)
    body_tex = markdown_to_latex(body)

    appendix_tex = ""

    if APPENDIX_MD.exists():

        appendix_md = APPENDIX_MD.read_text(encoding="utf-8")

        appendix_tex = "\n\\newpage\n\\appendix\n\n" + convert_appendix(appendix_md)

    latex = (
        PREAMBLE
        + "\n\\begin{abstract}\n"
        + abstract_tex
        + "\n\\end{abstract}\n\n"
        + body_tex
        + appendix_tex
        + POSTAMBLE
    )

    OUT.write_text(latex, encoding="utf-8")

    print(f"Generated {OUT}")

    try:

        cmds = [
            ["pdflatex", str(OUT)],
            ["bibtex", OUT.stem],
            ["pdflatex", str(OUT)],
            ["pdflatex", str(OUT)],
        ]

        for cmd in cmds:
            subprocess.run(cmd, cwd=ROOT, check=True)

        print(f"Generated PDF: {OUT.with_suffix('.pdf')}")

    except subprocess.CalledProcessError as e:

        print(f"Error during compilation: {e}")


if __name__ == "__main__":
    compile()