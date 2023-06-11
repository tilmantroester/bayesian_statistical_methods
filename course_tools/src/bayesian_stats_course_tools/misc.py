from pathlib import Path
from IPython.display import Markdown, display


def load_tex_defs(tex_defs_file=None):
    if tex_defs_file is None:
        tex_defs_file = Path(__file__).parent / "tex_defs.md"
    with open(tex_defs_file, "r") as f:
        defs = f.read()
    
    defs = Markdown(defs)
    return defs


def display_markdown_and_setup_tex(markdown=None, tex_defs_file=None):
    markdown = Markdown(markdown)
    defs = load_tex_defs(tex_defs_file)

    display(markdown)
    display(defs)
