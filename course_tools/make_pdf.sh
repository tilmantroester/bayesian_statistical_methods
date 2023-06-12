#!/bin/bash
jupyter nbconvert $1 \
--to webpdf \
--output-dir pdf \
--TemplateExporter.exclude_input_prompt=True \
--TemplateExporter.exclude_output_prompt=True \
--WebPDFExporter.embed_images=True \
--TagRemovePreprocessor.remove_cell_tags="remove-cell" \
--TagRemovePreprocessor.remove_all_outputs_tags="remove-output" \
--TagRemovePreprocessor.remove_input_tags="remove-input"

# MathJax 3 has weird artefacts with webpdf
# --WebPDFExporter.mathjax_url=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js \
# --TemplateExporter.extra_template_basedirs=course_tools/nbconvert_templates \
