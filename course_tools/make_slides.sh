#!/bin/bash
jupyter nbconvert $1 \
--to slides \
--output-dir slides \
--TemplateExporter.extra_template_basedirs=./nbconvert_templates \
--TemplateExporter.exclude_input_prompt=True \
--TemplateExporter.exclude_output_prompt=True \
--SlidesExporter.mathjax_url=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js \
--SlidesExporter.reveal_number="c" \
--SlidesExporter.reveal_theme="white" \
--SlidesExporter.reveal_scroll=True \
--TagRemovePreprocessor.remove_cell_tags="remove-cell" \
--TagRemovePreprocessor.remove_all_outputs_tags="remove-output" \
--TagRemovePreprocessor.remove_input_tags="remove-input"