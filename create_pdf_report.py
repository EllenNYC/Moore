#!/usr/bin/env python3
"""
Create a comprehensive PDF-ready report from the Jupyter notebook
Outputs both HTML (for browser PDF printing) and enhanced Markdown
"""

import json
import base64
from pathlib import Path
from datetime import datetime

# Read the notebook
notebook_path = Path('/home/ycnin/moore/data_exploration.ipynb')
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

print("Reading notebook...")

# Extract all markdown text and outputs
markdown_sections = []
chart_count = 0
table_count = 0

for i, cell in enumerate(notebook['cells']):
    cell_type = cell.get('cell_type')
    source = cell.get('source', [])
    source_text = ''.join(source)

    if cell_type == 'markdown':
        markdown_sections.append({
            'type': 'markdown',
            'content': source_text,
            'cell_num': i
        })

    elif cell_type == 'code':
        outputs = cell.get('outputs', [])

        for output in outputs:
            # Text output
            if output.get('output_type') == 'stream':
                text = ''.join(output.get('text', []))
                if text.strip():
                    markdown_sections.append({
                        'type': 'output',
                        'content': text,
                        'cell_num': i
                    })

            # Display data or execute result
            elif output.get('output_type') in ['execute_result', 'display_data']:
                # Check for images (charts)
                if 'image/png' in output.get('data', {}):
                    chart_count += 1
                    img_data = output['data']['image/png']

                    # Save image to file
                    img_path = Path(f'/home/ycnin/moore/charts/chart_{chart_count}.png')
                    img_path.parent.mkdir(exist_ok=True)

                    # Decode and save
                    import base64
                    img_bytes = base64.b64decode(img_data)
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_bytes)

                    markdown_sections.append({
                        'type': 'chart',
                        'path': str(img_path),
                        'number': chart_count,
                        'cell_num': i
                    })

                # Text output
                elif 'text/plain' in output.get('data', {}):
                    text = ''.join(output['data']['text/plain'])
                    if text.strip():
                        markdown_sections.append({
                            'type': 'output',
                            'content': text,
                            'cell_num': i
                        })

print(f"Extracted {len(markdown_sections)} sections")
print(f"Found {chart_count} charts")

# Create enhanced markdown report
md_report = f"""# Moore Loan Portfolio - Data Exploration Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Data Source:** loan tape - moore v1.0.csv, loan performance - moore v1.0.csv

---

"""

for section in markdown_sections:
    if section['type'] == 'markdown':
        md_report += section['content'] + '\n\n'

    elif section['type'] == 'output':
        md_report += '```\n' + section['content'] + '\n```\n\n'

    elif section['type'] == 'chart':
        md_report += f"![Chart {section['number']}]({section['path']})\n\n"
        md_report += f"*Chart {section['number']}: [Description based on analysis]*\n\n"

# Save enhanced markdown
md_path = Path('/home/ycnin/moore/analysis_report_with_charts.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_report)

print(f"\n✓ Enhanced markdown report created: {md_path}")
print(f"✓ Charts saved to: /home/ycnin/moore/charts/")
print(f"\nFiles created:")
print(f"  1. {md_path} - Markdown with embedded chart references")
print(f"  2. /home/ycnin/moore/charts/chart_*.png - Individual chart images ({chart_count} files)")
print(f"  3. /home/ycnin/moore/analysis_report.html - HTML report (already created)")

print(f"\n" + "="*80)
print("TO CREATE PDF:")
print("="*80)
print("\nOption 1 - Using Browser (Recommended):")
print("  1. Open analysis_report.html in your browser")
print("  2. Press Ctrl+P (or Cmd+P)")
print("  3. Select 'Save as PDF'")
print("  4. Adjust settings: Enable background graphics, set margins")
print("  5. Save as analysis_report.pdf")

print("\nOption 2 - Using Markdown converter (if you have pandoc/wkhtmltopdf):")
print(f"  pandoc {md_path} -o analysis_report.pdf --pdf-engine=wkhtmltopdf")

print("\nOption 3 - Using VS Code:")
print("  1. Install 'Markdown PDF' extension")
print(f"  2. Open {md_path}")
print("  3. Right-click → 'Markdown PDF: Export (pdf)'")

print("\n" + "="*80)
