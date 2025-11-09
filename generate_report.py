#!/usr/bin/env python3
"""
Generate HTML report from Jupyter notebook with all charts and analysis
"""

import json
import base64
import io
from pathlib import Path

# Read the notebook
notebook_path = Path('/home/ycnin/moore/data_exploration.ipynb')
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moore Loan Portfolio - Data Exploration Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 25px;
        }}
        .code-cell {{
            background-color: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        .output {{
            background-color: #ffffff;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .markdown {{
            line-height: 1.8;
        }}
        .markdown ul, .markdown ol {{
            margin-left: 20px;
        }}
        .markdown code {{
            background-color: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            color: #e83e8c;
        }}
        .markdown pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .markdown table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        .markdown th, .markdown td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .markdown th {{
            background-color: #667eea;
            color: white;
        }}
        .markdown tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc h2 {{
            color: #667eea;
            margin-top: 0;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            padding: 5px 0;
        }}
        .toc a {{
            color: #764ba2;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .insight {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .insight strong {{
            color: #856404;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .section {{
                box-shadow: none;
                page-break-inside: avoid;
            }}
            .header {{
                background: #667eea;
                page-break-after: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Moore Loan Portfolio</h1>
        <p>Comprehensive Data Exploration & Analysis Report</p>
        <p style="font-size: 0.9em; margin-top: 20px;">Generated from data_exploration.ipynb</p>
    </div>

    {content}

    <div class="section">
        <p style="text-align: center; color: #666; margin-top: 40px;">
            <em>Report generated using Python/Jupyter • Moore Loan Portfolio v1.0</em>
        </p>
    </div>
</body>
</html>
"""

def process_markdown(source):
    """Convert markdown source to HTML"""
    text = ''.join(source) if isinstance(source, list) else source

    # Simple markdown to HTML conversion
    # Headers
    text = text.replace('### ', '<h3>').replace('\n', '</h3>\n', 1) if '### ' in text else text
    text = text.replace('## ', '<h2>').replace('\n', '</h2>\n', 1) if '## ' in text else text
    text = text.replace('# ', '<h1>').replace('\n', '</h1>\n', 1) if '# ' in text else text

    # Bold and italic
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Code blocks
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)

    # Line breaks
    text = text.replace('\n\n', '</p><p>')

    return f'<div class="markdown"><p>{text}</p></div>'

def process_output(output):
    """Process cell output"""
    output_html = []

    if output.get('output_type') == 'stream':
        text = ''.join(output.get('text', []))
        output_html.append(f'<div class="output">{text}</div>')

    elif output.get('output_type') == 'execute_result':
        # Text output
        if 'text/plain' in output.get('data', {}):
            text = ''.join(output['data']['text/plain'])
            output_html.append(f'<div class="output">{text}</div>')

        # Image output
        if 'image/png' in output.get('data', {}):
            img_data = output['data']['image/png']
            output_html.append(f'<div class="chart"><img src="data:image/png;base64,{img_data}" alt="Chart"></div>')

    elif output.get('output_type') == 'display_data':
        # Image output
        if 'image/png' in output.get('data', {}):
            img_data = output['data']['image/png']
            output_html.append(f'<div class="chart"><img src="data:image/png;base64,{img_data}" alt="Chart"></div>')

    return '\n'.join(output_html)

# Generate HTML content
content_parts = []
section_count = 0

for i, cell in enumerate(notebook['cells']):
    cell_type = cell.get('cell_type')
    source = cell.get('source', [])

    if cell_type == 'markdown':
        # Check if this is a section header
        source_text = ''.join(source)
        if source_text.startswith('##') and not source_text.startswith('###'):
            section_count += 1
            content_parts.append('</div>' if section_count > 1 else '')
            content_parts.append('<div class="section">')

        content_parts.append(process_markdown(source))

    elif cell_type == 'code':
        # Optionally show code (comment out if you don't want to show code)
        # code_text = ''.join(source)
        # if code_text.strip():
        #     content_parts.append(f'<div class="code-cell">{code_text}</div>')

        # Process outputs
        outputs = cell.get('outputs', [])
        for output in outputs:
            output_html = process_output(output)
            if output_html:
                content_parts.append(output_html)

# Close last section
content_parts.append('</div>')

# Generate final HTML
html_content = html_template.format(content='\n'.join(content_parts))

# Write HTML file
output_path = Path('/home/ycnin/moore/analysis_report.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✓ HTML report generated: {output_path}")
print(f"\nTo create PDF:")
print(f"1. Open the HTML file in your browser: {output_path}")
print(f"2. Press Ctrl+P (or Cmd+P on Mac)")
print(f"3. Select 'Save as PDF' as the printer")
print(f"4. Save the PDF")
print(f"\nAlternatively, you can use wkhtmltopdf or Chrome headless to convert to PDF programmatically.")
