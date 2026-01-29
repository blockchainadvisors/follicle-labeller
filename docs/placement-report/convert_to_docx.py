"""
Convert placement report markdown files to a single formatted DOCX document.
Uses professional formatting with Calibri font and proper heading styles.
"""

import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# Configuration
REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(REPORT_DIR, "Placement_Report_Laurentiu_Nae.docx")

# File order
CHAPTERS = [
    "README.md",
    "chapters/01-project-overview.md",
    "chapters/02-technical-architecture.md",
    "chapters/03-development-logbook.md",
    "chapters/04-features-implemented.md",
    "chapters/05-challenges-and-solutions.md",
    "chapters/06-technologies-and-tools.md",
    "chapters/07-testing-and-quality.md",
    "chapters/08-conclusions-and-future-work.md",
]

def setup_styles(doc):
    """Configure document styles with professional formatting."""

    # Set default font for Normal style
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0, 0, 0)  # Black

    # Paragraph formatting
    paragraph_format = style.paragraph_format
    paragraph_format.space_after = Pt(6)
    paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE

    # Title style
    if 'Title' in doc.styles:
        title_style = doc.styles['Title']
        title_style.font.name = 'Calibri'
        title_style.font.size = Pt(26)
        title_style.font.bold = True
        title_style.font.color.rgb = RGBColor(0, 0, 0)
        title_style.paragraph_format.space_after = Pt(24)
        title_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Heading 1 - Chapter titles
    h1 = doc.styles['Heading 1']
    h1.font.name = 'Calibri'
    h1.font.size = Pt(18)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0, 0, 0)
    h1.paragraph_format.space_before = Pt(24)
    h1.paragraph_format.space_after = Pt(12)
    h1.paragraph_format.page_break_before = True

    # Heading 2 - Major sections
    h2 = doc.styles['Heading 2']
    h2.font.name = 'Calibri'
    h2.font.size = Pt(14)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0, 0, 0)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(8)

    # Heading 3 - Subsections
    h3 = doc.styles['Heading 3']
    h3.font.name = 'Calibri'
    h3.font.size = Pt(12)
    h3.font.bold = True
    h3.font.color.rgb = RGBColor(0, 0, 0)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    # Heading 4 - Sub-subsections
    h4 = doc.styles['Heading 4']
    h4.font.name = 'Calibri'
    h4.font.size = Pt(11)
    h4.font.bold = True
    h4.font.color.rgb = RGBColor(0, 0, 0)
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(4)

    # Create Code style
    try:
        code_style = doc.styles.add_style('Code', WD_STYLE_TYPE.PARAGRAPH)
    except:
        code_style = doc.styles['Code']
    code_style.font.name = 'Consolas'
    code_style.font.size = Pt(9)
    code_style.font.color.rgb = RGBColor(0, 0, 0)
    code_style.paragraph_format.space_before = Pt(6)
    code_style.paragraph_format.space_after = Pt(6)
    code_style.paragraph_format.left_indent = Inches(0.25)

    return doc

def add_table_of_contents(doc):
    """Add a table of contents placeholder."""
    doc.add_heading('Table of Contents', level=1)
    # Note: Actual TOC would need field codes, this is a placeholder
    p = doc.add_paragraph()
    p.add_run('(Update this table of contents in Word: References > Update Table)')
    p.italic = True
    doc.add_page_break()

def parse_markdown_line(line, doc, in_code_block, code_lines, in_table, table_rows):
    """Parse a single markdown line and add to document."""

    # Handle code blocks
    if line.startswith('```'):
        if in_code_block:
            # End code block - add accumulated code
            if code_lines:
                for code_line in code_lines:
                    p = doc.add_paragraph(code_line, style='Code')
            return False, [], in_table, table_rows
        else:
            return True, [], in_table, table_rows

    if in_code_block:
        code_lines.append(line)
        return in_code_block, code_lines, in_table, table_rows

    # Handle tables
    if '|' in line and not line.startswith('```'):
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]  # Remove empty cells

        # Skip separator lines
        if cells and all(set(c) <= set('-: ') for c in cells):
            return in_code_block, code_lines, True, table_rows

        if cells:
            table_rows.append(cells)
            return in_code_block, code_lines, True, table_rows
    elif in_table and table_rows:
        # End of table - create it
        create_table(doc, table_rows)
        return in_code_block, code_lines, False, []

    # Handle headings
    if line.startswith('#'):
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            # Clean up markdown formatting in heading
            text = clean_markdown_text(text)
            if level <= 4:
                doc.add_heading(text, level=level)
            else:
                p = doc.add_paragraph()
                run = p.add_run(text)
                run.bold = True
            return in_code_block, code_lines, in_table, table_rows

    # Handle horizontal rules
    if line.strip() in ['---', '***', '___']:
        p = doc.add_paragraph()
        p.add_run('_' * 50)
        return in_code_block, code_lines, in_table, table_rows

    # Handle bullet points
    if re.match(r'^[-*+]\s+', line):
        text = re.sub(r'^[-*+]\s+', '', line)
        text = clean_markdown_text(text)
        p = doc.add_paragraph(text, style='List Bullet')
        return in_code_block, code_lines, in_table, table_rows

    # Handle numbered lists
    if re.match(r'^\d+\.\s+', line):
        text = re.sub(r'^\d+\.\s+', '', line)
        text = clean_markdown_text(text)
        p = doc.add_paragraph(text, style='List Number')
        return in_code_block, code_lines, in_table, table_rows

    # Handle indented bullet points (sub-items)
    if re.match(r'^(\s{2,4})[-*+]\s+', line):
        text = re.sub(r'^(\s{2,4})[-*+]\s+', '', line)
        text = clean_markdown_text(text)
        p = doc.add_paragraph(text, style='List Bullet 2')
        return in_code_block, code_lines, in_table, table_rows

    # Handle regular paragraphs
    if line.strip():
        text = clean_markdown_text(line)
        p = doc.add_paragraph()
        add_formatted_text(p, text)

    return in_code_block, code_lines, in_table, table_rows

def clean_markdown_text(text):
    """Remove or convert markdown formatting."""
    # Remove links but keep text: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove image references
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[Image: \1]', text)
    return text

def add_formatted_text(paragraph, text):
    """Add text with inline formatting (bold, italic, code)."""
    # Pattern for inline code, bold, italic
    pattern = r'(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_)'
    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue
        if part.startswith('`') and part.endswith('`'):
            # Inline code
            run = paragraph.add_run(part[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(10)
        elif (part.startswith('**') and part.endswith('**')) or (part.startswith('__') and part.endswith('__')):
            # Bold
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif (part.startswith('*') and part.endswith('*')) or (part.startswith('_') and part.endswith('_')):
            # Italic
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        else:
            paragraph.add_run(part)

def create_table(doc, rows):
    """Create a formatted table from row data."""
    if not rows:
        return

    # Determine number of columns
    num_cols = max(len(row) for row in rows)

    # Create table
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(rows):
        row = table.rows[i]
        for j, cell_text in enumerate(row_data):
            if j < num_cols:
                cell = row.cells[j]
                cell.text = cell_text
                # Make header row bold
                if i == 0:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.bold = True
                # Set font
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = 'Calibri'
                        run.font.size = Pt(10)
                        run.font.color.rgb = RGBColor(0, 0, 0)

    # Add spacing after table
    doc.add_paragraph()

def process_markdown_file(doc, filepath):
    """Process a markdown file and add content to document."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    for line in lines:
        in_code_block, code_lines, in_table, table_rows = parse_markdown_line(
            line, doc, in_code_block, code_lines, in_table, table_rows
        )

    # Handle any remaining table
    if table_rows:
        create_table(doc, table_rows)

def add_title_page(doc):
    """Create a professional title page."""
    # Add some spacing
    for _ in range(4):
        doc.add_paragraph()

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Follicle Labeller")
    run.font.name = 'Calibri'
    run.font.size = Pt(32)
    run.bold = True
    run.font.color.rgb = RGBColor(0, 0, 0)

    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Medical Image Annotation Tool")
    run.font.name = 'Calibri'
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(80, 80, 80)

    doc.add_paragraph()
    doc.add_paragraph()

    # Report type
    report_type = doc.add_paragraph()
    report_type.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = report_type.add_run("Master's Placement Project Report")
    run.font.name = 'Calibri'
    run.font.size = Pt(16)
    run.bold = True

    # Add spacing
    for _ in range(6):
        doc.add_paragraph()

    # Author info
    info_lines = [
        ("Student:", "Laurentiu Nae"),
        ("Organization:", "Blockchain Advisors"),
        ("Period:", "January 2026"),
        ("Version:", "2.0.2"),
    ]

    for label, value in info_lines:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run1 = p.add_run(f"{label} ")
        run1.font.name = 'Calibri'
        run1.font.size = Pt(12)
        run1.bold = True
        run2 = p.add_run(value)
        run2.font.name = 'Calibri'
        run2.font.size = Pt(12)

    # Add spacing
    for _ in range(4):
        doc.add_paragraph()

    # Date
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run("January 29, 2026")
    run.font.name = 'Calibri'
    run.font.size = Pt(11)
    run.italic = True

    doc.add_page_break()

def main():
    """Main function to create the DOCX document."""
    print("Creating DOCX document...")

    # Create document
    doc = Document()

    # Set up styles
    setup_styles(doc)

    # Set page margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1.25)
        section.right_margin = Inches(1.25)

    # Add title page
    add_title_page(doc)

    # Process each chapter
    for chapter_file in CHAPTERS:
        filepath = os.path.join(REPORT_DIR, chapter_file)
        if os.path.exists(filepath):
            print(f"Processing: {chapter_file}")
            process_markdown_file(doc, filepath)
        else:
            print(f"Warning: File not found: {filepath}")

    # Save document
    doc.save(OUTPUT_FILE)
    print(f"\nDocument saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
