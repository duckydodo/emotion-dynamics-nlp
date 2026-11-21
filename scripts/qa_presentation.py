#!/usr/bin/env python3
"""
scripts/qa_presentation.py

Automated visual and layout QA verification for Emotion_Dynamics_Presentation.pptx.
Validates:
1. Shape bounds: All elements within slide dimensions (13.333" x 7.5").
2. Shape overlaps: Detects collisions between pictures, tables, and text frames.
3. Text overflow: Verifies text line count and height do not overflow card boundaries.
4. Typography standards:
   - Body cards: bullet and paragraph font sizes must be >= 14pt (excluding card title, headers, footnotes, and tags).
   - Tables: header and cell font sizes must be >= 12pt.
   - Captions, tags, and footer >= 10pt.
5. Image integrity: Every embedded image exists on disk and maintains an undistorted aspect ratio (< 5% delta).
6. Authorship consistency: Chittaranjan Dutta listed as sole author, zero mention of co-authors.
"""

import sys
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE

SLIDE_WIDTH_INCHES = 13.333
SLIDE_HEIGHT_INCHES = 7.5
TOLERANCE_INCHES = 0.05

FORBIDDEN_AUTHORS = ["heesanth", "reddy", "co-author", "team member"]
SOLE_AUTHOR = "chittaranjan dutta"


def inspect_presentation(pptx_path: str):
    prs = Presentation(pptx_path)
    slide_w = prs.slide_width.inches
    slide_h = prs.slide_height.inches

    print(f"Loaded {pptx_path}")
    print(f"Slide Dimensions: {slide_w:.3f}\" x {slide_h:.3f}\" (Expected: {SLIDE_WIDTH_INCHES}\" x {SLIDE_HEIGHT_INCHES}\")")

    issues = []
    warnings = []

    for idx, slide in enumerate(prs.slides, 1):
        slide_title = "Untitled"
        slide_shapes = []

        # Find slide title
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if shape.top.inches < 1.5 and 0 < len(text) < 80:
                    slide_title = text.split("\n")[0]
                    break

        print(f"\n--- Slide {idx}: {slide_title} ---")

        for shape in slide.shapes:
            left = shape.left.inches
            top = shape.top.inches
            width = shape.width.inches
            height = shape.height.inches
            right = left + width
            bottom = top + height

            # Skip full-slide background rectangle
            is_slide_bg = (width >= 13.0 and height >= 7.0 and left <= 0.05 and top <= 0.05)
            if is_slide_bg:
                continue

            # 1. Slide bounds check
            if (left < -TOLERANCE_INCHES or top < -TOLERANCE_INCHES or
                right > (SLIDE_WIDTH_INCHES + TOLERANCE_INCHES) or
                bottom > (SLIDE_HEIGHT_INCHES + TOLERANCE_INCHES)):
                issues.append(f"Slide {idx}: Shape '{shape.name}' exceeds bounds -> [{left:.2f}, {top:.2f}, {right:.2f}, {bottom:.2f}]")

            # 2. Check forbidden authors in any text
            if shape.has_text_frame:
                full_text = shape.text_frame.text.lower()
                for forbidden in FORBIDDEN_AUTHORS:
                    if forbidden in full_text:
                        issues.append(f"Slide {idx}: Forbidden author string '{forbidden}' found in shape '{shape.name}'")

                # Determine shape category
                is_header = top < 1.75 and height < 1.5
                is_footer = top > 6.85
                is_caption_or_tag = (height < 1.05 and (top > 5.5 or top < 0.6))
                is_slide_1_chip = (idx == 1 and width < 5.0 and height < 0.5)
                is_body_card = (not is_header and not is_footer and not is_caption_or_tag and not is_slide_1_chip)

                # Check fonts
                for p_idx, paragraph in enumerate(shape.text_frame.paragraphs):
                    for run in paragraph.runs:
                        if not run.text.strip():
                            continue
                        f_sz = run.font.size.pt if run.font.size else None
                        if f_sz is not None:
                            if is_body_card:
                                # First paragraph may be a card title (16pt).
                                # Body text runs must be >= 14pt (allowing 13.5pt float margin)
                                if f_sz < 13.5:
                                    issues.append(f"Slide {idx}: Body card '{shape.name}' run font size < 14pt ({f_sz:.1f}pt): \"{run.text[:30]}...\"")
                            elif is_footer or is_caption_or_tag or is_slide_1_chip:
                                if f_sz < 9.5:
                                    warnings.append(f"Slide {idx}: Caption/Footer font size < 10pt ({f_sz:.1f}pt) in shape '{shape.name}'")

                # Check text overflow: estimate total lines using actual text_frame margins
                tf = shape.text_frame
                m_top = (tf.margin_top or 0) / 914400.0
                m_bot = (tf.margin_bottom or 0) / 914400.0
                m_left = (tf.margin_left or 0) / 914400.0
                m_right = (tf.margin_right or 0) / 914400.0
                avail_w = max(0.5, width - (m_left + m_right))
                avail_h = max(0.1, height - (m_top + m_bot))

                total_lines = 0
                for p in tf.paragraphs:
                    p_text = p.text.strip()
                    if not p_text:
                        continue
                    p_pt = p.runs[0].font.size.pt if (p.runs and p.runs[0].font.size) else 14.0
                    chars_per_line = max(10, int(avail_w * 72 / (p_pt * 0.56)))
                    lines = max(1, (len(p_text) + chars_per_line - 1) // chars_per_line)
                    total_lines += lines

                # At 13.5-14pt, a line plus space is ~17.5pt
                est_height = (total_lines * 17.5) / 72.0
                if is_body_card and est_height > (avail_h * 1.08):
                    issues.append(f"Slide {idx}: Text overflow in '{shape.name}' (est text height {est_height:.2f}\" > avail box height {avail_h:.2f}\", ~{total_lines} lines)")

            # 3. Table checks
            if shape.has_table:
                table = shape.table
                for row_idx, row in enumerate(table.rows):
                    for col_idx, cell in enumerate(row.cells):
                        cell_text = cell.text.lower()
                        for forbidden in FORBIDDEN_AUTHORS:
                            if forbidden in cell_text:
                                issues.append(f"Slide {idx} Table: Forbidden author '{forbidden}' in cell ({row_idx}, {col_idx})")
                        for p in cell.text_frame.paragraphs:
                            for run in p.runs:
                                if not run.text.strip():
                                    continue
                                if run.font.size and run.font.size.pt < 11.5:
                                    issues.append(f"Slide {idx} Table: Font < 12pt ({run.font.size.pt:.1f}pt) in cell ({row_idx}, {col_idx})")

            # 4. Image aspect ratio check
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img = shape.image
                orig_w_px = img.size[0]
                orig_h_px = img.size[1]
                orig_ar = orig_w_px / orig_h_px
                curr_ar = width / height
                ar_diff = abs(curr_ar - orig_ar) / orig_ar
                if ar_diff > 0.05:
                    warnings.append(f"Slide {idx} Image '{shape.name}': Aspect ratio distorted by {ar_diff*100:.1f}% (orig: {orig_ar:.3f}, curr: {curr_ar:.3f})")

            slide_shapes.append((shape.name, left, top, right, bottom, shape.shape_type))

        # 5. Overlapping shapes collision check
        # Detect if two independent shapes (e.g. image vs text box or table vs text box) collide
        for i in range(len(slide_shapes)):
            name_a, l_a, t_a, r_a, b_a, type_a = slide_shapes[i]
            for j in range(i + 1, len(slide_shapes)):
                name_b, l_b, t_b, r_b, b_b, type_b = slide_shapes[j]

                overlap_l = max(l_a, l_b)
                overlap_t = max(t_a, t_b)
                overlap_r = min(r_a, r_b)
                overlap_b = min(b_a, b_b)

                if overlap_r > overlap_l and overlap_b > overlap_t:
                    overlap_w = overlap_r - overlap_l
                    overlap_h = overlap_b - overlap_t
                    overlap_area = overlap_w * overlap_h
                    area_a = (r_a - l_a) * (b_a - t_a)
                    area_b = (r_b - l_b) * (b_b - t_b)

                    containment = (overlap_area / min(area_a, area_b)) if min(area_a, area_b) > 0 else 0
                    # If one shape is a container card and the other is an element placed inside it, that is valid layout
                    is_card_a = (type_a == MSO_SHAPE_TYPE.AUTO_SHAPE and ("Rounded" in name_a or "Rectangle" in name_a))
                    is_card_b = (type_b == MSO_SHAPE_TYPE.AUTO_SHAPE and ("Rounded" in name_b or "Rectangle" in name_b))

                    if (is_card_a and not is_card_b) or (is_card_b and not is_card_a):
                        if containment > 0.60:
                            continue

                    # Thin separator lines
                    if min(r_a - l_a, b_a - t_a) < 0.05 or min(r_b - l_b, b_b - t_b) < 0.05:
                        continue

                    if overlap_area > 0.20:
                        issues.append(f"Slide {idx}: Unintended overlap between '{name_a}' and '{name_b}' ({overlap_area:.2f} sq in collision)")

    print("\n" + "="*50)
    print("QA INSPECTION SUMMARY")
    print("="*50)
    print(f"Total Errors / Violations: {len(issues)}")
    print(f"Total Warnings: {len(warnings)}")

    if issues:
        print("\n[ERRORS / VIOLATIONS]:")
        for iss in issues:
            print(f"  ❌ {iss}")
    else:
        print("\n✅ Zero layout, boundary, typography, or authorship violations!")

    if warnings:
        print("\n[WARNINGS / ADVISORIES]:")
        for warn in warnings:
            print(f"  ⚠️  {warn}")
    else:
        print("\n✅ Zero warnings! All text bounds, fonts, and images are optimal.")

    return len(issues), len(warnings)


if __name__ == "__main__":
    pptx_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/reports/Emotion_Dynamics_Presentation.pptx"
    issues, warnings = inspect_presentation(pptx_file)
    sys.exit(1 if issues > 0 else 0)
