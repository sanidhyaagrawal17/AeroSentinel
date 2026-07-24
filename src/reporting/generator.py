from fpdf import FPDF
from pathlib import Path
import datetime
import cv2
import tempfile
import os

class ReportGenerator:
    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'outputs'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, aircraft_id, inspector_name, results, annotated_image, original_image=None):
        """
        Generates an offline PDF report with detection results.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"InspectionReport_{aircraft_id}_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        pdf = FPDF()
        pdf.add_page()
        
        # Fonts
        pdf.set_font('Helvetica', 'B', 16)
        
        # Header
        pdf.cell(0, 10, 'AeroSentinel MRO Inspection Report', 0, 1, 'C')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5, 'AI-Powered Structural Vigilance', 0, 1, 'C')
        pdf.ln(10)
        
        # Metadata
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(50, 8, 'Aircraft ID:', 0, 0)
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, aircraft_id, 0, 1)
        
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(50, 8, 'Inspector Name:', 0, 0)
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, inspector_name, 0, 1)
        
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(50, 8, 'Date/Time:', 0, 0)
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
        pdf.ln(10)
        
        # Summary
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Inspection Summary', 0, 1)
        
        num_critical = sum(1 for r in results if r['severity'] == 'Critical')
        num_medium = sum(1 for r in results if r['severity'] == 'Medium')
        num_low = sum(1 for r in results if r['severity'] == 'Low')
        
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, f"Total Defects Detected: {len(results)}", 0, 1)
        pdf.cell(0, 8, f"Critical Severity: {num_critical}", 0, 1)
        pdf.cell(0, 8, f"Medium Severity: {num_medium}", 0, 1)
        pdf.cell(0, 8, f"Low Severity: {num_low}", 0, 1)
        pdf.ln(5)
        
        # Save image to temp file to embed in PDF
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_img_path = tmp_file.name
            
        cv2.imwrite(tmp_img_path, annotated_image)
        
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Annotated Imagery', 0, 1)
        
        # Embed image (auto-scale width to fit page)
        pdf.image(tmp_img_path, x=10, w=190)
        pdf.ln(5)
        
        os.unlink(tmp_img_path)
        
        # Defect Table
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Defect Log', 0, 1)
        
        pdf.set_font('Helvetica', 'B', 10)
        # Table Header
        col_widths = [15, 60, 30, 30, 55]
        pdf.cell(col_widths[0], 10, 'ID', 1, 0, 'C')
        pdf.cell(col_widths[1], 10, 'Class', 1, 0, 'C')
        pdf.cell(col_widths[2], 10, 'Confidence', 1, 0, 'C')
        pdf.cell(col_widths[3], 10, 'Severity', 1, 0, 'C')
        pdf.cell(col_widths[4], 10, 'Coordinates (x1,y1,x2,y2)', 1, 1, 'C')
        
        pdf.set_font('Helvetica', '', 10)
        for i, res in enumerate(results):
            pdf.cell(col_widths[0], 8, str(i+1), 1, 0, 'C')
            pdf.cell(col_widths[1], 8, res['class_name'], 1, 0, 'L')
            pdf.cell(col_widths[2], 8, f"{res['confidence']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[3], 8, res['severity'], 1, 0, 'C')
            bbox_str = f"[{int(res['bbox'][0])},{int(res['bbox'][1])},{int(res['bbox'][2])},{int(res['bbox'][3])}]"
            pdf.cell(col_widths[4], 8, bbox_str, 1, 1, 'C')
            
        pdf.ln(20)
        
        # Sign-off block
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Certifying Engineer Sign-off', 0, 1)
        pdf.ln(15)
        pdf.cell(80, 8, '_' * 40, 0, 0)
        pdf.cell(40, 8, '', 0, 0)
        pdf.cell(60, 8, '_' * 30, 0, 1)
        
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(80, 5, 'Signature', 0, 0, 'C')
        pdf.cell(40, 5, '', 0, 0)
        pdf.cell(60, 5, 'Date', 0, 1, 'C')
        
        pdf.output(str(filepath))
        return filepath
