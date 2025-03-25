import asyncio
import os
from pathlib import Path
import pytest
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from processors.factory import ProcessorFactory

# Create test data directory if it doesn't exist
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

def create_test_pdf():
    """Create a test PDF file with sample content."""
    # Create PDF content using ReportLab
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Add a title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Sample Investment Report")
    
    # Add some text content
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, "Company: ABC Corporation")
    c.drawString(100, 680, "Date: January 1, 2024")
    
    # Add a table-like structure
    y_position = 600
    headers = ["Quarter", "Revenue", "Profit", "Growth"]
    data = [
        ["Q1 2024", "$1M", "$200K", "10%"],
        ["Q2 2024", "$1.2M", "$240K", "12%"],
        ["Q3 2024", "$1.4M", "$280K", "15%"]
    ]
    
    # Draw headers
    x_position = 100
    for header in headers:
        c.drawString(x_position, y_position, header)
        x_position += 100
    
    # Draw data rows
    y_position -= 20
    for row in data:
        x_position = 100
        for cell in row:
            c.drawString(x_position, y_position, cell)
            x_position += 100
        y_position -= 20
    
    # Add some additional text
    c.drawString(100, 450, "Key Findings:")
    c.drawString(120, 430, "• Strong revenue growth")
    c.drawString(120, 410, "• Increasing profit margins")
    c.drawString(120, 390, "• Positive market outlook")
    
    c.save()
    
    # Create a PDF file with metadata
    buffer.seek(0)
    writer = PdfWriter()
    reader = PdfReader(buffer)
    writer.add_page(reader.pages[0])
    
    # Add metadata
    writer.add_metadata({
        "/Title": "Sample Investment Report",
        "/Author": "Financial Analyst",
        "/Subject": "Quarterly Financial Analysis",
        "/Keywords": "investment,finance,quarterly report",
        "/Creator": "PDF Processor Test",
        "/Producer": "Test Suite"
    })
    
    # Save the final PDF
    test_file = TEST_DATA_DIR / "test_report.pdf"
    with open(test_file, "wb") as output_file:
        writer.write(output_file)
    
    return test_file

@pytest.mark.asyncio
async def test_pdf_processor():
    """Test the PDF processor with a sample file."""
    # Create test PDF file
    test_file = create_test_pdf()
    
    try:
        # Get the PDF processor
        processor = ProcessorFactory.get_processor(test_file)
        
        # Process the document
        result = await processor.process(test_file)
        
        # Verify processing status
        assert result["processing_status"] == "success"
        
        # Verify metadata
        metadata = result["metadata"]
        assert metadata["extension"] == ".pdf"
        assert metadata["title"] == "Sample Investment Report"
        assert metadata["author"] == "Financial Analyst"
        assert metadata["subject"] == "Quarterly Financial Analysis"
        assert "investment" in metadata["keywords"].lower()
        
        # Verify content
        content = result["content"]
        
        # Check text content
        assert len(content["text"]) > 0
        assert "ABC Corporation" in content["text"]
        assert "Sample Investment Report" in content["text"]
        assert "Key Findings" in content["text"]
        
        # Check for specific data points
        assert "Q1 2024" in content["text"]
        assert "$1M" in content["text"]
        assert "Strong revenue growth" in content["text"]
        
        print("\nTest Results:")
        print("-" * 50)
        print(f"File: {test_file}")
        print(f"Status: {result['processing_status']}")
        print("\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print("\nContent Summary:")
        print(f"  Text Length: {len(content['text'])} characters")
        print(f"  Tables Found: {len(content['tables'])}")
        print(f"  Charts Found: {len(content['charts'])}")
        
        # Print first 200 characters of extracted text
        print("\nExtracted Text Preview:")
        print(f"  {content['text'][:200]}...")
        
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    asyncio.run(test_pdf_processor()) 