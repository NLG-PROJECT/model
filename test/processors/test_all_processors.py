import asyncio
import os
from pathlib import Path
import pytest
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from processors.factory import ProcessorFactory
from test_pdf_processor import create_test_pdf
from test_docx_processor import create_test_docx
from test_excel_processor import create_test_excel

async def test_all_processors():
    """Test all processors and show their output format."""
    # Create test files
    pdf_file = create_test_pdf()
    docx_file = create_test_docx()
    excel_file = create_test_excel()
    
    try:
        # Process PDF
        print("\n=== PDF Processor Output ===")
        pdf_processor = ProcessorFactory.get_processor(pdf_file)
        pdf_result = await pdf_processor.process(pdf_file)
        print("\nProcessing Status:", pdf_result["processing_status"])
        print("\nMetadata:")
        for key, value in pdf_result["metadata"].items():
            print(f"  {key}: {value}")
        print("\nContent Summary:")
        print(f"  Text Length: {len(pdf_result['content']['text'])} characters")
        print(f"  Tables Found: {len(pdf_result['content']['tables'])}")
        print(f"  Charts Found: {len(pdf_result['content']['charts'])}")
        print("\nText Preview (first 200 chars):")
        print(f"  {pdf_result['content']['text'][:200]}...")
        
        # Process DOCX
        print("\n=== DOCX Processor Output ===")
        docx_processor = ProcessorFactory.get_processor(docx_file)
        docx_result = await docx_processor.process(docx_file)
        print("\nProcessing Status:", docx_result["processing_status"])
        print("\nMetadata:")
        for key, value in docx_result["metadata"].items():
            print(f"  {key}: {value}")
        print("\nContent Summary:")
        print(f"  Text Length: {len(docx_result['content']['text'])} characters")
        print(f"  Tables Found: {len(docx_result['content']['tables'])}")
        print("\nFirst Table Structure:")
        if docx_result["content"]["tables"]:
            table = docx_result["content"]["tables"][0]
            print(f"  Total Rows: {len(table['rows'])}")
            print(f"  First Row: {table['rows'][0]}")
            print(f"  Second Row: {table['rows'][1]}")
        
        # Process Excel
        print("\n=== Excel Processor Output ===")
        excel_processor = ProcessorFactory.get_processor(excel_file)
        excel_result = await excel_processor.process(excel_file)
        print("\nProcessing Status:", excel_result["processing_status"])
        print("\nMetadata:")
        for key, value in excel_result["metadata"].items():
            print(f"  {key}: {value}")
        print("\nContent Summary:")
        print(f"  Text Length: {len(excel_result['content']['text'])} characters")
        print(f"  Tables Found: {len(excel_result['content']['tables'])}")
        print("\nTable Details:")
        for table in excel_result["content"]["tables"]:
            print(f"\n  Table ID: {table['table_id']}")
            print(f"  Rows: {table['row_count']}, Columns: {table['column_count']}")
            print(f"  First Row: {table['rows'][0]}")
            print(f"  Second Row: {table['rows'][1]}")
        
    finally:
        # Clean up test files
        for file in [pdf_file, docx_file, excel_file]:
            if file.exists():
                file.unlink()

if __name__ == "__main__":
    asyncio.run(test_all_processors()) 