import asyncio
from pathlib import Path
from processors.factory import ProcessorFactory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def process_document(file_path: str):
    """Process a document and print the results."""
    try:
        # Get the appropriate processor
        processor = ProcessorFactory.get_processor(Path(file_path))
        
        # Process the document
        result = await processor.process(Path(file_path))
        
        # Print results
        print("\nProcessing Results:")
        print("-" * 50)
        print(f"File: {file_path}")
        print(f"Status: {result['processing_status']}")
        print("\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        
        print("\nContent Summary:")
        print(f"  Text Length: {len(result['content']['text'])} characters")
        print(f"  Tables Found: {len(result['content']['tables'])}")
        print(f"  Charts Found: {len(result['content']['charts'])}")
        
        if result['processing_status'] == 'error':
            print(f"\nError: {result['error_message']}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")

async def main():
    # Test with a PDF file
    await process_document("path/to/your/document.pdf")
    
    # Test with a DOCX file
    await process_document("path/to/your/document.docx")

if __name__ == "__main__":
    asyncio.run(main()) 