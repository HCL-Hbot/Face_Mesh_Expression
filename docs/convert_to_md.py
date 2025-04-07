import os
import argparse
from markitdown import MarkItDown

def convert_to_markdown_with_output(input_path: str, output_path: str) -> str | None:
    """
    Convert a document to Markdown format with specified output path.
    
    Args:
        input_path: Path to the input document
        output_path: Path where the markdown file will be saved
        
    Returns:
        str: Path to the output file if successful, None otherwise
    """
    try:
        # Initialize MarkItDown
        md = MarkItDown()
        
        # Convert the document
        result = md.convert(input_path)
        
        # Get markdown content
        markdown_content = result.text_content
        
        # Save Markdown file
        with open(output_path, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
            
        return output_path
    
    except Exception as e:
        print(f"Error converting document: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Convert documents to Markdown format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input_file',
        help='Path to the input document to convert'
    )
    parser.add_argument(
        'output_file',
        help='Path where the markdown file will be saved'
    )
    
    args = parser.parse_args()
    
    result = convert_to_markdown_with_output(args.input_file, args.output_file)
    
    if result:
        print(f"Successfully converted to: {args.output_file}")
    else:
        print("Conversion failed")
        exit(1)

if __name__ == "__main__":
    main()
