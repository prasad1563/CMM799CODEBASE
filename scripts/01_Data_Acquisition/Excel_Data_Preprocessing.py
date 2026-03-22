import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.excel_to_processed import ExcelToProcessedConverter


def main():
    """Main function to convert Excel files"""
    
    excel_files = [
        {
            'input': 'data/source/leptospirosis.xlsx',
            'output': 'data/processed/leptospirosis_processed.csv',
            'name': 'Leptospirosis'
        },
        {
            'input': 'data/source/socioeconomic.xlsx',
            'output': 'data/processed/socioeconomic_processed.csv',
            'name': 'Socioeconomic'
        }
    ]
    
    for file_config in excel_files:
        input_file = file_config['input']
        output_file = file_config['output']
        name = file_config['name']
        
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            print(f"Skipping {name} data conversion...\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {name} Data")
        print(f"{'='*60}\n")
        
        try:
            converter = ExcelToProcessedConverter(input_file, output_file)
            processed_data = converter.convert()
            print(f"\n Successfully converted {name} data!")
            
        except Exception as e:
            print(f"\n Error processing {name} data: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

