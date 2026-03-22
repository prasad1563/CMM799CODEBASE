import PyPDF2
import pandas as pd
import os
import re

def week_to_month_445(week):
    """Convert week number to month using 4-4-5 calendar system"""
    month_ranges = {
        1: range(1, 5),    2: range(5, 9),    3: range(9, 14),
        4: range(14, 18),  5: range(18, 22),  6: range(22, 27),
        7: range(27, 31),  8: range(31, 35),  9: range(35, 40),
        10: range(40, 44), 11: range(44, 48), 12: range(48, 54)
    }
    
    for month, weeks in month_ranges.items():
        if week in weeks:
            return f"Month_{month:02d}"
    return None

def extract_leptospirosis_from_pdf(pdf_path):

    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        # Get the second-to-last page (page before last page)
        page = pdf_reader.pages[total_pages - 2]
        text = page.extract_text()
    
    # Split text into lines
    lines = text.split('\n')
    
    # District names in order as they appear in the table
    districts = [
        'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'NuwaraEliya',
        'Galle', 'Hambantota', 'Matara', 'Jaffna', 'Kilinochchi', 'Mannar',
        'Vavuniya', 'Mullaitivu', 'Batticaloa', 'Ampara', 'Trincomalee',
        'Kurunegala', 'Puttalam', 'Anuradhapur', 'Polonnaruwa', 'Badulla',
        'Monaragala', 'Ratnapura', 'Kegalle', 'Kalmune'
    ]
    
    # Extract data for each district
    district_data = {}
    
    for district in districts:
        # Find the line containing this district
        for line in lines:
            # Match district name at the start of line (case-insensitive)
            if re.match(rf'^{district}\s+', line, re.IGNORECASE):
                # Extract all numbers from this line
                numbers = re.findall(r'\d+', line)
                
                if len(numbers) >= 12:
                    # Leptospirosis A field is at index 10
                    leptospirosis_a = int(numbers[10])
                    district_data[district] = leptospirosis_a
                else:
                    district_data[district] = 0
                break
        else:
            # District not found
            district_data[district] = 0
    
    return district_data

def process_all_pdfs(pdf_folder, output_csv='leptospirosis_monthly_2023.csv'):
    """
    Process all PDF files in the folder and create monthly summation CSV.
    """
    
    data = {}
    
    # Process each PDF file
    for file in sorted(os.listdir(pdf_folder)):
        if not file.endswith(".pdf"):
            continue
        
        pdf_path = os.path.join(pdf_folder, file)
        
        # Extract week number from filename
        match = re.search(r"No_(\d+)", file)
        if not match:
            print(f"Skipping {file} - couldn't extract week number")
            continue
        
        week_num = int(match.group(1))
        
        # Convert week to month using 4-4-5 method
        month = week_to_month_445(week_num)
        
        if month is None:
            print(f"Skipping {file} - week {week_num} out of range")
            continue
        
        print(f"Processing {file} -> Week {week_num} -> {month}")
        
        try:
            # Extract data from this PDF
            district_data = extract_leptospirosis_from_pdf(pdf_path)
            
            # Add to monthly summation
            for district, count in district_data.items():
                if district not in data:
                    data[district] = {}
                
                if month not in data[district]:
                    data[district][month] = 0
                
                data[district][month] += count
                
            print(f"  Extracted data for {len(district_data)} districts")
            
        except Exception as e:
            print(f"  Error processing {file}: {str(e)}")
            continue
    
    # Create DataFrame and save to CSV
    final_df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    final_df.index.name = "District"
    final_df = final_df.sort_index(axis=1)
    
    # Fill NaN values with 0
    final_df = final_df.fillna(0).astype(int)
    
    final_df.to_csv(output_csv)
    
    print(f"\n SUCCESS: Monthly data saved to {output_csv}")
    print(f"  - Districts: {len(final_df)}")
    print(f"  - Months: {len(final_df.columns)}")
    print(f"\nMonthly totals:")
    print(final_df.sum())

# Usage
if __name__ == "__main__":
    PDF_FOLDER = "WER_2024"
    OUTPUT_FILE = "leptospirosis_monthly_2024.csv"
    
    process_all_pdfs(PDF_FOLDER, OUTPUT_FILE)