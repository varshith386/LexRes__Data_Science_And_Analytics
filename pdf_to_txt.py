import PyPDF2
import os

def pdf_to_txt(pdf_file, txt_file):
    # Generate a unique file name if the txt_file already exists
    base, extension = os.path.splitext(txt_file)
    counter = 1
    new_txt_file = txt_file
    
    while os.path.exists(new_txt_file):
        new_txt_file = f"{base}_{counter}{extension}"
        counter += 1
    
    # Open the PDF file
    with open(pdf_file, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize a string to hold the PDF content
        pdf_text = ''
        
        # Loop through all the pages in the PDF
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
        
    # Write the PDF content to a text file
    with open(new_txt_file, 'w', encoding='utf-8') as file:
        file.write(pdf_text)
    
    print(f"PDF content has been written to {new_txt_file}")

# Example usage
pdf_file = r'pdf1.pdf'  # Replace with your PDF file name
txt_file = 'output.txt'   # Replace with your desired output text file name
pdf_to_txt(pdf_file, txt_file)
