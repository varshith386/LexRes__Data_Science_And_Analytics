from ext2 import facts_summary
from ext2_issue import issues_summary
from ext3 import rules_summary
from ezt2_ana import analysis_summary
from ext_conc import conclusion_summary

# Genrates the whole summary and saves the file
file_path = r'summaries.txt'

# Open the file in write mode
with open(file_path, 'w') as file:
    # Write each summary to the file
    file.write("Facts Summary:\n")
    file.write(facts_summary + "\n\n")
    
    file.write("Issues Summary:\n")
    file.write(issues_summary + "\n\n")
    
    file.write("Rules Summary:\n")
    file.write(rules_summary + "\n\n")
    
    file.write("Analysis Summary:\n")
    file.write(analysis_summary + "\n\n")
    
    file.write("Conclusion Summary:\n")
    file.write(conclusion_summary + "\n\n")

# Print confirmation message
print(f"Summaries have been written to {file_path}")
