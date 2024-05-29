input_file = '/Users/shohag/Desktop/HKU/FYP/SpamDetection/spam_commentsSet1 copy 2.csv'
output_file = '/Users/shohag/Desktop/HKU/FYP/SpamDetection/output.csv'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        line = line.strip()
        if line.count(',') > 1:
            first_comma_index = line.find(',')
            modified_line = line[:first_comma_index + 1] + \
                line[first_comma_index + 1:].replace(',', '')
        else:
            modified_line = line

        outfile.write(modified_line + '\n')

print(f"Data extracted and modified, and saved to {output_file}.")
