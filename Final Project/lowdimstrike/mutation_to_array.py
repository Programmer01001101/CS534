import csv
import numpy as np

#read rows into dictionaries
contents = []

#read in contents to list
with open( '/Users/Bill7/desktop/lowdimstrike/data_mutations_mskcc.txt', 'rt') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        contents.append(row)
  
#get list of samples that were sequenced
row_samples = contents[0][0].split(' ')
samples = list(set(row_samples))
contents.pop(0)

#get dict keys
keys = contents[0]
contents.pop(0)

#capture each row in a dictionary
entries = []
for row in contents:
    entries.append(dict(zip(keys, row)))
    
#get list of mutated genes
genes = list(set([entry['Hugo_Symbol'] for entry in entries]))
    
#generate arrays for 'Consequence', 'Variant_Classification'
consequence = []
variant = []
for i, sample in enumerate(samples):
    
    #find matches
    matches = [j for j,row_sample in enumerate(row_samples) if sample == row_sample]
    
    #initialize output row
    consequence_row = [''] * len(genes)
    variant_row = consequence_row.copy()
    
    #record each match
    for match in matches:

        #capture protein consequence
        if not consequence_row[genes.index(entries[match]['Hugo_Symbol'])]:
            consequence_row[genes.index(entries[match]['Hugo_Symbol'])] = entries[match]['Consequence']
        else:
            consequence_row[genes.index(entries[match]['Hugo_Symbol'])] = consequence_row[genes.index(entries[match]['Hugo_Symbol'])] + ',' + entries[match]['Consequence']

        #capture variant classification
        if not variant_row[genes.index(entries[match]['Hugo_Symbol'])]:
            variant_row[genes.index(entries[match]['Hugo_Symbol'])] = entries[match]['Variant_Classification']
        else:
            variant_row[genes.index(entries[match]['Hugo_Symbol'])] = variant_row[genes.index(entries[match]['Hugo_Symbol'])] + ',' + entries[match]['Variant_Classification']


    #add rows to output
    consequence.append(consequence_row)
    variant.append(variant_row)
    
#write consequence outputs to file
with open('data_mutations_consequence.txt', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['\t'] + genes)
    for i, row in enumerate(consequence):
        writer.writerow([samples[i]] + row)

#write variant outputs to file
with open('data_mutations_variant.txt', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['\t'] + genes)
    for i, row in enumerate(variant):
        writer.writerow([samples[i]] + row)
        
reader