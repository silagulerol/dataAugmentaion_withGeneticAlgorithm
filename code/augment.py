# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *

##arguments to be parsed from command line
import argparse
#ArgumentParser: Object for parsing command line strings into Python objects.
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

##the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#---------- In this part of code we update our parameters due to command input

##number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

##how much to replace each word by synonyms
alpha_sr = 0.1 #default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

##how much to insert new words that are synonyms
alpha_ri = 0.1 #default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

##how much to swap words
alpha_rs = 0.1 #default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

##how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

#------------------

##generate more data with standard augmentation

#train_orig is my input file
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    # output_file opens in writing mode
    # input_file opens in reading mode
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r', encoding='utf-8').readlines() 
    # .readlines(): Return a list of lines from the stream.

    # now every line's structure is label\tsentence. We need to craete "parts" list 
    # which includes current line's label as first element and current line's sentence as second element
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        label = parts[0]
        sentence = parts[1]
        
        """
        #for trec dataset- label:text
        parts= line[:-1].split(":")
        label=parts[0]
        sentence= "".join(parts[1:])
        """

        """
        for emotions dataset- id,text,label
        parts = line.split(",")
        label=parts[-1]
        sentence="".join(parts[1:-1])
        """

        # after we create label and sentence parts, we use them as argument in eda function in eda.py file
        # eda function take only one sentce as input and returns a list which includes 9 sentences that are augmentations of current sentence
        # each element in aug_sentences is an augmented sentence of current sentence
        aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(label + "\t" + aug_sentence + '\n')

    # we close the output_file
    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))


##main function
if __name__ == "__main__":

    ##generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)