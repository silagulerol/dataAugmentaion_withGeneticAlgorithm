# ga_augment.py
import argparse
from ga_augment_sentence import ga_augment_sentence # The code snippet above
from eda import eda # if needed

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str)
ap.add_argument("--output", required=True, type=str)
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
def ga_augment_file(input_path, output_path):
    counter = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            """
            label_text = line.strip().split("\t")
            if len(label_text) != 2:
                continue
            label, original_sentence = label_text
            """

            """
            parts= line[:-1].split(":")
            label=parts[0]
            original_sentence= "".join(parts[1:])
            """
            parts = line.split(",")
            label=parts[-1]
            original_sentence="".join(parts[1:-1])
            pop_size=16;
            top_augments = ga_augment_sentence(original_sentence,
                                               pop_size,
                                               generations=3,
                                               alpha_sr=0.1,
                                               alpha_ri=0.1,
                                               alpha_rs=0.1,
                                               alpha_rd=0.1,
                                               num_aug=1)
            
            # Write them out
            for aug in top_augments:
                fout.write(f"{label}\t{aug}\n")
            
            
            #counter+=1
            #print(counter, aug)
            #if (counter>480):
            #    break

if __name__ == "__main__":
    ga_augment_file(args.input, args.output)