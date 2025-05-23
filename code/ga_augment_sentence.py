import random
import nltk
from eda import eda
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, fast

from transformers import AutoModelForCausalLM, AutoTokenizer

lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Suppose I have a function get_fitness(...) that returns a numeric fitness score
# (the higher, the better). This could bebased on:
# - Semantic similarity to original
# - Grammatical correctness
# - Diversity from original, etc.

def ga_augment_sentence(
    original_sentence,
    pop_size=16,
    generations=5,
    alpha_sr=0.1,
    alpha_ri=0.1,
    alpha_rs=0.1,
    alpha_rd=0.1,
    num_aug=9,
    crossover_rate=0.5,
    mutation_rate=0.3,
):
    """
    Apply a simple GA for text augmentation using EDA as mutation operator.
    """

    # 1) Create initial population using EDA
    population=[]
    cand=[]
    cand = eda(original_sentence, alpha_sr, alpha_ri,alpha_rs, alpha_rd, num_aug=16)
    population= cand
    """
    for _ in range(pop_size):
        # Each individual is an augmented sentence from EDA
        # EDA returns multiple augmented sentences, so pick 1
        # insted of that make num_aug=16 and create cand as a list, which stores every augmented sentence
        #then assign it to population
        cand = eda(original_sentence, alpha_sr, alpha_ri,alpha_rs, alpha_rd, num_aug=1)[0]
        population.append(cand)
    """
    # 2) Evolve for 'generations' steps
    for gen in range(generations):
        #Evaluate fitness for each individaul
        scored_pop = [(indiv, get_fitness(original_sentence, indiv,
                    model=model,
                    tokenizer=lm_tokenizer,
                    lm_model=lm_model)) for indiv in population]

        # Sort descending by fitness
        scored_pop.sort(key=lambda x: x[1], reverse=True)

        # Keep the top half as parents
        half = pop_size // 2
        parents = scored_pop[:half]

        # 3) Reproduce
        new_population= []
        while len(new_population) < pop_size:
            # Selection: pick two random parents from the top half
            parent1 = random.choice(parents)[0]
            parent2 = random.choice(parents)[0]

            # Crossover: Combine parts of parent1 & parent2 (word-level or phrase-level)
            if random.random() < crossover_rate:
                child = crossover_sentences(parent1, parent2)
            else:
                child=parent1 # or parent2, or a random pick
            
            # Mutation with probability, apply one of EDA ops
            if random.random() < mutation_rate:
                # simple approach: mutate child with EDA for 1 augmentation
                # or just apply a single EDA operator like random swp, etc.
                child = eda(child, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=1)[0]

            new_population.append(child)
        
        population=new_population

    # After final generation, pick the best or return entire pop
    final_scored_pop = [(indiv,get_fitness(original_sentence, indiv,
                    model=model,
                    tokenizer=lm_tokenizer,
                    lm_model=lm_model)) for indiv in population]
    final_scored_pop.sort(key=lambda x: x[1], reverse=True)
    #e.g., take top 16
    top_augmented = [pair[0] for pair in final_scored_pop[:16]]

    return top_augmented

def crossover_sentences(sentA, sentB):
    """
    Simple word-level crossover: 
    - Split each parent's word list in half,
    - Join first half of A with second half of B
    """
    wordsA = sentA.split()
    wordsB = sentB.split()
    midA = len(wordsA) // 2
    midB = len(wordsB) // 2
    child_words = wordsA[:midA] + wordsB[midB:]
    return " ".join(child_words)


def get_fitness(original, augmented, model, tokenizer=None, lm_model=None, alpha=0.7, beta=0.3):
    """
    Fitness = alpha * semantic_similarity - beta * perplexity_penalty
    """

    # 1. Semantic similarity
    emb1 = model.encode(original, convert_to_tensor=True)
    emb2 = model.encode(augmented, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()  # range: -1 to 1

    # 2. Fluency penalty via perplexity 
    if tokenizer and lm_model:
        import torch
        inputs = tokenizer(augmented, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = lm_model(**inputs, labels=inputs["input_ids"])
        perplexity = torch.exp(outputs.loss).item()
        perplexity_penalty = min(perplexity / 100, 1.0)  # normalize
    else:
        perplexity_penalty = 0

    # 3. Combine scores
    fitness = alpha * similarity - beta * perplexity_penalty
    return fitness



"""
# In this fitness function the sentences are slightly:
def get_fitness(original, augmented):
    
    Placeholder fitness function:
    1) High semantic similarity with original => higher score
    2) Possibly penalize too much overlap (lack of diversity)
    3) Possibly add a grammar or LM-based check
    
    # For a basic example, naive similarity: 
    # count how many unique tokens overlap.
    # more advanced in practice such as: a BERT-based similarity, or something from GTR-GA paper  
    
    set_orig = set(original.split())
    set_aug = set(augmented.split())
    overlap = len(set_orig.intersection(set_aug))
    # The more words in common, the more "similar"
    # But you might also want to penalize if it's too similar. 
    # For now, let's do a very simplistic measure:
    return overlap  # Higher means more overlap
"""

"""
    # In this fitness function the sentences are same:
    emb1 = model.encode(original, convert_to_tensor=True)
    emb2 = model.encode(augmented, convert_to_tensor=True)
    sim_score = util.cos_sim(emb1, emb2).item()
    return sim_score
"""