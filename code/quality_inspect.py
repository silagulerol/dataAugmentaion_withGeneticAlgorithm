import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from jiwer import wer
import pandas as pd

#################################
# 1) MODELLERİ YÜKLE
#################################
print("Loading models...")
lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_model = AutoModelForCausalLM.from_pretrained("gpt2")
lm_model.eval()

st_model = SentenceTransformer("all-MiniLM-L6-v2")

#################################
# 2) METRİK HESAPLAMA FONKSİYONU
#################################
def compute_metrics(original_text, augmented_text):
    """
    1) Semantic Similarity (Sentence-BERT)
    2) Perplexity (GPT-2)
    3) BLEU Score
    4) ROUGE (1,2,L) F-measure
    5) Word Error Rate (WER)
    6) Token-based Precision-Recall-F1
    7) Token-based 'Accuracy' 
    """
    metrics = {}

    # --------- 1) Semantic Similarity ---------
    emb1 = st_model.encode(original_text, convert_to_tensor=True)
    emb2 = st_model.encode(augmented_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    metrics["semantic_similarity"] = similarity

    # --------- 2) Perplexity (GPT-2) ---------
    with torch.no_grad():
        inputs = lm_tokenizer(augmented_text, return_tensors="pt", truncation=True, max_length=1024)
        outputs = lm_model(**inputs, labels=inputs["input_ids"])
        ppl = torch.exp(outputs.loss).item()
    metrics["perplexity"] = ppl

    # --------- 3) BLEU Score ---------
    ref_tokens = original_text.split()
    hyp_tokens = augmented_text.split()
    bleu = sentence_bleu([ref_tokens], hyp_tokens)
    metrics["bleu"] = bleu

    # --------- 4) ROUGE (1, 2, L) ---------
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(original_text, augmented_text)
    metrics["rouge1"] = rouge_scores["rouge1"].fmeasure
    metrics["rouge2"] = rouge_scores["rouge2"].fmeasure
    metrics["rougeL"] = rouge_scores["rougeL"].fmeasure

    # --------- 5) Word Error Rate (WER) ---------
    metrics["wer"] = wer(original_text, augmented_text)

    # --------- 6) Token-based Precision, Recall, F1 (BOW) ---------
    orig_set = set(ref_tokens)
    aug_set = set(hyp_tokens)
    overlap = len(orig_set.intersection(aug_set))

    precision = overlap / len(aug_set) if len(aug_set) > 0 else 0
    recall = overlap / len(orig_set) if len(orig_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics["token_precision"] = precision
    metrics["token_recall"] = recall
    metrics["token_f1"] = f1

    # --------- 7) Basit 'Accuracy' (BOW) ---------
    # Orijinal kelimelerin ne kadarını augment cümle içeriyor?
    metrics["token_accuracy"] = overlap / len(orig_set) if len(orig_set) > 0 else 0

    return metrics

#################################
# 3) MAIN
#################################
def main():
    # Her orijinal cümle için tam 5 augment cümlesi varsayıyoruz
    AUGMENTS_PER_SENTENCE = 9

    # A) Orijinal cümleleri oku
    with open("data/train_imdb.txt", "r", encoding="utf-8") as f:
        original_sentences = [line.strip().split("\t")[-1] for line in f if line.strip()]
        original_sentences = original_sentences[:50]

    # B) Augment cümleleri oku (5 katı satır olabilir)
    with open("augmented_imdb.txt", "r", encoding="utf-8") as f:
        augmented_sentences = [line.strip().split("\t")[-1] for line in f if line.strip()]
        augmented_sentences= augmented_sentences[:450]

    # C) Basit kontrol
    n_orig = len(original_sentences)
    n_aug = len(augmented_sentences)
    expected_aug = n_orig * AUGMENTS_PER_SENTENCE

    if n_aug < expected_aug:
        print(f"Uyarı! {n_orig} orijinal cümle var ama augment satır sayısı beklenenden az.\n"
              f"Beklenen: {expected_aug}, Bulunan: {n_aug}. Yine de elde olan kadar işlenecek.")
    elif n_aug > expected_aug:
        print(f"Uyarı! {n_aug} augment satırı var, beklenen {expected_aug}. Fazla satırları yok sayacağız.")

    # min_len = min(len(original_sentences), len(augmented_sentences))
    # Bu mantık 1'e 1 eşleştiriyordu. Artık her 5 augment'ı 1 orijinal cümleyle eşleştireceğiz:

    results = []
    for i, orig in enumerate(original_sentences):
        # i. orijinale denk gelen augment satırları [i*5, i*5+1, i*5+2, i*5+3, i*5+4]
        start_idx = i * AUGMENTS_PER_SENTENCE
        end_idx = start_idx + AUGMENTS_PER_SENTENCE
        if start_idx >= n_aug:
            break  # augment cümle kalmadı

        # Döngü: 5 augment cümlesi
        for j in range(start_idx, min(end_idx, n_aug)):
            aug = augmented_sentences[j]
            row_metrics = compute_metrics(orig, aug)
            row_metrics["pair_id"] = i
            row_metrics["original"] = orig
            row_metrics["augmented"] = aug
            # augment cümlenin i. orijinale karşılık 0..4. alt-örneği
            row_metrics["aug_index"] = j - start_idx
            results.append(row_metrics)

    total = 0
    for result in results:
        total += result["semantic_similarity"]
    
    mean= total / len(results)
    print(f"Ortalama benzerlik: {mean:.4f}")
    # DataFrame ve Excel çıkışı
    #df = pd.DataFrame(results)
    #df.to_excel("augmentation_metrics.xlsx", index=False)
    #print(f"✓ {len(df)} satır metrik hesaplandı ve augmentation_metrics.xlsx dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
