import json
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm  # Import tqdm for progress bar

# Load BERT for cosine similarity (optional)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def calculate_bleu_score(actual_caption, generated_caption):
    # Tokenize the sentences
    actual_tokens = actual_caption.split()
    generated_tokens = generated_caption.split()

    # Calculate BLEU score
    return sentence_bleu([actual_tokens], generated_tokens)

def calculate_cosine_similarity(actual_caption, generated_caption):
    # Tokenize and convert to BERT embeddings
    actual_inputs = tokenizer(actual_caption, return_tensors="pt", padding=True, truncation=True)
    generated_inputs = tokenizer(generated_caption, return_tensors="pt", padding=True, truncation=True)

    # Get the embeddings from BERT model
    actual_embeddings = bert_model(**actual_inputs).last_hidden_state.mean(dim=1)
    generated_embeddings = bert_model(**generated_inputs).last_hidden_state.mean(dim=1)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(actual_embeddings.detach().numpy(), generated_embeddings.detach().numpy())
    return cosine_sim[0][0]

def measure_scores(actual_caption, generated_caption):
    bleu_score = calculate_bleu_score(actual_caption, generated_caption)
    cosine_score = calculate_cosine_similarity(actual_caption, generated_caption)

    # Convert any NumPy float32 types to regular Python float for JSON serialization
    return {
        "bleu_score": float(bleu_score),
        "cosine_score": float(cosine_score)
    }

def process_scores(input_json, output_json_with_scores):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data  # Assuming the json data is already loaded
    total_images = len(images)
    
    print(f"Total images to process: {total_images}\n")  # Show total images
    
    results = []
    for i, item in enumerate(tqdm(images, desc="Processing Captions", unit="img", ncols=100)):
        img_url = item.get("image_url")  
        actual_caption = item.get("actual_caption", "Unknown Caption")
        generated_caption = item.get("generated_caption", "Unknown Caption")

        scores = measure_scores(actual_caption, generated_caption)
        
        results.append({
            "index": i + 1,  # Add the index starting from 1
            "image_url": img_url,
            "actual_caption": actual_caption,
            "generated_caption": generated_caption,
            "scores": scores
        })

        # Custom progress update
        remaining_images = total_images - (i + 1)
        print(f"[{i+1}/{total_images}] Processed: {img_url} - {remaining_images} images left")

    with open(output_json_with_scores, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nProcessing complete! Results with scores saved to", output_json_with_scores)

# Example usage
process_scores("output.json", "output_with_scores.json")
