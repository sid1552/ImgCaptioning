import json
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm  # Import tqdm for progress bar

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def download_image(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return Image.open(response.raw).convert("RGB")
    return None

def generate_caption(img):
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def process_images(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]  # Get the list of images
    total_images = len(images)
    
    print(f"Total images to process: {total_images}\n")  # Show total images
    
    results = []
    for i, item in enumerate(tqdm(images, desc="Processing Images", unit="img")):
        img_url = item.get("url_o")  
        title = item.get("title", "Unknown Caption")  

        if img_url:
            img = download_image(img_url)
            if img:
                caption = generate_caption(img)
                results.append({"image_url": img_url, "actual_caption": title, "generated_caption": caption})
            else:
                results.append({"image_url": img_url, "actual_caption": title, "generated_caption": "Failed to download image"})

        # Print progress manually as well
        print(f"[{i+1}/{total_images}] Processed: {img_url}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nProcessing complete! Results saved to", output_json)

# Example usage
process_images("100_images_pretty_train.description-in-isolation.json", "output.json")
