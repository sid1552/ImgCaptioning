import json
import os

def keep_first_100_images(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Keep only the first 100 images
    data['images'] = data['images'][:100]
    
    # Remove other fields (albums, info, annotations, type)
    data = { "images": data["images"] }
    
    # Write the filtered data to the output file (in the current directory)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    input_file = "pretty_train.description-in-isolation.json"  # Change to your input file
    
    # Generate the output file name with '100_images_' prefix and save it in the current directory
    base_name = os.path.basename(input_file)
    file_name, file_extension = os.path.splitext(base_name)
    output_file = f"100_images_{file_name}{file_extension}"  # Output file path in the current directory

    # Call the function to filter the first 100 images and save to the new output file
    keep_first_100_images(input_file, output_file)
