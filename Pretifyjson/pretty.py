import json
import os

def format_json(input_json, output_json):
    try:
        # Read the ugly JSON from the input file
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_json), exist_ok=True)

        # Write the pretty formatted JSON to the output file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Formatted JSON has been saved to {output_json}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    input_json = "val.story-in-sequence.json"  # Input JSON file (ugly JSON)

    # Generate the output file name by adding 'pretty_' prefix and saving in the 'Converted_json' folder
    base_name = os.path.basename(input_json)
    file_name, file_extension = os.path.splitext(base_name)
    output_json = os.path.join("Converted_json", f"pretty_{file_name}{file_extension}")  # Output file path

    # Call the function to format the JSON
    format_json(input_json, output_json)
