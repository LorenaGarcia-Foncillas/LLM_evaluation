import json
import os
import pandas as pd
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
_LOGGER = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Label Studio JSON annotations')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to Label Studio JSON file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder for converted annotations')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress verbose output')
    return parser.parse_args()

def extract_text_and_annotations(json_file_path, output_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    txt_folder = os.path.join(output_path, 'txt')
    ann_folder = os.path.join(output_path, 'ann')

    os.makedirs(txt_folder, exist_ok=True)
    os.makedirs(ann_folder, exist_ok=True)

    # Loop through json 
    for number, item in enumerate(data):
        # if number < 5:
            # doc_id = str(item['id'])
            doc_id = item["file_upload"].split("-")[1].split(".")[0]
            # Get the Textual Observation
            text = item['data']['text']
            # Save text to txt/ folder
            with open(os.path.join(txt_folder, f"{doc_id}.txt"), 'w') as txt_file:
                txt_file.write(text)
            # Initialize annotation_id
            annotation_id = 1
            # Loop through list of annotations
            for annotation in item['annotations']:
                # Create empty list
                csv_rows = []
                # Loop through list of results
                for result in annotation['result']:
                    # # Check if label TUMOUR (convention used during labelling)
                    # if "TUMOUR" in result['value']['labels'][0]:
                        # Extract annotation details (modify keys based on actual structure)
                        phi_type = result['value']['labels'][0]
                        start_idx = result['value']['start']
                        stop_idx = result['value']['end']
                        entity = text[start_idx:stop_idx]
                        csv_rows.append([doc_id, annotation_id, start_idx, stop_idx, entity, phi_type])
                        annotation_id += 1

            # Save annotations to ann/ folder
            df_out = pd.DataFrame(csv_rows, columns=['document_id', 'annotation_id', 'start', 'stop', 'entity', 'entity_type'])
            df_out.to_csv(os.path.join(ann_folder, f"{doc_id}.gs"), index=False)


def main():
    args = parse_args()

    if not args.quiet:
        _LOGGER.setLevel(logging.INFO)

    input_path = args.input
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    _LOGGER.info(f'Converting annotations from {input_path} to {output_path}')
    
    extract_text_and_annotations(input_path, output_path)

    _LOGGER.info('Conversion completed successfully.')

if __name__ == '__main__':
    main()

"""
This can be run: 

    python convert_label_studio_to_gs.py -i 400_reports_labelled_25July2024.json -o Data/
"""