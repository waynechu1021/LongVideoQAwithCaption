import json
import csv
from tqdm import tqdm
import re
import argparse

def convert(args):
    pred = json.load(open(args.pred_file))
    converted = []
    converted.append(['q_uid','answer'])
    idx = 0
    for item in tqdm(pred):
        idx += 1
        if item['answer'].startswith('Answer: ('):
            pred_list = item['answer'].lower().split(' ')
            pred_option, pred_content = pred_list[1][1], ' '.join(pred_list[1:])
        elif item['answer'].startswith('('):
            pred_option = re.findall(r"\((.*?)\)", item['answer'].lower())
            pred_option = pred_option[0]
        elif item['answer'][0] in ['A','B','C','D','E']:
            pred_option = item['answer'][0].lower()
        pred_option = ord(pred_option) - ord('a')
        assert pred_option >= 0 and pred_option<= 4
        converted.append([item['question uid'], pred_option])
    with open(args.output_file,'w',newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerows(converted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    convert(args)