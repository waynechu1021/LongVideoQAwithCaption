import json, os, argparse
from tqdm import tqdm

qtype = "multi-choice"

def print_result(eval_results):
    with open("playground/TempCompass/meta_info.json", 'r') as f:
        meta_infos = json.load(f)
    match_rate = 0  # the success rate of rule-based answer matching
    result_asp = {'action': 0, 'direction': 0, 'speed': 0, 'order': 0, 'attribute_change': 0}   # eval result under every temporal aspect
    qcount_asp = {'action': 0, 'direction': 0, 'speed': 0, 'order': 0, 'attribute_change': 0}   # question count result under every temporal aspect
    result_fasp = {'fine-grained action': 0, 'coarse-grained action': 0, 'object motion': 0, 'camera motion': 0, 
                   'absolute speed': 0, 'relative speed': 0, 'order': 0, 'color & light change': 0, 'size & shape change': 0, 'combined change': 0, 'other change': 0}  # eval result under every fine-grained temporal aspect
    qcount_fasp = {'fine-grained action': 0, 'coarse-grained action': 0, 'object motion': 0, 'camera motion': 0, 
                   'absolute speed': 0, 'relative speed': 0, 'order': 0, 'color & light change': 0, 'size & shape change': 0, 'combined change': 0, 'other change': 0}  # question count result under every fine-grained temporal aspect
    
    for id in eval_results:
        for asp in eval_results[id]:
            fasp = meta_infos[id.replace('.jpg', '').replace('.mp4', '')]["eval_dim"][asp]["type"] if asp!="order" else "order"
            for result in eval_results[id][asp]:
                result_asp[asp] += result["rating"]
                result_fasp[fasp] += result["rating"]
                qcount_asp[asp] += 1
                qcount_fasp[fasp] += 1
                if "match_success" in result:
                    match_rate += result["match_success"]

    match_rate = round(match_rate/sum(qcount_asp.values())*100, 1)
    result_asp['avg'] = round(sum(result_asp.values())*100/sum(qcount_asp.values()), 1)
    for asp in result_asp:
        if asp!='avg':
            result_asp[asp] = round(result_asp[asp]*100/qcount_asp[asp], 1)
    for fasp in result_fasp:
        result_fasp[fasp] = round(result_fasp[fasp]*100/qcount_fasp[fasp], 1)
    print("Accuracy Results:")
    print(result_asp)
    print(result_fasp)
    print(f"Match Success Rate={match_rate}")


def main(predictions, eval_results, output_file, disable_llm):
    for id in tqdm(predictions):

        if id not in eval_results:
            eval_results[id] = {}

        for dim, preds in predictions[id].items():

            if dim in eval_results[id] and eval_results[id][dim] and len(preds)==len(eval_results[id][dim]):    # skip if the eval result already exists
                continue
            eval_results[id][dim] = []
            
            for pred in preds:
                if "prediction" not in pred and "response" in pred:
                    pred["prediction"] = pred["response"]

                if pred["prediction"] is None:  # In some cases the Video LLM may refuse to produce a response
                    eval_result = {"question": pred["question"], "gt-answer": pred["answer"], "video-llm-prediction": pred["prediction"], "match_success": False, "rating": 0}
                    eval_results[id][dim].append(eval_result)
                    continue

                pred["prediction"] = pred["prediction"].replace('</s>', '').strip()
                eval_result = {"question": pred["question"], "gt-answer": pred["answer"], "video-llm-prediction": pred["prediction"], "match_success": True}

                # Some hand-crafted matching rules
                if pred["prediction"]==pred["answer"]:
                    eval_result["rating"] = 1
                elif pred["prediction"] in ["A", "B", "C", "D"]:
                    eval_result["rating"] = 1 if pred["prediction"]==pred["answer"][0] else 0
                elif any(pred["prediction"].startswith(prefix) for prefix in ['Answer: (A)', 'Answer: (B)', 'Answer: (C)', 'Answer: (D)']):
                    eval_result["rating"] = 1 if pred["prediction"].split(' ')[1][1]==pred["answer"][0] else 0
                elif any(pred["prediction"].startswith(prefix) for prefix in ['A.', 'B.', 'C.', 'D.']):
                    eval_result["rating"] = 1 if pred["prediction"].split('.')[0]==pred["answer"][0] else 0
                elif any(pred["prediction"].startswith(prefix) for prefix in ['A)', 'B)', 'C)', 'D)']):
                    eval_result["rating"] = 1 if pred["prediction"].split(')')[0]==pred["answer"][0] else 0
                elif disable_llm:
                    eval_result["match_success"] = False    
                    eval_result["rating"] = 0               # Fail to match answer in the video-llm response. Directly set rating to 0
                else:
                    raise NotImplementedError

                eval_results[id][dim].append(eval_result)

    with open(os.path.expanduser(output_file), "w") as f:
        json.dump(eval_results, f, indent=4)
    
    print_result(eval_results)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default="video-llava")
    parser.add_argument('--output_file', default="video-llava")
    parser.add_argument('--disable_llm', action='store_true', help="Whether to disable llm evaluation")
    args = parser.parse_args()

    disable_suffix = "_disable_llm" if args.disable_llm else ""
    input_file = f"{args.pred_file}"
    output_file = f"{args.output_file}"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Loading video-llm predictions and multi-choice questions
    with open(input_file, 'r') as f:
        predictions = json.load(f)

    # Loading already evaluated results
    # if os.path.isfile(output_file):
    #     with open(output_file, 'r') as f:
    #         eval_results = json.load(f)
    # else:
    #     eval_results = {}
    eval_results = {}

    main(predictions, eval_results, output_file, args.disable_llm)