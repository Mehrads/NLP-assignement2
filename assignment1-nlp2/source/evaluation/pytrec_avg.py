import json

import json

def pytrec_avg_evaluation(wordsList, name, evaluation_method):
    array_cosine = {"cosine": []}
    sum_cosine_ndcg = 0
    num = 0

    for g in wordsList:  # Iterate over each word in the list
        sample = str(g)
        try:
            with open(f'output/pytrec/pytrec_eval_per_word/{name}/pytrec_result_perword/pytrec_{sample}.json') as k:
                result = json.load(k)
            # Assuming the JSON structure allows directly accessing the required NDCG value
            sum_cosine_ndcg += result[0]["cosine"][0][sample][0]["ndcg"][evaluation_method]["ndcg"]
            num += 1
        except FileNotFoundError:
            print(f"File not found for {sample} in {evaluation_method} evaluation.")
            # If a file for a word is not found, you might choose to continue to the next word instead of returning
            continue

    if num > 0:
        avg_cosine_ndcg = sum_cosine_ndcg / num
        array_cosine["cosine"].append({
            evaluation_method: [["ndcg :", avg_cosine_ndcg]]
        })

        # Serialize and write the average NDCG to file
        j = json.dumps([array_cosine], indent=4)
        output_file = f'output/avg/{evaluation_method}/ndcg_avg_{name}.json'
        with open(output_file, 'w') as f:
            f.write(j)

        print("Result is:", array_cosine)
    else:
        print(f"No valid data found for {name} in {evaluation_method} evaluation.")


# Example usage (commented out to prevent accidental execution):
# pytrec_avg_evaluation('word', 'modelname', 'w2v')
# pytrec_avg_evaluation('word', 'modelname', 'tf_idf')
