
import json
import re
import llm.prompts
from llm.prompt_execution import generate_response
import numpy as np
from sentence_transformers import CrossEncoder

def cross_encoder_rerank_documents(documents, user_prompt, output_prompt, model_name, k):
    model = CrossEncoder(model_name)
    cross_scores_ques = model.predict([[user_prompt, item] for item in documents],apply_softmax = True, convert_to_numpy = True )
    cross_scores_ans = model.predict([[output_prompt, item] for item in documents],apply_softmax = True, convert_to_numpy = True  )
                                    
    top_indices_ques = cross_scores_ques.argsort()[-k:][::-1]
    top_values_ques = cross_scores_ques[top_indices_ques]
    sub_context = []
    for idx in list(top_indices_ques):
        sub_context.append(documents[idx])

    top_indices_ans = cross_scores_ans.argsort()[-k:][::-1]
    top_values_ans = cross_scores_ques[top_indices_ans]
    for idx in list(top_indices_ans):
        sub_context.append(documents[idx])

    combined_list_horizontal_indices = list(np.concatenate((top_indices_ans, top_indices_ques), axis=0))
    combined_list_horizontal_values = list(np.concatenate((top_values_ans, top_values_ques), axis=0))
                                    
    unique_dict = {}
    unique_values = []
    unique_score = []

    for i, value in enumerate(combined_list_horizontal_values):
        if value not in unique_dict:
            unique_dict[value] = i
            unique_values.append(documents[i])
            unique_score.append(value)

            # Print the unique lists
            print("Unique unique_values:", unique_values)
            print("Unique unique_score:", unique_score)

    return unique_values

def llm_rerank_documents(documents, question, deployment_name, temperature):
    rerank_context = ""
    for index, docs in enumerate(documents):
        rerank_context += "\ndocument " + str(index) + ":\n"
        rerank_context += docs + "\n"


    prompt = f"""
        Let's try this now:
        {rerank_context}
        Question: {question}
    """

    response1= generate_response(llm.prompts.rerank_prompt_instruction,prompt,deployment_name, temperature)
    print(response1)
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    try:
        matches = re.findall(pattern, response1)[0]
        response_dict = json.loads( matches )
        print(response_dict)
        return response_dict
    except:
        return None