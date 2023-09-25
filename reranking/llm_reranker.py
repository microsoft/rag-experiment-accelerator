
import json
import re
import llm.prompts
from llm.prompt_execution import generate_response


def rerank_documents(documents, question, deployment_name):

    rerank_context = ""
    for index, docs in enumerate(documents):
        rerank_context += "\ndocument " + str(index) + ":\n"
        rerank_context += docs + "\n"


    prompt = f"""
        Let's try this now:
        {rerank_context}
        Question: {question}
    """

    response1= generate_response(llm.prompts.rerank_prompt_instruction,prompt,deployment_name)
    print(response1)
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    try:
        matches = re.findall(pattern, response1)[0]
        response_dict = json.loads( matches )
        print(response_dict)
        return response_dict
    except:
        return None