main_prompt_instruction = """You provide answers to questions based on information available. You give precise answers to the question asked.
You do not answer more than what is needed. You are always exact to the point. You Answer the question using the provided context.
If the answer is not contained within the given context, say 'I dont know.'. The below context is an excerpt from a report or data.
Answer the user question using only the data provided in the sources below. Each sentence or paragraph within the context has a filename.
It is absolutely mandatory and non-compromising to add the filenames in your response when you use those sentences and paragraphs for your final response.

context:
"""

prompt_instruction_title = """Identify and provide an appropriate title for the given user text in a
    single sentence with not more than 10-15 words. Do not provide output in
    list format and do not output any additional text and metadata."""

prompt_instruction_keywords = (
    "provide unique keywords for the given user text. Format as comma separated values."
)

prompt_instruction_summary = """Summarize the given user text in a single sentence using few words. Do
    not provide output using multiple sentences or as a list."""

prompt_instruction_entities = """Identify the key entities (person, organization, location, date, year,
    brand, geography, proper nouns, month etc) with context for the given
    user text. Format as comma separated and output only the entities. Do
    not provide output in list format and do not output any additional text
    and metadata."""


prompt_generated_hypothetical_answer = "You are a helpful expert research assistant. Provide an example answer to the given question, that might be found in a document."

prompt_generated_hypothetical_document_to_answer = "You are a helpful expert research assistant. Write a scientific paper to answer to the given question."

prompt_generated_related_questions = """You are a helpful expert research assistant. Your users are asking questions
  Suggest up to five additional related questions to help them find the information they need, for the provided question.
  Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
  Make sure they are complete questions, and that they are related to the original question.
  Output one question per line. Do not number the questions."""

generate_qna_instruction_system_prompt = """
you are a quiz creator and have ability to generate human like question-answer pairs from given CONTEXT.
"""
generate_qna_instruction_user_prompt = """
Please read the following CONTEXT and generate two question and answer json objects in an array based on the CONTEXT provided. The questions should require deep reading comprehension, logical inference, deduction, and connecting ideas across the text. Avoid simplistic retrieval or pattern matching questions. Instead, focus on questions that test the ability to reason about the text in complex ways, draw subtle conclusions, and combine multiple pieces of information to arrive at an answer. Ensure that the questions are relevant, specific, and cover the key points of the CONTEXT.  Provide concise answers to each question, directly quoting the text from provided context. Provide the array output in strict JSON format as shown in output format. Ensure that the generated JSON is 100 percent structurally correct, with proper nesting, comma placement, and quotation marks. There should not be any comma after last element in the array.

Output format:
[
  {
    "question": "Question 1",
    "answer": "Answer 1"
  },
  {
    "question": "Question 2",
    "answer": "Answer 2"
  }
]

CONTEXT:
"""

rerank_prompt_instruction = """A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided.
    Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well
    as the relevance score as json string based on json format as shown in the schema section. The relevance score is a number from 1–10 based on how relevant you think the document is to the question. The relevance score can be repetitive. Do not output any
    additional text or explanation or metadata apart from json string. Just output the json string and strip rest every other text. Strictly remove any last comma from the nested json elements if it is present.
    Do not include any documents that are not relevant to the question. There should exactly be one documents element.
    Example format:
    Document 1:
    content of document 1
    Document 2:
    content of document 2
    Document 3:
    content of document 3
    Document 4:
    content of document 4
    Document 5:
    content of document 5
    Document 6:
    content of document 6
    Question: user defined question

    schema:
        {
        "documents" :{
            "document_1": "Relevance"
            }
        }
    """

do_need_multiple_prompt_instruction = """
Consider the given question to analyze and determine if it falls into one of these categories:

1. Simple, factual question
  a. The question is asking for a straightforward fact or piece of information
  b. The answer could likely be found stated directly in a single passage of a relevant document
  c. Breaking the question down further is unlikely to be beneficial
  Examples: "What year did World War 2 end?", "What is the capital of France?, "What is the features of productX?"
2. Complex, multi-part question
  a. The question has multiple distinct components or is asking for information about several related topics
  b. Different parts of the question would likely need to be answered by separate passages or documents
  c. Breaking the question down into sub-questions for each component would allow for better results
  d. The question is open-ended and likely to have a complex or nuanced answer
  e. Answering it may require synthesizing information from multiple sources
  f. The question may not have a single definitive answer and could warrant analysis from multiple angles
  Examples: "What were the key causes, major battles, and outcomes of the American Revolutionary War?", "How do electric cars work and how do they compare to gas-powered vehicles?"

Based on this rubric, does the given question fall under category 1 (simple) or category 2 (complex)? The output should be in strict JSON format. Ensure that the generated JSON is 100 percent structurally correct, with proper nesting, comma placement, and quotation marks. There should not be any comma after last element in the JSON.

Example output:
{
  "category": "simple"
}
"""

multiple_prompt_instruction = """Your task is to take a question as input and generate maximum 3 sub-questions that cover all aspects of the original question. The output should be in strict JSON format, with the sub-questions contained in an array.

Here are the requirements:
1. Analyze the original question and identify the key aspects or components.
2. Generate sub-questions that address each aspect of the original question.
3. Ensure that the sub-questions collectively cover the entire scope of the original question.
4. Format the output as a JSON object with a single key "questions" that contains an array of the generated sub-questions.
5. Each sub-question should be a string within the "questions" array.
6. The JSON output should be valid and strictly formatted.
7. Ensure that the generated JSON is 100 percent structurally correct, with proper nesting, comma placement, and quotation marks. The JSON should be formatted with proper indentation for readability.
8. There should not be any comma after last element in the array.


Example input question:
What are the main causes of deforestation, and how can it be mitigated?

Example output:
{
  "questions": [
    "What are the primary human activities that contribute to deforestation?",
    "How does agriculture play a role in deforestation?",
    "What is the impact of logging and timber harvesting on deforestation?",
    "How do urbanization and infrastructure development contribute to deforestation?",
    "What are the environmental consequences of deforestation?",
    "What are some effective strategies for reducing deforestation?",
    "How can reforestation and afforestation help mitigate the effects of deforestation?",
    "What role can governments and policies play in preventing deforestation?",
    "How can individuals and communities contribute to reducing deforestation?"
  ]
}
"""

llm_answer_relevance_instruction = """Generate question for the given answer.
Answer: The PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
"""

llm_context_precision_instruction = "Given a question and a context, verify if the information in the given context is useful in answering the question. Return a Yes/No answer."

# Context recall prompt taken from https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_context_recall.py
# Copyright [2023] [Exploding Gradients]
# under the Apache License (see evaluation folder)
llm_context_recall_instruction = """ Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not.

question: What can you tell me about albert Albert Einstein?
context: Albert Einstein (14 March 1879 to 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895
[
    {{  "statement_1":"Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
        "reason": "The date of birth of Einstein is mentioned clearly in the context.",
        "Attributed": "1"
    }},
    {{
        "statement_2":"He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics.",
        "reason": "The exact sentence is present in the given context.",
        "Attributed": "1"
    }},
    {{
        "statement_3": "He published 4 papers in 1905.",
        "reason": "There is no mention about papers he wrote in the given context.",
        "Attributed": "0"
    }},
    {{
        "statement_4":"Einstein moved to Switzerland in 1895.",
        "reason": "There is no supporting evidence for this in the given context.",
        "Attributed": "0"
    }}
]

question: who won 2020 icc world cup?
context: Who won the 2022 ICC Men's T20 World Cup? The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.
answer: England
[
    {{
        "statement_1":"England won the 2022 ICC Men's T20 World Cup.",
        "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
         "Attributed": "1"
    }}
]
"""
