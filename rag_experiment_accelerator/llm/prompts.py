main_prompt_instruction = """You provide answers to questions based on information available. You give precise answers to the question asked. 
You do not answer more than what is needed. You are always exact to the point. You Answer the question using the provided context.
If the answer is not contained within the given context, say 'I dont know.'. The below context is an excerpt from a report or data.
Answer the user question using only the data provided in the sources below. Each sentence or paragraph within the context has a filename. 
It is absolutely mandatory and non-compromising to add the filenames in your response when you use those sentences and paragraphs for your final response.

context:
"""

prompt_instruction_title = "Identify and provide an appropriate title for the given user text in a single sentence with not more than 10-15 words. Do not provide output in list format and do not output any additional text and metadata."

prompt_instruction_keywords = (
    "provide unique keywords for the given user text. Format as comma separated values."
)

prompt_instruction_summary = "Summarize the given user text in a single sentence using few words. Do not provide output using multiple sentences or as a list."

prompt_instruction_entities = "Identify the key entities (person, organization, location, date, year, brand, geography, proper nouns, month etc) with context for the given user text. Format as comma separated and output only the entities. Do not provide output in list format and do not output any additional text and metadata."


generate_qna_instruction_system_prompt = """you are a prompt creator and have ability to generate new JSON prompts based on the given CONTEXT.
Generate 1 most relevant new prompt in valid json format according to "RESPONSE SCHEMA EXAMPLE" completely from scratch.
"RESPONSE SCHEMA EXAMPLE":
[
    {
        "role: "user", 
        "content": "This is the generated prompt text",
    },
    {
        "role: "assistant",
        "content": "the expected, rightful and correct answer for the question"
    },
]
"""
generate_qna_instruction_user_prompt = """The response must be valid JSON array containing two objects. The first object must contain the keys "role" and "content". The second object must also contain the keys "role" and "content".
    The response must follow the "RESPONSE SCHEMA EXAMPLE".
    The most important thing is that the response must be valid JSON ARRAY. DO NOT include anything other than valid schema.

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

do_need_multiple_prompt_instruction1 = """classify the given question into either 'HIGH' or 'LOW'. If the question must absolutely be broken down into smaller questions to search for an answer because it is not straightforward then provide a single word response as 'HIGH' and if the question is not complex and straightforward then provide a single word response as 'LOW'. Do not generate any other text apart from 'YES' or 'NO' and this is non-compromising requirement. 
    e.g. 
    How was Ritesh Modi life different before, during, and after YC?
    HIGH
    
    who is Ritesh Modi?
    LOW
    
    compare revenue for last 2 quarters?
    HIGH
    
    what was the revenue of a company last quarter?
    LOW
        
    what is the capital of Denmark?
    LOW
    """

do_need_multiple_prompt_instruction = """Classify the complexity of a given question as either 'high' or 'low' based on the following criteria. Assume all and complete information on all topics and subjects is already available to you to answer the question. No additional input or information is required apart from the contextual information you already have. You have all the information already available to answer the question. Do not output anything apart from either high or low.

Questions that require critical thinking, detailed understanding of any topic and subject, are considered 'low' in complexity.
Questions such as statistical analytics are considered 'high' in complexity.

"""

multiple_prompt_instruction = """Generate two questions in json format based on given schema for user question if it needs multiple questions to search relevant, complete and comprehensive answer. Always generate accurate json without any compromise. The output absolutely must only contains json and nothing apart from json. This is a non-compromising requirement.
    schema: 
    {
        questions:[ "question1", "question2", "questionN" ]
    }
"""

llm_answer_relevance_instruction = """Generate question for the given answer.
Answer: The PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
"""

llm_context_precision_instruction = "Given a question and a context, verify if the information in the given context is useful in answering the question. Return a Yes/No answer."

llm_answer_correctness_instruction = """Extract statements following from a given question and ground truth answer:
Question:What powers the sun and what is its primary function?
Answer: The sun is powered by nuclear fission, similar to nuclear reactors on Earth, and its primary function is to provide light to the solar system.
Ground truth: The sun is actually powered by nuclear fusion, not fission. In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy. This energy is what lights up the sun and provides heat and light, essential for life on Earth. The sun's light also plays a critical role in Earth's climate system and helps to drive the weather and ocean currents.
Extracted statements:
[
{{
  "statements that are present in both the answer and the ground truth": ["The sun's primary function is to provide light"],
  "statements present in the answer but not found in the ground truth": ["The sun is powered by nuclear fission", "similar to nuclear reactors on Earth"],
  "relevant statements found in the ground truth but omitted in the answer": ["The sun is powered by nuclear fusion, not fission", "In its core, hydrogen atoms fuse to form helium, releasing a tremendous amount of energy", "This energy provides heat and light, essential for life on Earth", "The sun's light plays a critical role in Earth's climate system", "The sun helps to drive the weather and ocean currents"]
}}
]
"""

llm_context_recall_instruction =  """ Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be 
attributed to the given context or not. 

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

llm_faithfulness_statements_instruction = """ Create one or more statements from each sentence in the given answer.

question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements in json:
{{
    "statements": [
        "Albert Einstein was born in Germany.",
        "Albert Einstein was best known for his theory of relativity."
    ]
}}

question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements in json:
{{
    "statements": [
        "Cadmium Chloride is slightly soluble in alcohol."
    ]
}}

question: Were Hitler and Benito Mussolini of the same nationality?
answer: Sorry, I can't provide answer to that question.
statements in json:
{{
    "statements": []
}}

question:{question}
answer: {answer}
statements in json:"""


llm_faithfulness_nli_instruction =  """  Natural language inference. Use only "Yes" (1) or "No" (0) as a binary verdict.
Context:
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statement_1: John is majoring in Biology.
statement_2: John is taking a course on Artificial Intelligence. 
statement_3: John is a dedicated student. 
statement_4: John has a part-time job.
Answer:
[
    {{
        "statement_1": "John is majoring in Biology.",
        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
        "verdict": "0"
    }},
    {{
        "statement_2": "John is taking a course on Artificial Intelligence.",
        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
        "verdict": "0"
    }},
    {{
        "statement_3": "John is a dedicated student.",
        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
        "verdict": "1"
    }},
    {{
        "statement_4": "John has a part-time job.",
        "reason": "There is no information given in the context about John having a part-time job.",
        "verdict": "0"
    }}
]

Context:
Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.
statement_1: Albert Einstein was a genius.
Answer:
[
     {{
        "statement_1": "Albert Einstein was a genius.",
        "reason": "The context and statement are unrelated"
        "verdict": "0"
    }}
]

Context:
Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.
statement_1: Nil
Answer:
[
     {{
        "statement_1": "Nil",
        "reason": "The statement is invalid",
        "verdict": "0"
    }}
]

context:
{context}
statements:
{statements}
Answer:
"""  