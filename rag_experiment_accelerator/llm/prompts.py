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
        "role: "assitant",
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
