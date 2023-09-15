import time
import os
import logging
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI

import initialize

openai_key = os.getenv('OPENAI_API_KEY')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
database = os.getenv('DB_NAME')

# connect to database
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@localhost:3306/{database}"
)


def get_time_str():
    import re, datetime as dt
    return re.sub('[-:. ]','_',dt.datetime.now().isoformat(' '))


def write_text(text, file_path='text_output/text', file_ext='txt'):
    logging.info("received transation from asr model")
    if text:
        logging.info(f"writing to file {file_path}.{file_ext}")
        with open(f"{file_path}.{file_ext}", "w") as f:
            f.write(text)
        logging.info("written")
    else:
        logging.info("no text to write")


def fetch_hns_response(query: str):
    from langchain.chains import RetrievalQA
    vector_store = initialize.vector_store_initialize()
    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    # llm = OpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-3.5-turbo")
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    query_template = """
    This is the input question: {question}

    Fetch response by taking relevant detail from the retrieved documents
    The return statement should be at least  relevant to the question asked and contain domain specific knowledge from the document embeddings.
    """
    return qa.run(query_template.format(question=query))


def fetch_general_response(question):
    query = """
    feed with the input question, first create correct mysql query with right syntax which can run, 
    secondly observe the results of the query and finally return the answer.
    Use the below format while answering:

    Input: Input Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    {question}
    """

    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    # llm = OpenAI(temperature=0, openai_api_key=openai_key, model_name="gpt-3.5-turbo")
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # set up the llm chain
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    try:
        question = query.format(question=question)
        time.sleep(1)
        return db_chain.run(question)
    except Exception as e:
        logging.error(e)
        return None


def get_translation_whisper(input_audio_file):
    from transformers import pipeline
    asr_model = pipeline("automatic-speech-recognition", "openai/whisper-large-v2")
    return asr_model(input_audio_file)['text']


def translate_to_english(input_str: str):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source='auto', target='en').translate(input_str)


def get_translation_faster_whisper(input_audio_file):
    from faster_whisper import WhisperModel
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(input_audio_file, beam_size=5)
    logging.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    final_text = ''
    for segment in segments:
        logging.info("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        final_text += translate_to_english(segment.text)
    return final_text


def get_translation_openai(input_audio_file: str):
    import openai
    from deep_translator import GoogleTranslator
    audio_file = open(input_audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    text = GoogleTranslator(source='auto', target='en').translate(transcript['text'])
    logging.info(f"translated text: {text}")
    return text


def classify_text(input_text: str):
    from langchain.chains import create_tagging_chain

    schema = {
        "properties": {
            "category": {
                "type": "string",
                "enum": ["help", "general"],
                "description": """classify if the question requires help and support, 
                or else falls under miscellaneous category
                """},
        },
        "required": ["category"],
    }
    llm_classify = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    chain = create_tagging_chain(schema, llm_classify)
    resp = chain.run(input_text)
    return resp['category']
