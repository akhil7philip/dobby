import time
import os
import logging

from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
import initialize
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

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
    This is the input question: {question}

    If the input question has keywords like company names and asks for these following actions,
    buy or sell, show profit, returns, trend;
    then use the companies table to return the concat value of 'https://groww.in/stocks/' and 
    search_id for the relevant company name from company_name using the correct mysql query.
    The return value is a logical REST url from the above values.

    If the input question has keywords like industry average,
    then take average of all stocks corresponding to that industry given in tag column of company table
    using the correct mysql query.

    If the input question has "mutual funds" keyword, 
    then use the mutual_funds table to return the concat value of 'https://groww.in/mutual-funds/collections/' and 
    endpoint value based on the collections column using the correct mysql query.
    for example, if they ask large cap mutual fund, then collection='Large cap funds', endpoint='best-large-cap' and 
    return 'https://groww.in/mutual-funds/collections/best-large-cap'
    The return value is a logical REST url from the above values.

    If the input question depends on general financial knowledge like what, how, when about a company etc, 
    then respond to the user in less than 100 words as though the user is an intelligent person,
    but with simple financial terms that the user can understand.

    The return statement should be relevant to the question asked, using the above classification provided above.
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
        old_query = """
        This is the input question: {question}
        
        If the input question has keywords like company names and asks for these following actions,
        buy or sell, show profit, returns, trend;
        then use the companies table to return the concat value of 'https://groww.in/stocks/' and 
        search_id for the relevant company name from company_name using the correct mysql query.
        return the SQLResult
        
        If the input question has keywords like industry average,
        then take average of all stocks corresponding to that industry given in tag column of company table
        using the correct mysql query.
        
        If the input question has mutual funds keyword, 
        then return the concat value of endpoint from explore table, config columns from mutual fund category table if requested.
        The return value is a logical REST url from the above values.
        
        If the input question depends on general financial knowledge like what, how, when about a company etc, 
        then respond to the user in less than 100 words as though the user is an intelligent person,
        but with simple financial terms that the user can understand.
        
        
        The return statement should be relevant to the question asked, using the above classification provided above.
        """

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
