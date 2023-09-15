import logging
import flask
import re
from dotenv import find_dotenv, load_dotenv
from flask_cors import CORS
# import io
# import soundfile as sf
load_dotenv()
model_name="gpt-3.5-turbo"

import helpers
import webbrowser
import initialize


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

logging.info(f'load env: {load_dotenv(find_dotenv())}')

app = flask.Flask(__name__)
CORS(app)

@app.route('/hello')
async def hello():
    global vector_store
    vector_store = initialize.vector_store_initialize()
    data = {
        'text': "complete",
    }
    return flask.jsonify(data)


@app.route('/get_user_audio_input', methods=['GET', 'POST'])
async def get_user_audio_input():
    logging.info("get_user_audio_input triggered")
    request = flask.request.files.get('audio')
    print(f"REQUEST: {flask.request.files}")
    request.save(request.filename)
    print("SAVED")

    # response = flask.request.files.get('audio')
    # input_audio_file = response.stream.read()
    # print(f"FILE response: {response}")
    # print(f"FILENAME: {response.filename}")
    # print(f"READ OBJECT: {flask.request.stream.read()}")
    # print(f"STREAM TYPE: {type(response.stream.read())}")
    # data, samplerate = sf.read(io.BytesIO(input_audio_file))
    # input_audio_file = data
    # input_audio_file_path = '/Users/akhil.philip/learn/lang_chain/audio_files/zomato.mp3'
    # with open(request.filename, "rb") as f:
    #     input_audio_file = f.read()

    input_text = ''
    if request.filename:
        logging.info("attempting translation")
        # input_text = helpers.get_translation_faster_whisper(request.filename)
        input_text = helpers.get_translation_openai(request.filename)
        # input_text = 'I want to buy 2 stocks of hdfc'
        logging.info("received translation from asr model")
    logging.info(f"input_text: {input_text}")
    # data = {
    #     'text': input_text,
    # }
    # yield flask.jsonify(data)
    # return flask.redirect(flask.url_for('get_user_text_input', question=input_text))


# @app.route('/get_user_text_input')
# async def get_user_text_input():
#     logging.info("get_user_text_input triggered")
#     response = flask.request.json
#     input_text = response.get('input_text', '')
    # input_text = 'I want to buy 2 stocks of hdfc'
    logging.info("triggering classification")
    classification = helpers.classify_text(input_text)
    if classification == 'help':
        logging.info(f"input_text {input_text} falls under help category")
        answer = helpers.fetch_hns_response(input_text)
    else:
        logging.info(f"input_text {input_text} falls under general category")
        answer = helpers.fetch_general_response(input_text)
    logging.info(f"answer {answer}")
    redirect, text, image = '', '', ''
    # url = answer.find('https://groww.in')
    url_array = re.findall('/groww.in', answer)
    if len(url_array) == 1:
        try:
            url = answer.find('https://groww.in')
            redirect = answer[url:].split(' ')[0].strip('.,;)')
            webbrowser.open(redirect)
        except Exception as e:
            logging.error(e)
            text = answer
    else:
        text = answer
    data = {
        'question': input_text,
        'redirect': redirect,
        'response': text,
    }
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # app.run(host='0.0.0.0', debug=True, port=8080)
