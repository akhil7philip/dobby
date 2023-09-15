import logging
import flask
import re
from dotenv import find_dotenv, load_dotenv
from flask_cors import CORS

model_name="gpt-3.5-turbo"
logging.info(f'load env: {load_dotenv(find_dotenv())}')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

import helpers
import webbrowser

app = flask.Flask(__name__)
CORS(app)


@app.route('/get_user_audio_input', methods=['GET', 'POST'])
async def get_user_audio_input():
    logging.info("get_user_audio_input triggered")
    request = flask.request.files.get('audio')
    logging.info(f"REQUEST: {flask.request.files}")
    request.save(request.filename)
    logging.info("SAVED")
    
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
    data = {
        'question': input_text,
        'answer': answer
    }
    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # app.run(host='0.0.0.0', debug=True, port=8080)
