import os
from flask import Flask, render_template, request, jsonify
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import asttokens, ast
from IPython.display import Markdown, display
import openai
import config

os.environ["OPENAI_API_KEY"] = config.DevelopmentConfig.OPENAI_KEY



def page_not_found(e):
  return render_template('404.html'), 404

app = Flask(__name__, static_folder="static")

app.register_error_handler(404, page_not_found)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query = request.form['query']
    response = index.query(query)
    return jsonify({'response': response.response})

if __name__ == '__main__':
    # set maximum input size
    max_input_size = 4096 
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader("static/data").load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    index.save_to_disk('static/vector_index/index.json')


    app.run( debug=True)
