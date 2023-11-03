from flask import Flask, request, jsonify
from langchain.document_loaders import DirectoryLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import re
import urllib.request
import xml.etree.ElementTree as ET
import requests
from typing import List
from tenacity import retry, wait_exponential
import openai
from vercel import VercelRequest, VercelResponse


app = Flask(__name__)

# API key for OpenAI
api_key = ""

# Define model parameters
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.1,
    max_tokens=4000,
    n=1,
    openai_api_key=api_key
)

# Sitemap URL
sitemap_url = "https://famaservices.net/sitemap.xml"

# Function to extract URLs from the sitemap
def extract_urls_from_sitemap(sitemap: str) -> List[str]:
    urls = []

    def process_sitemap(sitemap_url):
        response = requests.get(sitemap_url)
        sitemap_content = response.text

        # Extract URLs enclosed within <loc> tags
        extracted_urls = re.findall(r"<loc>(.*?)</loc>", sitemap_content)

        if not extracted_urls:
            # Extract URLs enclosed within <url> tags
            extracted_urls = re.findall(r"<url>\s*<loc>(.*?)</loc>\s*</url>", sitemap_content)

        if not extracted_urls:
            # Extract URLs separated by tabs
            extracted_urls = re.findall(r"\t(https?://[^\s]*)", sitemap_content)

        if not extracted_urls:
            # Extract URLs separated by line breaks
            extracted_urls = re.findall(r"\n(https?://[^\s]*)", sitemap_content)

        urls.extend(extracted_urls)

        # Check if the extracted URLs are sitemap URLs ending with .xml
        nested_sitemap_urls = [url for url in extracted_urls if url.endswith('.xml')]

        # Recursively process the nested sitemaps
        for nested_sitemap_url in nested_sitemap_urls:
            process_sitemap(nested_sitemap_url)

    # Attempt regular expression-based extraction first
    process_sitemap(sitemap)

    # If the URLs list is still empty, try using xml.etree.ElementTree as a fallback
    if not urls:
        try:
            response = urllib.request.urlopen(sitemap)
            sitemap_data = response.read()

            root = ET.fromstring(sitemap_data)
            for element in root.iter():
                if "url" in element.tag:
                    for loc in element:
                        if "loc" in loc.tag:
                            urls.append(loc.text)

                # Check if there are nested sitemaps
                if "sitemap" in element.tag:
                    for loc in element:
                        if "loc" in loc.tag:
                            try:
                                urls.extend(extract_urls_from_sitemap(loc.text))
                            except ET.ParseError:
                                # Skip this nested sitemap if it is not well-formed
                                continue

        except ET.ParseError:
            # Skip the current sitemap if it is not well-formed
            return []
    return urls

# Extract URLs from the sitemap
sitemap_urls = extract_urls_from_sitemap(sitemap_url)

# Folder path for additional data files (.txt, .pdf, .csv, .docx, .xlsx)
data_folder = "./data"

# Load data from the sitemap and data folder
data_loader = DirectoryLoader(data_folder)
url_loader = UnstructuredURLLoader(sitemap_urls)
data = [] 
data += data_loader.load()
data += url_loader.load()

# Create vector database of the loaded data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
split_data = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")
vector_data = Chroma.from_documents(split_data, embeddings)

# Define connection to model
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_data.as_retriever())

# Define the chatbot function
def chatbot(request: VercelRequest) -> VercelResponse:
    # Extract the input_text and history from the request
    input_text = request.get('input', 'System message: User input not provided. Please reply asking the user to ask their question again. ')
    history = request.get('history', [])

    # Your logic here
    qa_prompt = "Context: You are a customer service chatbot and your primary role is to provide answers based on the knowledge stored in your internal database stored in the ./data folder and at the provided sitemap. When answering questions, please include the source or document name to provide the user with the origin of the information. If the information is not available in the knowledgebase, please respond with 'Information not found in knowledgebase'. If a user is asking anything about FAS or Fama Automation Services, using all the context you have, try to respond to their question."

    try:
        # First API call to the initial chatbot model
        response = qa({"query": qa_prompt + input_text})
        response_text = f"{response['result']}"

        # Check if a second API call to a different model is needed
        if "Information not found" in response_text.lower() or "i'm sorry" in response_text.lower() or "i don't have that information" in response_text.lower():

            general_knowledge_model = openai.ChatCompletion.create(
                messages=[
                    {'role': 'system', 'content': 'You are a customer service chatbot. Your primary role is to provide answers based on the knowledge that you have generally. Do your best to respond to the question, indicating to the user that your response is based on general knowledge, and provide the source of your information.'},
                    {'role': 'user', 'content': input_text},
                ],
                model="gpt-4",
                temperature=0.7,
            )
            general_response_text = general_knowledge_model['choices'][0]['message']['content']

            if general_response_text.strip() != "":
                response_text = general_response_text
            else:
                response_text = "SYS FAILURE: I couldn't find a suitable response in my knowledge base or anywhere else."

        return VercelResponse.json({"history": history, "response": response_text})
    except Exception as e:
        # Handle exceptions gracefully
        return VercelResponse.text(f"An exception has occurred: {e}")

# Define a route for the chatbot function
@retry(wait=wait_exponential(multiplier=0.02, max=32))
@app.route('/index', methods=['POST'])
def chatbot_flask():
    # Call the chatbot function
    result = chatbot(request)

    # Extract the response and return as a Flask response
    return jsonify(result.json())

if __name__ == '__main__':
    app.run()