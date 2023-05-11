# import to get credentials
import os
from dotenv import load_dotenv

# for gpu
import torch

# imports from other files
from text_preprocessing import clean_text
from database_connection import DocumentOpenai, DocumentComments, TestEmpty, DocumentPalm, Session

# models from langchain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# imports for google embedding model
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

# load credentials from environment
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CONNSTR = os.getenv('CONNSTR_POSTGRES')
PROJECT_ID = os.getenv('PROJECT_ID')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('PATH_TO_CREDENTIALS')

# init database structure and session
Session = Session(CONNSTR=CONNSTR)

# get device - gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# most important - openAI model with api key
openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model='text-embedding-ada-002')

# hugging face model if needed for tests
model_name = 'paraphrase-multilingual-MiniLM-L12-v2 '
model_kwargs = {'device': device}
#hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# init vertexai and define palm model (google)
vertexai.init(project=PROJECT_ID)
palm = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

# everything about session - create engine, connect new session and find closest neighbors
Session.create_engine()
Session.connect_session()

# try to download data in batches
batch_size = 1000
offset = 0

# the main loop - change to while True when testing is closed
while offset == 0:

    # Fetch the data in batches
    rows = Session.session.query(DocumentComments).offset(offset).limit(batch_size).all()

    if not rows:
        # exit if no more data
        break

    # define rows to be added to the main database
    destination_rows = []

    # loop through rows and add all to the list
    for row in rows:
        # send it to new table
        destination_row = DocumentPalm()
        destination_row.id = row.id
        destination_row.embedding = palm.get_embeddings([row.comment_text])[0].values

        #destination_rows.append(destination_row)
        
        Session.session.add(destination_row)
        Session.session.commit()

        print('Row added')

    # Session.session.add_all(destination_rows)
    # Session.session.commit()

    offset += batch_size

    print(f'Done {offset + batch_size} rows')

Session.close_session()
