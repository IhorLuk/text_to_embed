# import to get credentials
import os
from dotenv import load_dotenv

# imports from other files
from text_preprocessing import clean_text
from database_connection import DocumentOpenai, DocumentComments, TestEmpty, Session

from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CONNSTR = os.getenv('CONNSTR_POSTGRES')

# init database structure and session
Session = Session(CONNSTR=CONNSTR)

# most important - openAI model with api key
openai = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model='text-embedding-ada-002')

# everything about session - create engine, connect new session and find closest neighbors
Session.create_engine()
Session.connect_session()

# try to download data in batches
batch_size = 1000
offset = 0

total_rows = 0

while True:

    # Fetch the data in batches
    rows = Session.session.query(DocumentComments).offset(offset).limit(batch_size).all()

    if not rows:
        # exit if no more data
        break

    for row in rows:
        # send it to new table
        destination_row = TestEmpty()
        destination_row.id = row.id
        destination_row.transformed_text = row.comment_text[:2]

        Session.session.add(destination_row)
        Session.session.commit()

    offset += batch_size

print(f'Total rows: {total_rows}.')
Session.close_session()
