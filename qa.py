"""Ask a question to the notion database."""
import faiss
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
chain = RetrievalQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0, max_tokens=4097), 
    retriever=store.as_retriever(search_kwargs={"k": 3, "filter": {"type": "filter" }}),
    max_tokens_limit=4097, reduce_k_below_max_tokens=True)

result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
