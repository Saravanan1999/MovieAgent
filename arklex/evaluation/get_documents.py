import os
import sys
import json
import pickle
from pathlib import Path
from os.path import dirname, abspath

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from arklex.utils.loader import Loader, CrawledURLObject

def get_domain_info(documents):
    summary = None
    for doc in documents:
        if doc['URL'] == 'summary':
            summary = doc['content']
            break
    return summary

def load_docs(document_dir, doc_config, limit=10):
    if "rag_docs" not in doc_config:
        if "task_docs" not in doc_config:
            raise ValueError("The config json file must have a key 'rag_docs' or 'task_docs' with a list of documents to load.")
        else:
            task_docs = doc_config['task_docs']
            # Handle JSON files in task_docs
            if isinstance(task_docs, list) and len(task_docs) > 0 and isinstance(task_docs[0], str) and task_docs[0].endswith('.json'):
                documents = []
                for doc_path in task_docs:
                    try:
                        with open(doc_path, 'r') as f:
                            doc_content = json.load(f)
                            documents.append({"url": doc_path, "content": json.dumps(doc_content), "metadata": {}})
                    except Exception as e:
                        print(f"Error loading JSON file {doc_path}: {e}")
                return documents
            else:
                rag_docs = task_docs
                filename = "task_documents.pkl"
    else:
        rag_docs = doc_config['rag_docs']
        filename = "documents.pkl"
    
    if document_dir is not None:
        filepath = os.path.join(document_dir, filename)
        total_num_docs = sum([doc.get("num") if doc.get("num") else 1 for doc in rag_docs])
        loader = Loader()
        if Path(filepath).exists():
            docs = pickle.load(open(os.path.join(document_dir, filename), "rb"))
        else:
            docs = []
            for doc in rag_docs:
                source = doc.get("source")
                num_docs = doc.get("num") if doc.get("num") else 1
                urls = loader.get_all_urls(source, num_docs)
                crawled_urls = loader.to_crawled_obj(urls)
                docs.extend(crawled_urls)
            Loader.save(filepath, docs)
        
        if not docs:  # If docs is empty, return empty list
            return []
            
        if total_num_docs > 50:
            limit = total_num_docs // 5
        else:
            limit = 10
        if isinstance(docs[0], CrawledURLObject):
            documents = loader.get_candidates_websites(docs, limit)
        else:
            documents = []
            for doc in docs:
                documents.append({"url": "", "content": doc.page_content, "metadata": doc.metadata})  
    else:
        documents = []
    return documents

if __name__ == "__main__":
    doc_config = json.load(open('./temp_files/richtech_config.json'))
    docs = load_docs('./temp_files', doc_config, 10)