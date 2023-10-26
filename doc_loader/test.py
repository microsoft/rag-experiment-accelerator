from documentLoader import load_documents


docs = load_documents("html", "../data/", 2000, 0)

for doc in docs:
    print(doc)