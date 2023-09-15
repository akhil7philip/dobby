import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def init_pinecone():
    import pinecone
    from langchain.vectorstores import Pinecone
    import os, time
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings

    with open('docs/data_dump.txt') as f:
        data = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
    )

    texts = text_splitter.create_documents([data])

    embeddings = OpenAIEmbeddings(model_name='text-embedding-ada-002')

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    index_name = "dobby"
    if index_name not in pinecone.list_indexes():
        # pinecone.delete_index(index_name)
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

    return Pinecone.from_documents(texts, embeddings, index_name=index_name)


def vector_store_initialize():
    logging.info("initializing Pinecone")
    vs = init_pinecone()
    logging.info("Pinecone initialized")
    return vs

