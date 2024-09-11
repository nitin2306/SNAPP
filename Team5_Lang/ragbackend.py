import os
import fitz
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langdetect import detect
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from pprint import pprint
from langgraph.graph import END, StateGraph

#Embeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

#Choose the LLM Model
llm = ChatGroq(temperature=0,
                      model_name="Llama3-70b-8192",
                      api_key="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",)

#Tavily API
os.environ['TAVILY_API_KEY'] = "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"
web_search_tool = TavilySearchResults(k=3)

#Global counter
counter = 0

#Prompt for the translator
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a language translator for translating the text to English.
    Strictly return the only the trasnlated text without any addditional information.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Input Text: {input}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["input"],
    )
translator = prompt | llm | StrOutputParser()

def extract_pdf_content(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = ""
    for page in doc:
        # Extract text
        text_content += page.get_text()
    return text_content

def initialize_vectorstore(file_path):

    long_text  = extract_pdf_content(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=900)
    # Split the text into chunks
    chunks = splitter.split_text(long_text)
    # Initialize the embeddings model and Chroma DB
    vectorstore = Chroma(embedding_function=embed_model)
    #Translate
    for i in range(len(chunks)):
        if detect(chunks[i]) != 'en':
            chunks[i] = translator.invoke({"input": chunks[i]})

    # Embed and store the chunks in Chroma DB
    vectorstore.add_texts(chunks)

    # Initialize the retriever
    global retriever
    retriever = vectorstore.as_retriever()
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_prompt_templates():

    #Prompt for the question router
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
        user question to a vectorstore or web search. Use the vectorstore for questions on Purchase Order (PO),
        supplier details, item details, company information, summaries and order related details. You do not need to be stringent with the keywords
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
        no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()

    #Prompt for QA Assistant
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Context: {context}
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )

    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}  | prompt  | llm | StrOutputParser())
    # rag_chain = prompt | llm | StrOutputParser()

    #Grading relevance of the answer
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    retrieval_grader = prompt | llm | JsonOutputParser()

    #Check for hallucinations
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = prompt | llm | JsonOutputParser()

    #Grade the answer according to the question
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    answer_grader = prompt | llm | JsonOutputParser()
    return question_router,rag_chain,retrieval_grader, hallucination_grader, answer_grader

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
#
def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
#
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
#
def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def irrelevant(state):
    """
    Web search based based on the question
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Appended web results to documents
    """
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents = [web_results]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"question": question, "generation": "The question seems not related to the document provided. The web results that were found are as followed: " + generation}

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ASKED QUESTION IS NOT RELEVANT---")
        return "nodata"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")

    global counter

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            counter = counter + 1
        if counter < 3:
            return "not useful"
        else:
            return "useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        counter = counter + 1
        if counter < 3:
            return "not supported"
        else:
            return "useful"


def create_workflow(vectorstores, question_routers, rag_chains, retrieval_graders, hallucination_graders, answer_graders):
    from langgraph.graph import END, StateGraph
    from typing_extensions import TypedDict

    global question_router, rag_chain, retrieval_grader, hallucination_grader, answer_grader, vectorstore

    vectorstore = vectorstores
    question_router = question_routers
    rag_chain = rag_chains
    retrieval_grader = retrieval_graders
    hallucination_grader = hallucination_graders
    answer_grader = answer_graders

    ### State
    class GraphState(TypedDict):
        question : str
        generation : str
        web_search : str
        documents : List[str]

    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search) # web search
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("generate", generate) # generatae
    workflow.add_node("irrelevant", irrelevant) # irrelevant

    workflow.set_conditional_entry_point(
        route_question,
        {
            "nodata": "irrelevant",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("irrelevant", "__end__")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    app = workflow.compile()
    return app