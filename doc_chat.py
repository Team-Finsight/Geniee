from PIL import Image
import pytesseract
import logging
# LangChain Imports
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chat_models import ChatAnyscale
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
import fitz
import logging
from pythonjsonlogger import jsonlogger
from docx import Document

def setup_logging(level=logging.INFO):
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    log_handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[log_handler])
    logger = logging.getLogger('flask.app')
    logger.addHandler(log_handler)
    logger.setLevel(level)

setup_logging()

 # Load environment variables

api_key = 'esecret_sn18dj7defst1n3aacssvnc94m'
api_base = "https://api.endpoints.anyscale.com/v1"



def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""
from langchain.document_loaders import PyPDFLoader

def pdf_to_text(pdf_path):
    try:
        # Initialize PyPDFLoader with the pdf_path
        loader = PyPDFLoader(file_path=pdf_path)
        text = loader.load()
        return text
    except Exception as e:
        print(f"Error during OCR processing of PDF: {e}")
        return ""
    
def ocr_pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
        doc.close()
        return text
    except Exception as e:
        print(f"Error during OCR processing of PDF: {e}")
        return ""



def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        print(fullText)
        return '\n'.join(fullText)
    except Exception as e:
        print(f"Error during DOCX processing: {e}")
        return ""
    
def qa(extracted_text, query):
    # load document
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
    texts = text_splitter.split_text(extracted_text)
    docs=text_splitter.create_documents(texts)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=embeddings
    )
    vectorstore.add_documents(docs)
    template = """Based on the provided text and context, the goal is to generate a single rewritten query that is most relevant for the Document Question Answering Assistant.

    Original Text: {text}
    Context: {context}

    Process:
    1. Analyze both the original text and the context.
    2. Determine the main focus or intent behind the text.
    3. Rewrite the query in a clear detailed manner, ensuring it aligns with the specific needs of the question.
    4. Identify key terms in the related document content that are relevant to the query.

    Output:
    Rewritten Query: {{Insert the single, most relevant rewritten query here, aligning with the text and context.}}
    Key Terms: {{List key terms here that are relevant to the rewritten query and related document content.}}
    """
    llm = ChatAnyscale(anyscale_api_key=api_key, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",max_tokens=10000,streaming=True,temperature=0.1)
    # Define the prompt template with both 'text' and 'context' as input variables
    prompt = PromptTemplate(template=template, input_variables=["text","context"])

    # Define the LLMChain with the prompt and LLM configuration
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(text=query,context=docs)

    memory = ConversationBufferMemory(
                                    memory_key="chat_history",
                                    max_len=5,
                                    return_messages=True,output_key='answer'
                                )
    general_system_template = """Instructions: Your role as a document-focused question-answering assistant is to deliver comprehensive and precise responses. Adhere to these steps:

    1. Comprehensive Analysis: Thoroughly review the entire document content related to the query. Your answer should cover all pertinent aspects without omitting any significant detail.

    2. Confirmation and Integration: Post-provision of the initial answer, conduct a secondary review of the document. This is to ensure no critical information has been missed. Should you discover additional relevant facts, seamlessly integrate these into your existing response.

    3. Structured and Clear Responses: Format your answers in a manner that is easy to understand, with a logical sequence. This aids in enhancing the reader's comprehension and retaining their engagement.

    4. Source Attribution: Conclude your answer by citing the specific sections or paragraphs of the document that your response is based on. This adds credibility and allows for easy reference.

    Your objective is to deliver exhaustive and well-structured answers, ensuring they encompass all related content from the document.


    Provided Document: {context}

    Expected Output Format:
    Answer: [Provide a detailed and well-structured response here]
    Source: [Cite the specific parts of the document referenced in the answer]

    """

    general_user_template = "Question:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )


    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm=ChatAnyscale(anyscale_api_key=api_key, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",max_tokens=10000,streaming=True,temperature=0.1)
    # create a chain to answer questions
    qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=retriever,
    return_source_documents=True,combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    query=response

    result = qa({"question": query,'chat_history': memory.chat_memory.messages})
    return result['answer']
