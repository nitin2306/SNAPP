# Smart Networked Assistant for Purchase Processing (SnApP)
A web application which leverages generative AI and various RAG architectures to analyse the purchase order and provide the approver with a chatbot which answer all the queries related directly or indirectly to the purchase order regardless of the language.
## Prerequisites
Before running the project, ensure you have Python installed on your machine. You also need to install the following libraries.
### Installation
Run the following commands in your terminal or command prompt to install the required Python libraries:
- pip install langchain-groq
- pip install langchain_community
- pip install fastembed
- pip install pymupdf
- pip install langdetect
- pip install langgraph
- pip install chromadb
### Using Groq API
- The user does not need to explicitly download any model to host on their machine. Instead, the code uses Groq APIs to function
(Since the Groq API is free to use, it has certain limitations which prevents processing of larger docs. Usage of premium APIs can resolve this issue)
- Navigate to https://console.groq.com/keys and create your own API Key.
- Copy the API key and then place it in the ChatGroq function api_key field
### Using Tavily API
- Next, the user shall navigate to https://app.tavily.com/home to create another API Key
- Copy the API key and place it in ‘TAVILY_API_KEY’ environmental variable mentioned in the code
### Execution
- In LangCode.py Set variable UPLOAD_FOLDER as the Path where you want to save pdf during execution
Run the LangCode.py
- Open index.html in your browser to view the chatbot
- Upload the pdf and start asking questions
- (Due to the usage of free API with limited capabilities, expect some delay in the results)
This `README.md` provides clear instructions for setting up and running the project, including installation, API key creation, configuration, and running the application.
### Developed by Team 5
- Srinivas Lakshmi Bannuru
- Nitin Chaudhary
- Ardhendu Banerjee
-  Prashant Singh Chauhan
- Pratik Bendre
### References
- Jeong, Soyeong, et al. "Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity." arXiv preprint arXiv:2403.14403 (2024).
- Yan, Shi-Qi, et al. "Corrective retrieval augmented generation." arXiv preprint arXiv:2401.15884 (2024).
- Asai, Akari, et al. "Self-rag: Learning to retrieve, generate, and critique through self-reflection." arXiv preprint arXiv:2310.11511 (2023).
- https://medium.com/the-ai-forum/build-a-reliable-rag-agent-using-langgraph-2694d55995cd

