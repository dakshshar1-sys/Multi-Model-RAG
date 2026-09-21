# Failure analysis (generated)

Produced by `python -m eval.failure_analysis` from the latest result files. Rows are the
system's actual outputs; nothing here is edited by hand.


## 1. Retrieval misses (dense top-5) — 14 of 64

| id | question | source | gold evidence | dense rank | hybrid rank |
|---|---|---|---|---|---|
| q04 | What three popular RAG tools does the survey report list? | rag_report.txt | LlamaIndex: A deep learning based index which can read acro… | None | 3 |
| q05 | What are the three retrieval-level errors listed in the report? | rag_report.txt | Irrelevance: Retrieving the wrong document | 10 | 1 |
| q06 | What example does the report give of an irrelevance error? | rag_report.txt | (e.g. Finance as Legal) | None | 7 |
| q07 | Which two generation-level errors does the report identify? | rag_report.txt | Drift: Losing query focus | None | 3 |
| q14 | How does the report characterise iteration-based correction? | rag_report.txt | This is a simple approach that is not deep-acting | 8 | 2 |
| q20 | Name two research gaps identified in the survey report. | rag_report.txt | Non-availability of dataset of domain-specific RAG benchmar… | None | None |
| q21 | What future research direction does the report suggest for images and charts? | rag_report.txt | Exploring multi-modal RAG for images and charts | 7 | 1 |
| q24 | What does FAISS stand for? | system_architecture.txt | FAISS (Facebook AI Similarity Search) | 8 | 6 |
| q29 | What file is described as the most critical file in the backend? | system_architecture.txt | The most critical file in the backend | 6 | 3 |
| q32 | How does the reranker reduce context noise for the LLM? | system_architecture.txt | take the top 10 results | 8 | 3 |
| q36 | In the RAG lifecycle example, what does the verifier flag if it sees $40B inste… | system_architecture.txt | If it sees $40B instead of $100B | None | 10 |
| q53 | What does the problem statement say current RAG-based synthesis systems lack? | chapter_report.txt | lacking a unified framework that integrates context-aware e… | 9 | 1 |
| q54 | Which backend technologies does the proposed model list for managing secure dat… | chapter_report.txt | Backend services built with Node.js and Python | 8 | 3 |
| q64 | What must the administrator dashboard display according to the functional requi… | chapter_report.txt | detected error rates, correction logs, and performance anal… | 8 | 4 |

A missing rank means the evidence was not in the top-10 at all. Where the hybrid rank is
small and the dense rank large, the question's wording matched the chunk lexically but not
semantically: typically a short bulleted line whose embedding is dominated by its neighbours.

## 2. Generation faults — evidence retrieved, gold fact missing from the answer: 17 of 64

| id | question | gold answer | answer (excerpt) | lexical support | judge |
|---|---|---|---|---|---|
| q13 | What is the drawback of cross-encoder models for deep learning-based … | They are very accurate but require a lot of computational power. | The drawback of cross-encoder models for deep learning-based detection, as mentioned in [Source 3], is that they requir… | 1.00 | PASS |
| q14 | How does the report characterise iteration-based correction? | A multi-turn dialogue that measures the consistency of answers and co… | The report characterises iteration-based correction as a simple approach used in automated fact correction techniques. … | 0.50 | PASS |
| q22 | What does the architecture document mean by a Glass Box AI experience? | The system streams a live Pipeline Execution Path to the frontend via… | The architecture document refers to a "Glass Box" AI experience as one where users can see every step of the decision-m… | 0.50 | PASS |
| q25 | Why does the document say running Llama 3.2 locally through Ollama ma… | It ensures data privacy because no data leaves the local machine duri… | Running Llama 3.2 locally through Ollama is crucial for ensuring both data privacy and speed, especially on systems wit… | 0.67 | PASS |
| q26 | Which frontend framework and version does the architecture document s… | Next.js 15 with React 18, using the App Router and Server Components. | The architecture document specifies that the frontend is built using Next.js 15. | 1.00 | PASS |
| q27 | Which libraries does the Visualizer Agent use to generate charts? | Pandas, Matplotlib and Seaborn. | The Visualizer Agent uses Pandas and Matplotlib libraries to dynamically generate charts and graphs when numerical data… | 1.00 | PASS |
| q30 | Which embedding model does the architecture document name and how man… | All-MiniLM-L6-V2 from SentenceTransformers, producing 384-dimensional… | The architecture document names a specific embedding model called `SentenceTransformers` used to convert text into 384-… | 1.00 | PASS |
| q31 | What are the three tiers of the web search engine described in the re… | It scrapes DuckDuckGo, falls back to the Wikipedia REST API, and uses… | The three tiers of the web search engine described in the retrieval layer are as follows:  - The first tier uses DuckDu… | 0.40 | PASS |
| q32 | How does the reranker reduce context noise for the LLM? | A cross-encoder takes the top 10 results from FAISS and re-sorts them… | The reranker reduces context noise for the LLM by re-ranking the top-10 results from FAISS based on semantic relevance … | 0.67 | PASS |
| q33 | What approximate chunk size does document_parser.py produce? | Approximately 1000 characters, overlapping. | The approximate chunk size that document_parser.py produces is 1000 characters. This information is explicitly stated i… | 0.00 | PASS |
| q44 | How many candidates does FAISS fetch initially and how many chunks re… | FAISS fetches the top 10; only the top 5 highest-scoring chunks are p… | FAISS fetches **initially** Top-10 candidates. After applying the Reranker model, only the **Top-5 highest-scoring chun… | 1.00 | PASS |
| q46 | Which three services does docker-compose.yml define according to the … | backend (FastAPI), frontend (Next.js 15) and an optional ollama servi… | The `docker-compose.yml` defines the following three services:  - backend - frontend - ollama  These services are descr… | 1.00 | PASS |
| q50 | What are the minimum RAM and VRAM requirements stated for the system? | 16GB system RAM and 4GB VRAM (RTX 2050/3050). | The minimum RAM requirement stated for the system is **16GB**. The minimum VRAM requirement is **4GB**.  Detailed Analy… | 0.80 | PASS |
| q51 | What end-to-end pipeline latency does the document benchmark? | 10 to 15 seconds end to end. | The end-to-end pipeline latency benchmarks provided in the document are 10s to 15s. [1] | 1.00 | PASS |
| q56 | What is a stated con of NLP-based contextual correction for RAG syste… | Limited adaptability to highly unstructured or noisy documents, needs… | The cons of the NLP-based contextual correction for RAG systems are that it requires high-quality vector datasets and d… | 1.00 | PASS |
| q58 | What do transformers do differently from CNNs or RNNs when processing… | They analyse the entire retrieved context at once and identify relati… | Transformers process retrieved text by analyzing the entire context at once and identifying relationships between dista… | 1.00 | PASS |
| q62 | Which document processing libraries are listed for PDFs and image pre… | PDFPlumber, PyMuPDF or OpenCV. | The document processing libraries mentioned for handling PDFs and image preprocessing are:  - For handling PDFs: `PyMuP… | 0.75 | PASS |

These are the cases the retrieval stack cannot fix: the fact was in front of the model.
Read the excerpts: the common patterns are paraphrase that drops the key term, a partial
list, or the model answering a neighbouring question from the same chunk.

## 3. LLM judge vs lexical proxy disagreements — 17

| id | lexical support | judge | judge reason | contains gold |
|---|---|---|---|---|
| q06 | 0.33 | PASS | The answer accurately summarizes the context provided and does not contain any significan… | 0 |
| q07 | 0.50 | PASS | The answer accurately summarizes the two main areas of interest for the survey as describ… | 0 |
| q14 | 0.50 | PASS | The answer accurately summarizes the description provided for iteration-based correction … | 0 |
| q18 | 0.33 | PASS | The answer is based on factual information provided in the context about Blockchain-Based… | 1 |
| q19 | 0.00 | PASS | The answer is based on the provided context and accurately summarizes the threats mention… | 1 |
| q22 | 0.50 | PASS | The answer is based on and accurately reflects the information provided in the context. I… | 0 |
| q23 | 0.50 | PASS | The answer is based on the provided context without significant hallucinations. It accura… | 0 |
| q31 | 0.40 | PASS | The answer accurately describes the three-tier web search engine as described in the cont… | 0 |
| q33 | 0.00 | PASS | The answer directly states "document_parser.py: Uses `PyPDF` and `LangChain` text splitte… | 0 |
| q34 | 0.33 | PASS | The provided Answer accurately reflects the functionality described in the Context for th… | 1 |
| q35 | 1.00 | FAIL | The answer provided contains a significant hallucination. The context mentions a color pa… | 1 |
| q40 | 0.50 | PASS | The answer accurately summarizes and correctly interprets the constraint provided in the … | 1 |
| q42 | 1.00 | FAIL | The context mentions that the Pipeline Visualizer's `Timeline Component` is a side panel,… | 0 |
| q45 | 0.33 | PASS | The answer accurately reflects the approach described in the provided context without int… | 1 |
| q59 | 0.50 | PASS | The answer is directly supported by the provided context, which mentions "Author: Senka D… | 1 |
| q60 | 1.00 | FAIL | The provided answer contains a significant hallucination by stating that the document pro… | 1 |
| q64 | 0.75 | FAIL | The provided answer contains a significant hallucination by mentioning "AI-driven RAG" wh… | 1 |

A PASS with low support and no gold fact is the judge being lenient; a FAIL with high
support and the gold fact present is the judge being wrong. Count both before trusting it.

## 4. Routing confusions — strict accuracy 0.906, unstable 0

| expected | got | count | example query |
|---|---|---|---|
| Web_Search | Visualize_Data | 3 | chart Samsung's 2025 revenue by quarter |
| Search_Knowledge_Base | Web_Search | 3 | what do my uploaded notes say about ensemble methods? |
| Ambiguous_Query | Web_Search | 3 | AAPL |

## 5. Model alone, no retrieval — 2 of 64 contain the gold fact

| id | question | model's answer (excerpt) | contains gold |
|---|---|---|---|
| q01 | According to the survey report, what were early RAG systems based on? | Early RAG (Retrieval-Augmented Generation) systems were often based on pre-existing retrieval techniques and … | 0 |
| q02 | Which two encoding techniques does the report name under machine lear… | The report doesn't specifically name the encoding techniques used for Machine Learning-based Retrieval-Augmen… | 0 |
| q03 | In agentic RAG, which technique is used to identify important fields … | In agentic RAG, Named Entity Recognition (NER) techniques are commonly used to identify important fields. | 1 |
| q04 | What three popular RAG tools does the survey report list? | I'm sorry, but I don't have that specific information readily available. Could you possibly check the most re… | 0 |
| q05 | What are the three retrieval-level errors listed in the report? | I'm sorry, I don't have access to your report. Could you provide more details or check the report directly? | 0 |
| q06 | What example does the report give of an irrelevance error? | The report gives an example where incorrect assumptions led to misleading conclusions. | 0 |
| q07 | Which two generation-level errors does the report identify? | I'm sorry, but I don't have information about a specific report ready for you. Could you provide more details… | 0 |
| q08 | What does the report describe as a contextual error? | A contextual error refers to an issue where information is incorrect or inconsistent based on the context pro… | 0 |
| q09 | What are the structural errors named in the report? | I'll need more details about which specific report you're referring to. Could you please specify what kind of… | 0 |
| q10 | What limitation do semantic-based error detection methods have, accor… | According to the report, one major limitation of semantic-based error detection methods is their sensitivity … | 0 |
| q11 | What do statistical detection methods use to approximate chunk releva… | Statistical detection methods often use probabilistic models to estimate the relevance of chunks in data. | 0 |
| q12 | Which three context-based agent methods are listed for error detectio… | I'm sorry, but I don't have specific information about three context-based agent methods for error detection.… | 0 |

(first 12 of 64 shown; the ones it gets right are general knowledge, e.g. q03, q61.)
