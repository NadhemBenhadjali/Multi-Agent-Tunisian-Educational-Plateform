# Etude.AI – Tunisian Dialect-Aware Educational AI Platform

##  Introduction  
Etude.AI was developed for the **Artificial Intelligence National Summit (AINS) Hackathon**, under *Track 3: Tunisian Dialect-Aware AI*.  
It is a **smart web platform** designed to support children with learning difficulties through **personalized, AI-powered educational content** in the Tunisian dialect.  

The platform provides:  
-  **Lesson Summaries** – Simplify concepts from the curriculum.  
-  **Question Answering** – Respond to student queries in Tunisian dialect.  
-  **Quizzes** – Test understanding of specific lessons.  
-  **Progress Reports** – Track learning activities and generate detailed PDF reports.  

---

##  System Overview  
Etude.AI enhances learning with **multi-agent collaboration**, dynamically generating educational content based on student input, interests, and curricular structure.  

Key components include:  
- **Summary Agent** – Generates textual & voice-enabled lesson summaries.  
- **Q&A Agent** – Answers general and lesson-specific questions.  
- **Quiz Agent** – Builds interactive quizzes.  
- **History Agent** – Tracks activities and generates structured reports.  

A **Knowledge Graph (Neo4j)** serves as the backbone, semantically linking **branches → topics → lessons → images**, enabling precise retrieval and adaptive learning.

---

## General Workflow  
1. **Student Input**: Learner specifies class and subject.  
2. **Mode Selection**: Choose between *Summary, Q&A, Quiz*.  
3. **Agent Invocation**: Specialized AI agent handles the request.  
4. **Knowledge Graph Retrieval**: Context fetched via semantic similarity + embeddings.  
5. **Response Generation**: Summaries, answers, or quizzes delivered.  
6. **Progress Tracking**: History agent logs interactions into a session report (PDF).  

---

## Technical Architecture  

### AI & Retrieval Layer
- **CrewAI** – Multi-agent orchestration  
- **LangChain** – Workflow chaining  
- **Chroma** – Vector database for embeddings  
- **Neo4j** – Graph database for curriculum structure  
- **Hugging Face Models** – Summarization, embeddings, ranking  
- **Gemini** – Image captioning for curriculum visuals  
---

##  Repository Overview

This repository contains the full implementation of **Etude.AI**.  
Each file and folder is organized by its responsibility in the system.

###  Core Intelligence
- **agents.py** → Implements the AI agents (Summary, Q&A, Quiz, History).  
- **retrieval.py** → Embedding-based context search and semantic retrieval.  
- **utils_text.py** → Helper functions for summarization, text formatting, and cleaning.  
- **runtime.py** → Runtime utilities for orchestrating jobs and managing execution.  

###  Application Layer
- **app.py** → FastAPI application setup and initialization.  
- **main.py** → Entry point to run the FastAPI server (`uvicorn main:app`).  
- **handlers.py** → Request handlers that route API calls to the right agents.  
- **cli.py** → Command-line tool for testing agents without the frontend.  

###  Knowledge & Data
- **kg.py** → Integration with Neo4j Knowledge Graph (nodes, relationships, queries).  
- **config_files/** → Stores model, database, and system configuration files.  
- **lessons/** → Educational lesson content used by the platform.  

###  Processing & Media
- **ocr_pdf.py** → Extracts and parses textual content from PDFs using OCR.  
- **images.py** → Extracts curriculum images and generates captions.  
- **pdf_report.py** → Creates structured student progress reports.  

### Reports & Outputs
- **reports/** → Stores generated reports (PDFs).  
- **session_report.pdf** → Example of a generated student session report.  

##  Tech Stack

* **Python** (FastAPI, LangChain, CrewAI, Hugging Face, PyMuPDF)
* **Neo4j** (Knowledge Graph storage & querying)
* **Chroma** (vector DB for embeddings)
* **Angular** (frontend UI)
* **ElevenLabs** (Text-to-Speech API)

---

## Acknowledgments

Developed as part of the **Artificial Intelligence National Summit (AINS) Hackathon 2025**, *Track 3: Tunisian Dialect-Aware AI* alongside Seifeddine Hamdi.


---

Do you also want me to include a **"Getting Started" section** (installation + run commands) so that anyone cloning your repo can launch it immediately?
```
