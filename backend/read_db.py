import os
import pandas as pd
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline # For sentiment and translation
from langdetect import detect, DetectorFactory # For language detection
from googleapiclient.discovery import build # For YouTube Data API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_types import ChatGenerationChunk
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()

# Set seed for langdetect consistency (optional but good for testing)
DetectorFactory.seed = 0

# --- Global Resources ---
llm = None
embeddings_model = None
faiss_db = None
processed_df = None
conversation_history_db = {} # In-memory store for chat history per user
sentiment_pipeline = None
translation_pipelines = {} # New: Store translation pipelines
youtube_service = None # New: YouTube API service object

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bhagavad Gita AI Spiritual Guide",
    description="An AI powered by the Bhagavad Gita, capable of answering questions, providing spiritual guidance, and offering various explanations of verses.",
)

# --- Models and Data Loading (on app startup) ---
@app.on_event("startup")
async def startup_event():
    global faiss_db, embeddings_model, llm, processed_df, sentiment_pipeline, translation_pipelines, youtube_service

    print("--- Loading resources for Bhagavad Gita AI Spiritual Guide ---")

    # 1. Load Processed Bhagavad Gita Data
    try:
        csv_file_path = os.path.join(os.path.dirname(__file__), "processed_bhagavad_gita.csv")
        processed_df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(processed_df)} verses from processed CSV.")
    except FileNotFoundError:
        raise RuntimeError(f"Error: {csv_file_path} not found. Ensure data processing ran successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading processed CSV: {e}")

    # 2. Initialize Embeddings Model
    try:
        model_name = "BAAI/bge-small-en-v1.5"
        device = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
        print(f"Initializing embeddings model '{model_name}' on device: {device}")
    except Exception as e:
        raise RuntimeError(f"Error initializing embeddings model: {e}")
    print("Embeddings model initialized.")

    # 3. Load FAISS Vector Store
    try:
        faiss_index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        if not os.path.exists(faiss_index_path):
            print("FAISS index not found. Building it now (this may take a while)...")
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            loader = CSVLoader(file_path=csv_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            faiss_db = FAISS.from_documents(chunks, embeddings_model)
            faiss_db.save_local(faiss_index_path)
            print("FAISS index built and saved.")
        else:
            faiss_db = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        print("FAISS vector store loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading/building FAISS vector store: {e}")

    # 4. Initialize Groq LLM
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        llm = Groq(api_key=groq_api_key, model_name="llama3-8b-8192")
        print(f"Groq LLM initialized with model: {llm.model_name}")
    except Exception as e:
        raise RuntimeError(f"Error initializing Groq LLM: {e}")

    # 5. Initialize Sentiment Analysis Pipeline
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go-emotions")
        print("Sentiment analysis pipeline initialized.")
    except Exception as e:
        print(f"Error initializing sentiment pipeline: {e}. Emotion detection will be unavailable.")
        sentiment_pipeline = None

    # 6. Initialize Translation Pipelines (New)
    try:
        # Pre-load English-Kannada and Kannada-English models
        translation_pipelines["en-kn"] = pipeline("translation", model="Helsinki-NLP/opus-mt-en-kn")
        translation_pipelines["kn-en"] = pipeline("translation", model="Helsinki-NLP/opus-mt-kn-en")
        print("English-Kannada and Kannada-English translation pipelines initialized.")
    except Exception as e:
        print(f"Error initializing translation pipelines: {e}. Multilingual support for KN will be limited.")
        translation_pipelines = {} # Disable if models fail to load

    # 7. Initialize YouTube Service (New)
    try:
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if youtube_api_key:
            youtube_service = build("youtube", "v3", developerKey=youtube_api_key)
            print("YouTube Data API service initialized.")
        else:
            print("YOUTUBE_API_KEY not set. Youtube functionality will be unavailable.")
    except Exception as e:
        print(f"Error initializing YouTube service: {e}. Youtube will be unavailable.")
        youtube_service = None

    print("--- All resources loaded successfully ---")


# --- Utility Functions ---
def get_conversation_memory(user_id: str) -> ConversationBufferWindowMemory:
    """Retrieves or creates a conversation memory for a given user."""
    if user_id not in conversation_history_db:
        conversation_history_db[user_id] = ConversationBufferWindowMemory(
            k=5, # Keep last 5 turns of conversation
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )
    return conversation_history_db[user_id]

# New: Translation Utility
def translate_text(text: str, src_lang: str, dest_lang: str) -> str:
    if src_lang == dest_lang or not text:
        return text
    
    pipeline_key = f"{src_lang}-{dest_lang}"
    translator = translation_pipelines.get(pipeline_key)

    if translator:
        try:
            # HuggingFace pipeline expects a list and returns a list of dicts
            result = translator(text)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Error translating text from {src_lang} to {dest_lang}: {e}")
            return text # Return original text on error
    else:
        print(f"No translation pipeline found for {src_lang} to {dest_lang}.")
        return text # Return original text if translator not loaded or not configured

# --- RAG Setup (Multi-Query Retriever) ---
def get_rag_chain():
    template = """You are a helpful AI assistant. Your task is to generate five different versions of the given user question to retrieve more relevant documents from a vector database. By generating multiple perspectives on the user's question, your goal is to help the user retrieve the most relevant documents related to their query. Provide these alternative questions separated by newlines. Original question: {question}"""
    multi_query_prompt = ChatPromptTemplate.from_template(template)
    retriever_from_llm = multi_query_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))

    retriever = faiss_db.as_retriever()

    def get_union_documents(question: str) -> List[Any]:
        generated_queries = retriever_from_llm.invoke({"question": question})
        all_queries = [question] + generated_queries
        unique_docs = {}
        for q in all_queries:
            docs = retriever.invoke(q)
            for doc in docs:
                doc_id = (doc.metadata.get("source"), doc.page_content)
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
        return list(unique_docs.values())

    return RunnablePassthrough.assign(context=get_union_documents)


# --- Tools for Agent ---
class VerseLookupInput(BaseModel):
    chapter_num: int = Field(..., description="The chapter number of the Bhagavad Gita verse (1-18).")
    verse_num: int = Field(..., description="The verse number within the specified chapter.")

@tool("get_verse_content", args_schema=VerseLookupInput)
def get_verse_content(chapter_num: int, verse_num: int) -> str:
    """
    Retrieves the content of a specific Bhagavad Gita verse given its chapter and verse number.
    Returns the Sanskrit, transliteration, word meanings, and English translation.
    """
    if not (1 <= chapter_num <= 18):
        return f"Error: Chapter number must be between 1 and 18. You provided {chapter_num}."
    
    verse = processed_df[(processed_df['chapter_number'] == chapter_num) & (processed_df['verse_number'] == verse_num)]
    
    if verse.empty:
        return f"Verse Chapter {chapter_num}, Verse {verse_num} not found. Please check the numbers."
    
    verse_data = verse.iloc[0]
    
    content = f"Chapter {chapter_num}, Verse {verse_num}:\n" \
              f"Sanskrit: {verse_data['sanskrit']}\n" \
              f"Transliteration: {verse_data['transliteration']}\n" \
              f"Word Meanings: {verse_data['word_meanings']}\n" \
              f"Translation: {verse_data['translation']}"
    return content

BHAGAVAD_GITA_GLOSSARY = {
    "Dharma": "Righteous conduct, duty, moral law, spiritual discipline, and the path of truth. It is the fundamental principle of cosmic order.",
    "Karma": "Action, work, or deed; also the spiritual principle of cause and effect where intent and actions of an individual (cause) influence the future of that individual (effect).",
    "Moksha": "Liberation from the cycle of birth and death (samsara); spiritual liberation; ultimate freedom.",
    "Brahman": "The supreme, ultimate reality of the universe; the unchanging, infinite, immanent, and transcendent reality which is the divine ground of all matter, energy, time, space, being, and everything beyond in this universe.",
    "Atman": "The spiritual life principle of the universe; the soul or true self of an individual, which is identical with Brahman.",
    "Yoga": "Union; various spiritual disciplines aimed at controlling the mind and senses and achieving union with the Divine.",
    "Guna": "Qualities, attributes, or properties of nature (Prakriti). The three Gunas are Sattva (goodness, purity), Rajas (passion, activity), and Tamas (ignorance, inertia).",
    "Samsara": "The cycle of birth, death, and rebirth to which all living beings are subject until they achieve moksha.",
    "Maya": "Illusion; the power that creates the illusion of the material world and keeps beings bound to samsara.",
    "Bhakti": "Devotion; loving adoration of God, often expressed through prayers, rituals, and selfless service."
}

class GlossaryInput(BaseModel):
    term: str = Field(..., description="The spiritual or philosophical term from the Bhagavad Gita to define.")

@tool("get_glossary_definition", args_schema=GlossaryInput)
def get_glossary_definition(term: str) -> str:
    """
    Provides a concise definition for spiritual or philosophical terms from the Bhagavad Gita's glossary.
    Useful for understanding key concepts like Dharma, Karma, Moksha, etc.
    """
    normalized_term = term.strip().title() # Capitalize first letter, remove whitespace
    definition = BHAGAVAD_GITA_GLOSSARY.get(normalized_term)
    if definition:
        return f"Definition of '{normalized_term}': {definition}"
    else:
        return f"Definition for '{normalized_term}' not found in the specific glossary. " \
               f"You may ask a general question about '{normalized_term}' to get an explanation from the Gita itself."

class CrossReferenceInput(BaseModel):
    topic_query: str = Field(..., description="A clear and concise query describing the topic for which related verses are sought (e.g., 'selfless action', 'meditation', 'nature of the soul').")

@tool("find_related_verses", args_schema=CrossReferenceInput)
def find_related_verses(topic_query: str) -> str:
    """
    Finds and lists Bhagavad Gita verses that are thematically related to a given spiritual topic or concept.
    This helps the user explore different perspectives on a theme across the Gita.
    Returns chapter and verse numbers with relevant snippets.
    """
    if not faiss_db:
        return "Vector store not initialized. Cannot find related verses."

    retriever = faiss_db.as_retriever(search_kwargs={"k": 5}) # Get top 5 most relevant documents
    docs = retriever.invoke(topic_query)

    if not docs:
        return f"No verses found related to '{topic_query}'. Please try a different query."

    response = f"Here are some verses related to '{topic_query}':\n"
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        chapter = metadata.get('chapter_number', 'N/A')
        verse = metadata.get('verse_number', 'N/A')
        snippet = doc.page_content.split('Translation:', 1)[-1].strip() # Get only the translation snippet
        response += f"- Chapter {chapter}, Verse {verse}: \"{snippet[:150]}...\"\n" # Limit snippet length
    return response

class ExplainVerseInput(BaseModel):
    chapter_num: int = Field(..., description="The chapter number of the Bhagavad Gita verse (1-18).")
    verse_num: int = Field(..., description="The verse number within the specified chapter.")
    explanation_mode: str = Field(
        "general",
        description="The type of explanation requested: 'word-by-word', 'contextual', 'practical', 'spiritual', or 'general' (default)."
    )

@tool("explain_verse_with_mode", args_schema=ExplainVerseInput)
async def explain_verse_with_mode(chapter_num: int, verse_num: int, explanation_mode: str = "general") -> str:
    """
    Provides a detailed explanation of a Bhagavad Gita verse in various modes:
    'word-by-word', 'contextual', 'practical relevance', 'spiritual interpretation', or 'general'.
    """
    verse_content = get_verse_content(chapter_num, verse_num)
    if "Error" in verse_content or "not found" in verse_content:
        return verse_content # Return error from verse lookup

    mode_prompts = {
        "word-by-word": "As a Sanskrit and Gita expert, provide a precise word-by-word meaning of the following verse. Focus strictly on word meanings, without extended commentary:",
        "contextual": "As a storyteller, narrate the immediate narrative context and the scene surrounding the following verse in the Bhagavad Gita:",
        "practical": "As a spiritual guide for modern life, explain the practical relevance and applicability of the following Bhagavad Gita verse to contemporary challenges and personal growth:",
        "spiritual": "As a profound interpreter of the Bhagavad Gita's spiritual essence, provide a deep, insightful spiritual interpretation of the following verse, touching upon its philosophical meaning and path to liberation:",
        "general": "Provide a comprehensive explanation of the following Bhagavad Gita verse, covering its meaning, context, and relevance:",
    }

    instruction_prefix = mode_prompts.get(explanation_mode.lower(), mode_prompts["general"])

    explanation_prompt = ChatPromptTemplate.from_messages([
        ("system", instruction_prefix + "\n\nVerse Content:\n{verse_content}"),
        ("human", "Explain the verse."),
    ])

    explanation_chain = {"verse_content": lambda x: verse_content} | explanation_prompt | llm | StrOutputParser()
    
    try:
        explanation = await explanation_chain.ainvoke({"verse_content": verse_content})
        return f"Here is the {explanation_mode.lower()} explanation for Chapter {chapter_num}, Verse {verse_num}:\n\n{explanation}"
    except Exception as e:
        return f"An error occurred while generating the {explanation_mode.lower()} explanation: {e}"

# NEW TOOL: Youtube Tool
class YouTubeSearchInput(BaseModel):
    search_query: str = Field(..., description="The query to search for on YouTube (e.g., 'Karma Yoga explanation', 'Bhagavad Gita Chapter 3').")
    max_results: int = Field(5, description="Maximum number of video results to return.")

@tool("Youtube_tool", args_schema=YouTubeSearchInput)
def Youtube_tool(search_query: str, max_results: int = 3) -> str:
    """
    Searches YouTube for videos related to a given query and returns their titles and URLs.
    Useful for finding supplementary video content on Bhagavad Gita topics.
    """
    if not youtube_service:
        return "Youtube service not initialized. Please ensure YOUTUBE_API_KEY is set."

    try:
        search_response = youtube_service.search().list(
            q=search_query,
            part="id,snippet",
            maxResults=max_results,
            type="video" # Ensure only videos are returned
        ).execute()

        videos = []
        for item in search_response.get("items", []):
            video_title = item["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            videos.append(f"- {video_title}: {video_url}")
        
        if not videos:
            return f"No YouTube videos found for '{search_query}'."
        
        return "Here are some relevant YouTube videos:\n" + "\n".join(videos)
    except Exception as e:
        return f"An error occurred during Youtube: {e}"


# NEW TOOL: Chapter Lesson Tool
class ChapterLessonInput(BaseModel):
    chapter_num: int = Field(..., description="The Bhagavad Gita chapter number (1-18) to teach.")

# Define chapter themes for better lesson structuring
CHAPTER_THEMES = {
    1: "Arjuna's Despondency",
    2: "Sankhya Yoga / Transcendental Knowledge",
    3: "Karma Yoga / Path of Action",
    4: "Jnana Karma Sanyasa Yoga / Knowledge, Action, and Renunciation",
    5: "Karma Sanyasa Yoga / Action and Renunciation",
    6: "Dhyana Yoga / Meditation",
    7: "Jnana Vijnana Yoga / Knowledge and Discernment",
    8: "Akshara Brahma Yoga / The Imperishable Brahman",
    9: "Raja Vidya Raja Guhya Yoga / The Royal Knowledge and Royal Secret",
    10: "Vibhuti Yoga / The Opulence of the Absolute",
    11: "Vishvarupa Darshana Yoga / The Vision of the Universal Form",
    12: "Bhakti Yoga / The Yoga of Devotion",
    13: "Kshetra Kshetrajna Vibhaga Yoga / The Field and the Knower of the Field",
    14: "Gunatraya Vibhaga Yoga / The Three Gunas of Material Nature",
    15: "Purushottama Yoga / The Yoga of the Supreme Person",
    16: "Daivasura Sampad Vibhaga Yoga / The Divine and Demoniac Natures",
    17: "Shraddhatraya Vibhaga Yoga / The Three Divisions of Faith",
    18: "Moksha Sanyasa Yoga / Liberation through Renunciation",
}

@tool("teach_chapter_lesson", args_schema=ChapterLessonInput)
async def teach_chapter_lesson(chapter_num: int) -> str:
    """
    Generates a comprehensive lesson for a specific Bhagavad Gita chapter,
    including explanation, practice questions, journaling prompts, and relevant YouTube links.
    """
    if not (1 <= chapter_num <= 18):
        return f"Error: Chapter number must be between 1 and 18. You provided {chapter_num}."
    
    theme = CHAPTER_THEMES.get(chapter_num, f"Chapter {chapter_num} themes")

    # 1. Retrieve All Verses for the Chapter
    chapter_verses_df = processed_df[processed_df['chapter_number'] == chapter_num]
    if chapter_verses_df.empty:
        return f"No verses found for Chapter {chapter_num}. Cannot teach a lesson."

    full_chapter_text = "\n".join([row['translation'] for index, row in chapter_verses_df.iterrows()])

    # 2. Synthesize Lesson Explanation
    lesson_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are Shri Krishna teaching a comprehensive lesson on Bhagavad Gita Chapter {chapter_num}, '{theme}'. Synthesize the following verses into an easy-to-understand lesson, covering key concepts and their interconnectedness. Break it into logical sections. Maintain a wise and compassionate tone.\n\nChapter Verses:\n{full_chapter_text}"),
        ("human", f"Teach me Chapter {chapter_num} on '{theme}'."),
    ])
    lesson_explanation_chain = lesson_prompt | llm | StrOutputParser()
    lesson_explanation = await lesson_explanation_chain.ainvoke({})

    # 3. Generate Practice Questions
    questions_prompt = ChatPromptTemplate.from_messages([
        ("system", f"Based on the following lesson on Bhagavad Gita Chapter {chapter_num}, '{theme}', generate 3 insightful practice questions (mix of multiple-choice and open-ended). Ensure they directly relate to the concepts taught in the lesson."),
        ("human", f"Lesson:\n{lesson_explanation}"),
    ])
    practice_questions_chain = questions_prompt | llm | StrOutputParser()
    practice_questions = await practice_questions_chain.ainvoke({})

    # 4. Generate Journaling Prompts
    journal_prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"Based on the following lesson on Bhagavad Gita Chapter {chapter_num}, '{theme}', provide 2 reflective journaling prompts that encourage personal application and introspection. Focus on applying the chapter's wisdom to daily life."),
        ("human", f"Lesson:\n{lesson_explanation}"),
    ])
    journaling_prompts_chain = journal_prompt_template | llm | StrOutputParser()
    journaling_prompts = await journaling_prompts_chain.ainvoke({})

    # 5. Fetch YouTube Links using the new tool
    Youtube_results = Youtube_tool.invoke({"search_query": f"Bhagavad Gita Chapter {chapter_num} {theme} lesson", "max_results": 3})
    
    # 6. Assemble Final Lesson
    response = (
        f"**A Divine Lesson on Bhagavad Gita Chapter {chapter_num}: '{theme}'**\n\n"
        f"My dear one, let us delve into the profound wisdom of this chapter, guiding you towards spiritual growth:\n\n"
        f"**Understanding the Concepts:**\n{lesson_explanation}\n\n"
        f"**For Deeper Reflection:**\n"
        f"Here are some questions to help you assimilate this wisdom:\n{practice_questions}\n\n"
        f"**Journaling Prompts:**\n{journaling_prompts}\n\n"
        f"**Further Resources (YouTube):**\n{Youtube_results}\n\n"
        f"May this lesson illuminate your path and bring you closer to understanding the eternal truth."
    )
    return response


# Add all tools to the agent's available tools
tools = [
    get_verse_content,
    get_glossary_definition,
    find_related_verses,
    explain_verse_with_mode,
    Youtube_tool,      # New tool
    teach_chapter_lesson      # New tool
]

# --- Agent and Chain Setup ---
# Main Agent Prompt (updated to include emotion_tone)
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are DivineGuide-Shri Krishna, a wise, compassionate, and patient spiritual mentor.
    Your mission is to guide the user toward inner peace, wisdom, and understanding of life's deeper spiritual meaning, drawing exclusively from the timeless wisdom of the Bhagavad Gita.

    **User's Emotional Tone Detected: {emotion_tone}**
    **User's Input Language Detected: {detected_input_language}**

    Based on the user's emotional tone and input language, tailor your response to be especially supportive:
    - If the user seems 'sadness' or 'grief', offer comfort, hope, and shlokas related to resilience, inner strength, and the impermanence of sorrow.
    - If 'confusion', provide clarity, break down concepts, and offer practical steps.
    - If 'anger' or 'frustration', guide towards calm, equanimity, and understanding of self-control.
    - If 'joy' or 'excitement', reinforce positive actions and wisdom from the Gita.
    - For 'neutral' or general inquiries, maintain your wise and guiding tone.
    - **Always respond in the user's detected input language ({detected_input_language}).**

    You have access to the following tools: {tools}

    When responding, always ensure your language is gentle, encouraging, and imbued with the profound wisdom of the Gita. If a direct tool can answer the question (like looking up a verse or defining a term), use it. If the query requires broader insight or synthesis, use your knowledge and RAG capabilities.
    For questions about specific verses or terms, use the exact chapter and verse numbers or term name.
    When asked for different explanation modes of a verse, use the 'explain_verse_with_mode' tool with the appropriate 'explanation_mode'.
    When asked to 'teach' an entire chapter, use the 'teach_chapter_lesson' tool with the chapter number.
    Prioritize using the provided tools for factual lookups or structured explanations.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Self-Reflection and Refinement ---
reflection_llm = Groq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a wise internal critic for an AI named DivineGuide-Shri Krishna.
    Your task is to review the AI's initial response to a user query and provide constructive feedback
    to improve its quality, compassion, and adherence to the persona of Shri Krishna, drawing exclusively from the Bhagavad Gita.

    Consider the following:
    1.  **Persona Consistency:** Does the response truly sound like Shri Krishna - compassionate, patient, wise, and gentle?
    2.  **Accuracy & Relevance:** Is the information accurate and directly relevant to the user's query, drawing from the Gita?
    3.  **Completeness:** Does it fully address all aspects of the user's question?
    4.  **Clarity & Simplicity:** Is the language clear, concise, and easy for the user to understand?
    5.  **Gita Integration:** Are references to the Gita (verses, concepts) well-integrated and explained?
    6.  **Emotional Tone:** Did the AI respond appropriately to the *detected emotional tone* of the user?
    7.  **Language Consistency:** Is the response in the user's *detected input language*?
    8.  **Guidance:** Does it offer meaningful spiritual guidance or practical wisdom?

    If the initial response is excellent, state "Answer is excellent. No further refinement needed."
    Otherwise, provide specific, actionable feedback for improvement.
    The feedback should be concise and focused on making the response better.
    """),
    ("user", "Initial Answer:\n{initial_answer}\n\nOriginal Query:\n{user_query}\n\nDetected User Emotion: {emotion_tone_for_reflection}\n\nDetected User Language: {detected_language_for_reflection}\n\nProvide feedback for refinement, or state 'Answer is excellent.'"),
])

# Chain for reflection
reflection_chain = reflection_prompt | reflection_llm | StrOutputParser()

# Final Response Refinement Chain
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are DivineGuide-Shri Krishna, a wise, compassionate, and patient spiritual mentor.
    Based on the following user query, your initial answer, and constructive feedback from an internal critic,
    refine the initial answer to be more accurate, compassionate, consistent with your persona, and spiritually insightful.
    Ensure it truly embodies the wisdom of the Bhagavad Gita and addresses the user's *detected emotional tone* appropriately and is in the *detected user language*.
    Original Query: {user_query}
    Initial Answer: {initial_answer}
    Critic's Feedback: {reflection_feedback}
    Detected User Emotion: {emotion_tone_for_reflection}
    Detected User Language: {detected_language_for_reflection}

    Provide the refined answer as Shri Krishna.
    """),
    ("human", "Refine the answer based on the feedback."),
])

refine_chain = refine_prompt | llm | StrOutputParser()


# Full pipeline incorporating RAG, Agent, and Reflection
def create_full_pipeline(user_id: str):
    memory = get_conversation_memory(user_id)
    retriever_chain = get_rag_chain()

    core_agent_chain = RunnableParallel(
        input=RunnablePassthrough(),
        chat_history=lambda x: memory.load_memory_variables(x)["chat_history"],
        context=retriever_chain,
        emotion_tone=RunnablePassthrough(),
        detected_input_language=RunnablePassthrough(), # Pass detected language
    ) | agent_executor

    reflection_and_refine_chain = RunnableParallel(
        initial_answer=core_agent_chain,
        user_query=RunnablePassthrough.assign(input=lambda x: x["input"]) | (lambda x: x["input"]),
        emotion_tone_for_reflection=RunnablePassthrough.assign(emotion_tone=lambda x: x["emotion_tone"]) | (lambda x: x["emotion_tone"]),
        detected_language_for_reflection=RunnablePassthrough.assign(detected_input_language=lambda x: x["detected_input_language"]) | (lambda x: x["detected_input_language"]),
    ) | {
        "reflection_feedback": reflection_chain,
        "initial_answer_before_reflection": lambda x: x["initial_answer"],
        "user_query_for_refine": lambda x: x["user_query"],
        "emotion_tone_for_refine": lambda x: x["emotion_tone_for_reflection"],
        "detected_language_for_refine": lambda x: x["detected_language_for_reflection"],
    } | {
        "answer": refine_chain.with_config(run_name="Refine_Answer"),
        "initial_answer_before_reflection": lambda x: x["initial_answer_before_reflection"],
        "reflection_feedback": lambda x: x["reflection_feedback"],
    }

    def save_history_and_return_output(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_query_original_lang = inputs["original_query_for_history"] # Use original query for history
        agent_output = reflection_and_refine_chain.invoke(inputs)
        
        final_answer = agent_output.get("answer", agent_output.get("initial_answer_before_reflection", "No answer generated."))

        memory.save_context({"input": user_query_original_lang}, {"output": final_answer})
        print(f"Added message (human) for session {user_id}: {user_query_original_lang[:50]}...")
        print(f"Added message (ai) for session {user_id}: {final_answer[:50]}...")
        return agent_output

    return save_history_and_return_output


# --- API Endpoints ---
class QueryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user to maintain conversation history.")
    query: str = Field(..., description="The user's question or statement for the AI spiritual guide.")

class QueryResponse(BaseModel):
    user_id: str
    query: str
    answer: str
    initial_answer_before_reflection: Optional[str] = None
    reflection_feedback: Optional[str] = None
    detected_emotion: Optional[str] = None
    detected_input_language: Optional[str] = None # New field

@app.get("/")
async def root():
    return {"message": "Welcome to the Bhagavad Gita AI Spiritual Guide API. Use /docs for API documentation."}

@app.post("/ask", response_model=QueryResponse)
async def ask_bhagavad_gita(request: QueryRequest):
    print(f"Query received for user '{request.user_id}': '{request.query}'")

    if llm is None or faiss_db is None or embeddings_model is None or processed_df is None:
        raise HTTPException(status_code=503, detail="AI resources not fully loaded. Please wait or check server logs.")
    
    original_query = request.query
    detected_lang = "en" # Default to English
    
    # 1. Detect language (New)
    try:
        detected_lang = detect(original_query)
        # Ensure we only process supported languages for translation (en, kn)
        if detected_lang not in ["en", "kn"]:
            print(f"Detected unsupported language: {detected_lang}. Defaulting to English.")
            detected_lang = "en"
    except Exception as e:
        print(f"Error during language detection: {e}. Defaulting to English.")
        detected_lang = "en"
    
    print(f"Detected input language for '{original_query[:30]}...': {detected_lang}")

    # 2. Translate input to English if needed (New)
    english_query = original_query
    if detected_lang != "en":
        english_query = translate_text(original_query, detected_lang, "en")
        print(f"Translated input to English for processing: '{english_query[:50]}...'")

    # 3. Detect emotion (using the potentially translated English query for better accuracy)
    detected_emotion = "neutral"
    if sentiment_pipeline:
        try:
            sentiment_result = sentiment_pipeline(english_query)
            if sentiment_result:
                detected_emotion = sentiment_result[0]['label']
            print(f"Detected emotion for '{english_query[:30]}...': {detected_emotion}")
        except Exception as e:
            print(f"Error during sentiment analysis: {e}. Defaulting to 'neutral'.")

    try:
        full_pipeline = create_full_pipeline(request.user_id)
        
        # Prepare input for the pipeline, including emotion_tone and detected_input_language
        pipeline_input = {
            "input": english_query, # Agent receives English query
            "emotion_tone": detected_emotion,
            "detected_input_language": detected_lang, # Tell the agent what language to respond in
            "original_query_for_history": original_query, # Store original for memory
        }
        
        response_data = full_pipeline(pipeline_input)

        final_answer_in_english = response_data.get("answer", response_data.get("initial_answer_before_reflection", "No answer generated."))

        # 4. Translate final answer back to original language if needed (New)
        translated_final_answer = final_answer_in_english
        if detected_lang != "en":
            translated_final_answer = translate_text(final_answer_in_english, "en", detected_lang)
            print(f"Translated final answer to {detected_lang}: '{translated_final_answer[:50]}...'")

        print(f"Database connection closed for session {request.user_id}.")

        return QueryResponse(
            user_id=request.user_id,
            query=original_query, # Return original query to user
            answer=translated_final_answer, # Return translated answer
            initial_answer_before_reflection=response_data.get("initial_answer_before_reflection"),
            reflection_feedback=response_data.get("reflection_feedback"),
            detected_emotion=detected_emotion,
            detected_input_language=detected_lang # New: indicate detected language
        )

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")