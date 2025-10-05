from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain.agents import initialize_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
import json
import os
import warnings

# Ignorar todos os avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Defina sua chave de acesso 
load_dotenv()
API_KEY = os.getenv("API_KEY")
NASA_API_KEY = os.getenv("NASA_API_KEY")


global retriever

def rag(documento):
    global retriever

    # Exemplo com URL
    urls = [documento]

    # Load documentos
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    
    # Split documentos
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", "\n", ". ", ", ", " ", ""], 
                                                   chunk_size=200, 
                                                   chunk_overlap=20)
    doc_splits = text_splitter.split_documents(docs)
    
    # Create VectorStore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="docs",
        embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever


# Tool para RAG
@tool
def retrieve_context(query: str):
    """Pesquise notícias recentes sobre astronomia."""
    global retriever
    results = retriever.invoke(query)
    print(results)
    return "\n".join([doc.page_content for doc in results])

tools = [retrieve_context]

# LLM model
model = ChatDeepInfra(
            model= "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            temperature=0,
            max_tokens = 256,
            deepinfra_api_token=API_KEY
        )

agent_assistente = initialize_agent(
    tools, 
    model,
    agent="zero-shot-react-description",  # usa ReAct pronto
    verbose=True
)

agent_moderador = model


class State(TypedDict):
    messages: List[BaseMessage] 
    documento: str
    avaliacao: str
    feedback: str


# Nó moderador
def moderador(state: State):

    # Pega a última mensagem AI. O caso 2 é para funcionar no langsmith.
    last_ai_message = None
    for msg in reversed(state["messages"]):
        # caso 1: já é AIMessage
        if isinstance(msg, AIMessage):
            last_ai_message = msg.content
            break
        # caso 2: veio como dict serializado
        if isinstance(msg, dict) and msg.get("type") == "ai":
            last_ai_message = msg.get("content")
            break
    
    # Pega a última mensagem humana. O caso 2 é para funcionar no langsmith.
    last_human_message = None
    for msg in reversed(state["messages"]):
        # caso 1: já é HumanMessage
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break
        # caso 2: veio como dict serializado
        if isinstance(msg, dict) and msg.get("type") == "human":
            last_human_message = msg.get("content")
            break

    # Prompt template
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um moderador de RESPOSTAS de um agente que só deve falar sobre o tema astronomia."),
        ("human", 
         """Dada a Pergunta e a Resposta, responda se a Resposta está adequada para a Pergunta.
         Responda "sim" se a resposta estiver adequada e "não" se a resposta não estiver adequada.
         
         Se sua resposta for não, dê um feedback curto para que o agente assistente melhore sua resposta.
         
         Responda APENAS em JSON válido no formato:
         {{"resposta": "sim", "feedback": "Você não possui feedback."}} 
         ou 
         {{"resposta": "não", "feedback": "Escreva seu feedback aqui"}}.

         Pergunta: {pergunta}
         Resposta: {resposta}"""
        )
    ])

    chain = prompt | agent_moderador | parser
    try:
        result = chain.invoke({"pergunta": last_human_message, "resposta": last_ai_message})
        resposta = result.get("resposta", "").strip()
        feedback = result.get("feedback", "").strip()

    except Exception as e:
        print("Erro ao interpretar resposta:", e)
        resposta = "sim"
        feedback = "Você não possui feedback."

    print({"avaliacao": resposta,"feedback": feedback})

    state["avaliacao"] = resposta
    state["feedback"] = feedback

    return state

# Nó assitente
def assistente(state: State):
    global retriever
    retriever = rag(state["documento"])
    feedback = state.get("feedback", "")

    # Pega a última mensagem humana. O caso 2 é para funcionar no langsmith.
    last_human_message = None
    for msg in reversed(state["messages"]):
        # caso 1: já é HumanMessage
        if isinstance(msg, HumanMessage):
            last_human_message = msg.content
            break
        # caso 2: veio como dict serializado
        if isinstance(msg, dict) and msg.get("type") == "human":
            last_human_message = msg.get("content")
            break

    prompt_system = f"""
    Você é um assistente de astronomia super gentil que se comunica em português.
    """
    prompt_assistente = f""" 
    Responda a Pergunta usando a dica para formular melhor sua resposta.

    Pergunta: {last_human_message}
    Dica: {feedback}

    Responda APENAS em JSON válido no formato:
    {{"resposta": "sua resposta aqui"}} 

    """
    messages = [SystemMessage(content=prompt_system), HumanMessage(content=prompt_assistente)]
    response = agent_assistente.invoke(messages)
    
    result = json.loads(response['output'])

    resposta = result.get("resposta", "").strip()

    state["messages"] = state["messages"] + [AIMessage(content=resposta)]

    return state


# Nó roteador
def roteador(state: State):
    """Roteia para o agente_assistente ou finaliza."""
    avaliacao = state.get("avaliacao")
    
    if avaliacao == 'não':
            return "refazer"

    return 'fim'

# Definindo o grafo
workflow = StateGraph(State)

# Adicionando os nós
tool_node = ToolNode(tools=tools)
workflow.add_node("assistente", assistente)
workflow.add_node("tools", tool_node)
workflow.add_node("moderador", moderador)

# Conectando os nós
workflow.add_edge(START, "assistente")  
workflow.add_conditional_edges("assistente", tools_condition)  
workflow.add_edge("tools", "assistente")  
workflow.add_edge("assistente", "moderador")
workflow.add_conditional_edges(
    "moderador",
    roteador,
    {
        "refazer": "assistente",
        "fim": END
    }
)

# Para incluir memória inclua checkpointer no compile.
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer) 

# Execução 
final_state = app.invoke(
    {"messages": [HumanMessage(content="Olá! Quais as novidades astronômicas de hoje?"),],
        "documento": 'https://www.nasa.gov/news/recently-published/',
        "avaliacao": "",
        "feedback": ""
     },
    config={"configurable": {"api_key": API_KEY, "thread_id": 42}}
)

# Estado final do grafo
print(final_state)


# Visualize o grafo
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
