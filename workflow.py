from typing import Literal
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.chat_models import ChatDeepInfra
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph.message import add_messages
from langchain import hub
from langchain.agents import create_react_agent, initialize_agent
from langchain_community.tools.tavily_search import TavilySearchResults
import json
import os
import warnings

# Ignorar todos os avisos de depreciação
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Defina sua chave de acesso 
load_dotenv()
API_KEY = os.getenv("API_KEY")


global retriever

def rag(documento):
    global retriever

    # Exemplo com URL
    urls = [
        documento
    ]
    # Load documentos
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    
    # Split documentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)
    
    # Create VectorStore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="docs",
        embedding = DeepInfraEmbeddings(model_id="BAAI/bge-base-en-v1.5", deepinfra_api_token=API_KEY),
    )
    retriever = vectorstore.as_retriever()
    return retriever


# Tool para RAG
@tool
def retrieve_context(query: str):
    """Search for relevant documents."""
    global retriever
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

tools = [retrieve_context]

# LLM model
model = ChatDeepInfra(
            model= "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            temperature=0,
            deepinfra_api_token=API_KEY
        )

agent_executor = initialize_agent(
    tools, 
    model,
    agent="zero-shot-react-description",  # usa ReAct pronto
    verbose=True
)

agent_avaliador = model



class State(TypedDict):
    messages: List[BaseMessage]
    documento: str
    avaliacao: str
    feedback: str
    contador: int


# Nó avaliador
def avaliador(state: State):
    contador = state["contador"]

    last_ai_message = None
    last_human_message = None

    # Pega a última mensagem AI. O caso 2 é para funcionar no langsmith.
    for msg in reversed(state["messages"]):
        # caso 1: já é HumanMessage
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

    print(f"ultima msg usuário: {last_human_message}")
    print(f"ultima msg ai: {last_ai_message}")

    prompt_avaliador = f"""
    Você é um avaliador pouco criterioso de RESPOSTAS de um agente que é assistente de astronomia.
    Não seja muito rigoroso, o agente assistente não possui muitas fontes de consulta.
    Dada a Pergunta e a Resposta, responda se a Resposta está adequada para a Pergunta.
    Responda "sim" se a resposta estiver adequada e "não" se a resposta não estiver adequada.
 
    Se sua resposta for não, dê um feedback curto para que o agente assistente melhore sua resposta.
    
    Responda APENAS em JSON válido no formato:
    {{"resposta": "sim", "feedback": "Você não possui feedback."}} 
    ou 
    {{"resposta": "não", "feedback": "Escreva seu feedback aqui"}}.

    Pergunta: {last_human_message}
    Resposta: {last_ai_message}
    """

    messages = [HumanMessage(content=prompt_avaliador)]
    response = agent_avaliador.invoke(messages)

    try:
        result = json.loads(response.content)
        resposta = result.get("resposta", "").strip()
        feedback = result.get("feedback", "").strip()
    except Exception as e:
        print("Erro ao interpretar resposta:", e)
        resposta = "sim"
        feedback = "Você não possui feedback."

    if resposta == "não":
        contador = state["contador"] + 1

    if contador > 5:
        resposta = "sim"
        feedback = "Você não possui feedback."

    print({
        "avaliacao": resposta,
        "feedback": feedback,
        "contador": contador

    })
    return {
        "avaliacao": resposta,
        "feedback": feedback,
        "contador": contador
    }


# Nó executor
def executor(state: State):
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

    prompt_executor = f"""Você é um assistente de astronomia. 
    Responda a Pergunta usando a dica para formular melhor sua resposta.

    Pergunta: {last_human_message}
    Dica: {feedback}

    Responda APENAS em JSON válido no formato:
    {{"resposta": "sua resposta aqui"}} 

    """
    messages = [HumanMessage(content=prompt_executor)]
    response = agent_executor.invoke(messages)
    
    result = json.loads(response['output'])
    # result = json.loads(response.content)

    resposta = result.get("resposta", "").strip()

    new_messages = state["messages"] + [AIMessage(content=resposta)]

    return {
        "messages": new_messages
    }


# Nó roteador
def roteador(state: State):
    """Roteia para o agente_executor ou finaliza."""
    avaliacao = state.get("avaliacao")
    
    if avaliacao == 'não':
            return "refazer"

    return 'fim'


# Nó que finaliza o workflow
def finaliza(state: State):
    """Finaliza o workflow e retorna a resposta do agente_executor"""
    return {"messages": state['messages']}

# Definindo o workflow
workflow = StateGraph(State)

# Adicionando os nós
tool_node = ToolNode(tools=tools)
workflow.add_node("agent_executor", executor)
workflow.add_node("tools", tool_node)
workflow.add_node("agent_avaliador", avaliador)
workflow.add_node("finaliza", finaliza)

# Conectando os nós
workflow.add_edge(START, "agent_executor")  
workflow.add_conditional_edges("agent_executor", tools_condition)  
workflow.add_edge("tools", "agent_executor")  
workflow.add_edge("agent_executor", "agent_avaliador")
workflow.add_conditional_edges(
    "agent_avaliador",
    roteador,
    {
        "refazer": "agent_executor",
        "fim": "finaliza"
    }
)
workflow.add_edge("finaliza", END)


# Para incluir memória inclua checkpointer no compile.
checkpointer = MemorySaver()

# No langsmith, não use o checkpointer, pois a própria ferramenta já salva o histórico.
app = workflow.compile()

# Sem o langsmith
app = workflow.compile(checkpointer=checkpointer) 


# Desabilite o app.invoke para executar com o langsmith
final_state = app.invoke(
    {"messages": [HumanMessage(content="Olá! O que é o 3I/ATLAS?")],
     "documento": 'https://science.nasa.gov/solar-system/comets/3i-atlas/',
     "contador": 1
     },
    config={"configurable": {"api_key": API_KEY, "thread_id": 42}}
)

# Show the final response
print(final_state)
