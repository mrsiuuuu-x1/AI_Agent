from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatAnthropic(model="claude-3-5-sonnet-20240924")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant that gathers information on a given topic and provides a summary, sources, and tools used. Research the topic: {topic} and provide a summary, sources, and tools used.")
    ("human", "{query}"),
]).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "What are the latest advancements in AI research?", "name": "ResearchAgent"})
print(raw_response)