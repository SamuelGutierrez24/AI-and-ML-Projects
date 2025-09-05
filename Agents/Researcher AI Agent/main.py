from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from tools import search_tool, wiki_tool, save_pdf_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    references: list[str]
    tools_used: list[str]

#llm = ChatOpenAI(model="gpt-3.5-turbo")
#llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
llm3 = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text \n{format_instructions}

            """,
         ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions().replace("json", "text"))

tools = [search_tool, wiki_tool, save_pdf_tool]
agent = create_tool_calling_agent(
    llm=llm3,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("What can I help you research? ")
response = agent_executor.invoke({"query": query})

try:
    structured_output = parser.parse(response.get("output"))
    print(structured_output)
except Exception as e:
    print("Error parsing output:", e)
    