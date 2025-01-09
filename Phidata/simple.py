
from dotenv import load_dotenv
load_dotenv()

from phi.agent import Agent
from phi.model.openai.chat import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources at the end"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# web_agent.print_response("Tell me about Launch agents by launch ventures",stream=True)

finance_agent = Agent(
    name="Finance Agent",
    role="Get the financial data",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,company_info=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

# finance_agent.print_response("Summarize analyst recommendations for Mastek",stream=True)

agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include source", "Use tables to display the table"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

agent_team.print_response("Summarize analyst recommendations and share the latest news for Apple",stream=True)
