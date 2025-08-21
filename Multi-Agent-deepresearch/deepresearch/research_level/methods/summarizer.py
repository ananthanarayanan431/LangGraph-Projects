
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from deepresearch.utils.helper import get_today_str
from deepresearch.research_level.prompt import SUMMARIZE_WEBPAGE_PROMPT
from deepresearch.research_level.model import Summary

summarization_model = init_chat_model(model="openai:gpt-4.1-mini")


def summarize_webpage_content(webpage_content: str)->str:
    """Summarize the webpage content"""

    try:
        structured_model = summarization_model.with_structured_output(Summary)
        template = SUMMARIZE_WEBPAGE_PROMPT.format(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        summary = structured_model.invoke([HumanMessage(content=template)])
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        return formatted_summary
    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
