import logging
import os, json, asyncio, traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from dotenv import load_dotenv
from anyio import ClosedResourceError
import urllib.parse
from odr import OpenDeepResearch 
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def odr_tool_async(topic: str):
    research = OpenDeepResearch()
    report = await research.generate_research_report(topic)
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    report_path = os.path.join(temp_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return (report, {"report_content": report, "report_path": report_path})

def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args_schema).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

async def create_agent(coral_tools, agent_tools):
    coral_tools_description = get_tools_description(coral_tools)
    agent_tools_description = get_tools_description(agent_tools)
    combined_tools = coral_tools + agent_tools

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a specialized research agent interacting with the tools from Coral Server and having your own tools. Your task is to perform any instructions coming from any agent.
                Follow these steps in order:

                1. Call wait_for_mentions from coral tools (timeoutMs: 30000) to receive mentions from other agents.
                2. When you receive a mention, keep the thread ID and the sender ID.
                3. Think about the content (instruction) of the message and check only from the list of your tools available for you to action.
                4. Check the tool schema and make a plan in steps for the task you want to perform.
                5. Only call the tools you need to perform for each step of the plan to complete the instruction in the content.
                6. Think about the content and see if you have executed the instruction to the best of your ability and the tools. Make this your response as "answer".
                7. Use send_message from coral tools to send a message in the same thread ID to the sender Id you received the mention from, with content: "answer" containing the report content and the file path where the report is saved.
                8. If any error occurs, use send_message to send a message in the same thread ID to the sender Id you received the mention from, with content: "error".
                9. Always respond back to the sender agent even if you have no answer or error.
                10. Repeat the process from step 1.

            These are the list of coral tools: {coral_tools_description}
            These are the list of your tools: {agent_tools_description}."""
        ),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4.1"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=os.getenv("MODEL_TEMPERATURE", "0.1"),
        max_tokens=os.getenv("MODEL_MAX_TOKENS", "8000"),
        base_url=os.getenv("MODEL_BASE_URL") if os.getenv("MODEL_BASE_URL") else None
    )

    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "The Open Deep Research agent is an open-source research assistant.  It can perform in-depth web searches, generate structured reports, support human-in-the-loop feedback, and integrate with APIs like Tavily, Linkup, DuckDuckGo, and Azure AI Search, using customizable LLMs for tailored, high-quality research outputs."
    }

    query_string = urllib.parse.urlencode(coral_params)

    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    logger.info(f"Connecting to Coral Server: {CORAL_SERVER_URL}")
    
    timeout = os.getenv("TIMEOUT_MS", 300)

    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_SERVER_URL,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )
    logger.info("Coral Server Connection Established")

    coral_tools = await client.get_tools(server_name="coral")
    logger.info(f"Coral tools count: {len(coral_tools)}")

    agent_tools = [
        Tool(
            name="open_deepresearch",
            func=None,
            coroutine=odr_tool_async,
            description="Generates a comprehensive research report on a given topic using OpenDeepResearch. Saves the report to a temporary directory as report.txt and returns the complete research report content along with the file path.",
            args_schema={
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic for the research report"
                    }
                },
                "required": ["topic"],
                "type": "object"
            },
            response_format="content_and_artifact"
        )
    ]
    
    agent_executor = await create_agent(coral_tools, agent_tools)

    while True:
        try:
            logger.info("Starting new agent invocation")
            await agent_executor.ainvoke({"agent_scratchpad": []})
            logger.info("Completed agent invocation, restarting loop")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in agent loop: {str(e)}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())