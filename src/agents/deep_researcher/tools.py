from langchain_core.tools import tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
)
from tavily import TavilyClient
# Corrected, async-native utility functions
from playwright.async_api import async_playwright, Browser, Playwright
from helpers.config import get_settings  # ← Add this import
settings = get_settings()  # ← Load settings to access TAVILY_API_KEY
async def create_async_playwright_browser_fixed( headless: bool = True) -> Browser:
    """
    A corrected async-native function to create a Playwright browser.
    This function is a proper coroutine and MUST be awaited.
    It assumes the Playwright process is already started and passed in.
    """
    playwright_process: Playwright = await async_playwright().start()
    browser = await playwright_process.chromium.launch(headless=headless)
    return browser

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

async def get_browser_toolkit():
    """Initialize and return the Playwright browser toolkit for web scraping.
    
    This function is now async and should be called only when needed,
    not during initialization to avoid event loop conflicts.
    """
    browser = await create_async_playwright_browser_fixed()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools()


search_client = TavilyClient(
    api_key=settings.TAVILY_API_KEY

)
@tool(
    description="Search the web for relevant information using Tavily. ")
def tavily_search(query: str) -> str:
    """Use Tavily to search the web and return relevant information.

    Args:
        query: The search query string.
    """
    response = search_client.search(query)
    return response