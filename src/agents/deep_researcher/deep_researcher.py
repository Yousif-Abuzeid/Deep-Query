"""Deep Research Agent with LangGraph implementation."""

import logging
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from stores.llm.templates.locales.en.deep_researcher import (
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
)

from .state import (
    AgentState,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    SupervisorState,
)
from .tools import get_browser_toolkit, tavily_search, think_tool

# Set up logger
logger = logging.getLogger("uvicorn")


class DeepResearch:
    """Deep Research Agent using LangGraph with React agents for tool-calling."""

    def __init__(self, generation_client, max_concurrent_research_units: int = 3):
        self.generation_client = generation_client
        self.llm = self.generation_client.get_langchain_chat_model()
        self.max_concurrent_research_units = max_concurrent_research_units
        self.max_researcher_iterations = 3

        # Initialize browser tools as None - will be created on first use
        self._browser_tools = None

        # Initialize search tool immediately (no async issues)
        self.search_tool = tavily_search

        # Build the graph
        self.graph = self._build_graph()

    async def _get_browser_tools(self):
        """Lazy initialization of browser tools."""
        if self._browser_tools is None:
            self._browser_tools = await get_browser_toolkit()
        return self._browser_tools

    def get_today_str(self) -> str:
        """Get today's date as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d")

    async def _build_supervisor_agent(self):
        """Build the supervisor/lead researcher agent."""
        # Supervisor only needs the think tool for strategic planning
        supervisor_agent = create_react_agent(
            model=self.llm,
            tools=[think_tool],
        )
        return supervisor_agent

    async def _build_researcher_agent(self):
        """Build an individual researcher agent with all tools."""
        # Get browser tools lazily
        browser_tools = await self._get_browser_tools()

        # Combine all research tools
        research_tools = [
            self.search_tool,
            think_tool,
        ] + browser_tools

        researcher_agent = create_react_agent(
            model=self.llm,
            tools=research_tools,
        )
        return researcher_agent

    async def supervisor(self, state: SupervisorState) -> Dict:
        """Supervisor node that plans research and coordinates researchers."""
        supervisor_agent = await self._build_supervisor_agent()

        # Format system prompt with parameters
        system_prompt = lead_researcher_prompt.format(
            date=self.get_today_str(),
            max_concurrent_research_units=self.max_concurrent_research_units,
            max_researcher_iterations=self.max_researcher_iterations,
        )

        # Prepare messages for the supervisor - CRITICAL: System prompt must be first
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research Question: {state['research_question']}"),
        ]

        # Add previous research if available
        if state.get("all_research_outcomes"):
            research_summary = "\n\n".join(
                [
                    f"Research on '{r.research_topic}':\n{r.research_outcome}"
                    for r in state["all_research_outcomes"]
                ]
            )
            messages.append(
                HumanMessage(content=f"Previous Research:\n{research_summary}")
            )

        # Invoke supervisor
        result = await supervisor_agent.ainvoke({"messages": messages})
        last_message = result["messages"][-1].content

        # 🔍 DEBUG: Log supervisor's response
        logger.info("=" * 80)
        logger.info("SUPERVISOR RESPONSE:")
        logger.info(last_message)
        logger.info("=" * 80)

        # Parse the supervisor's decision
        tasks = self._extract_research_tasks(last_message)

        # 🔍 DEBUG: Log extracted tasks
        logger.info(f"EXTRACTED TASKS: {tasks}")
        logger.info(f"Number of tasks: {len(tasks)}")

        iteration = state.get("iteration", 0) + 1

        # Check if supervisor wants to complete research
        should_complete = "RESEARCH COMPLETE" in last_message.upper()

        # 🔍 DEBUG: Log decision
        logger.info(f"Should complete: {should_complete}")
        logger.info(f"Iteration: {iteration}/{self.max_researcher_iterations}")
        logger.info(f"Has tasks: {len(tasks) > 0}")
        logger.info(
            f"Has research outcomes: {len(state.get('all_research_outcomes', []))}"
        )

        # FIX: Stop if max iterations reached OR if supervisor says complete AND we have done research
        if iteration >= self.max_researcher_iterations:
            logger.info("✓ Max iterations reached - Generating final report")
            return {
                "next_action": ResearchComplete(),
                "iteration": iteration,
            }
        elif should_complete and state.get("all_research_outcomes"):
            logger.info("✓ Supervisor says research complete - Generating final report")
            return {
                "next_action": ResearchComplete(),
                "iteration": iteration,
            }
        elif len(tasks) > 0 and not should_complete:
            logger.info(f"✓ Conducting research on {len(tasks)} tasks")
            return {
                "next_action": ConductResearch(research_topics=tasks),
                "iteration": iteration,
            }
        else:
            # Fallback: Force at least one research iteration if no research done yet
            if not state.get("all_research_outcomes"):
                logger.warning(
                    "⚠ No valid tasks and no research done yet. Using research question as task."
                )
                return {
                    "next_action": ConductResearch(
                        research_topics=[state["research_question"]]
                    ),
                    "iteration": iteration,
                }
            else:
                # We have research but supervisor didn't explicitly say complete
                logger.info(
                    "✓ Have research but ambiguous signal - Generating final report"
                )
                return {
                    "next_action": ResearchComplete(),
                    "iteration": iteration,
                }

    def _extract_research_tasks(self, content: str) -> List[str]:
        """Extract research tasks from supervisor's response."""
        tasks = []
        lines = content.split("\n")

        in_tasks_section = False
        for line in lines:
            line = line.strip()

            # Look for the RESEARCH_TASKS section
            if "RESEARCH_TASKS:" in line.upper() or "RESEARCH TASKS:" in line.upper():
                in_tasks_section = True
                # If the task is on the same line, extract it
                if ":" in line:
                    task_part = line.split(":", 1)[1].strip()
                    if task_part and len(task_part) > 10:
                        tasks.append(task_part)
                continue

            if in_tasks_section:
                # Stop if we hit another section
                if (
                    "RESEARCH COMPLETE" in line.upper()
                    or line.startswith("When complete")
                    or line.startswith("**")
                ):
                    break

                # Extract numbered or bulleted tasks
                if line and (
                    line[0].isdigit() or line.startswith("-") or line.startswith("*")
                ):
                    # Remove numbering, bullets, and clean up
                    task = line.lstrip("0123456789.-*)• ").strip()
                    # Remove any remaining formatting
                    task = task.replace("**", "").replace("__", "")
                    if task and len(task) > 10:  # Avoid very short tasks
                        tasks.append(task)
                # Also catch tasks without bullets/numbers
                elif line and len(line) > 10 and not line.endswith(":"):
                    tasks.append(line)

        # If no tasks found, try to find tool calls or actions
        if not tasks:
            import re

            # Look for tool calls in the agent's actions
            tool_match = re.search(r'"query":\s*"([^"]+)"', content)
            if tool_match:
                tasks = [tool_match.group(1)]
            else:
                # Look for any sentence that looks like a research task
                sentences = [
                    s.strip() for s in content.split(".") if len(s.strip()) > 20
                ]
                if sentences:
                    # Filter for sentences that look like research tasks
                    research_like = [
                        s
                        for s in sentences
                        if any(
                            kw in s.lower()
                            for kw in [
                                "research",
                                "investigate",
                                "analyze",
                                "examine",
                                "study",
                                "explore",
                            ]
                        )
                    ]
                    if research_like:
                        tasks = research_like[: self.max_concurrent_research_units]

        logger.info(f"_extract_research_tasks found: {tasks}")
        return tasks[: self.max_concurrent_research_units]

    async def _conduct_research_tasks(
        self, tasks: List[str]
    ) -> List[ResearcherOutputState]:
        """Conduct multiple research tasks concurrently."""
        import asyncio

        logger.info(f"Starting research on {len(tasks)} tasks...")
        research_results = await asyncio.gather(
            *[self._conduct_single_research(task) for task in tasks]
        )
        logger.info(f"Completed research on {len(tasks)} tasks")
        return research_results

    async def _conduct_single_research(
        self, research_topic: str
    ) -> ResearcherOutputState:
        """Conduct a single research task using a researcher agent."""
        logger.info(f"🔬 Researching: {research_topic[:100]}...")

        researcher_agent = await self._build_researcher_agent()

        # Format system prompt
        system_prompt = research_system_prompt.format(
            date=self.get_today_str(),
        )

        # CRITICAL: System prompt must be first, then the specific research task
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research this specific topic: {research_topic}"),
        ]

        result = await researcher_agent.ainvoke({"messages": messages})

        # Extract and compress the research
        research_messages = result["messages"]
        compressed_research = await self._compress_research(
            research_messages, research_topic
        )

        logger.info(f"✓ Completed research on: {research_topic[:50]}...")

        return ResearcherOutputState(
            research_topic=research_topic,
            research_outcome=compressed_research,
        )

    async def _compress_research(self, messages: List, research_topic: str) -> str:
        """Compress research findings into organized format."""
        # Extract the actual research content
        research_content = "\n".join(
            [msg.content for msg in messages if hasattr(msg, "content") and msg.content]
        )

        compression_messages = [
            SystemMessage(content=compress_research_system_prompt),
            HumanMessage(
                content=compress_research_simple_human_message.format(
                    research_topic=research_topic,
                    research_content=research_content,
                )
            ),
        ]

        response = await self.llm.ainvoke(compression_messages)
        return response.content

    async def generate_final_report(self, state: AgentState) -> Dict:
        """Generate the final research report."""
        logger.info("📝 Generating final report...")

        # Combine all research outcomes
        all_research = "\n\n".join(
            [
                f"## {r.research_topic}\n{r.research_outcome}"
                for r in state["all_research_outcomes"]
            ]
        )

        messages = [
            SystemMessage(content="You are an expert report writer."),
            HumanMessage(
                content=final_report_generation_prompt.format(
                    research_question=state["research_question"],
                    date=self.get_today_str(),
                    all_research=all_research,
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)

        logger.info("✓ Final report generated")

        return {
            "final_report": response.content,
        }

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", self.supervisor)
        workflow.add_node("conduct_research", self._conduct_research_tasks_node)
        workflow.add_node("generate_report", self.generate_final_report)

        # Add edges
        workflow.add_edge(START, "supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            lambda state: (
                "conduct_research"
                if isinstance(state["next_action"], ConductResearch)
                else "generate_report"
            ),
            {
                "conduct_research": "conduct_research",
                "generate_report": "generate_report",
            },
        )

        workflow.add_edge("conduct_research", "supervisor")
        workflow.add_edge("generate_report", END)

        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _conduct_research_tasks_node(self, state: AgentState) -> Dict:
        """Node wrapper for conducting research tasks."""
        if isinstance(state["next_action"], ConductResearch):
            tasks = state["next_action"].research_topics
            results = await self._conduct_research_tasks(tasks)

            # Append to existing research
            all_outcomes = state.get("all_research_outcomes", []) + results

            return {
                "all_research_outcomes": all_outcomes,
            }
        return {}

    async def conduct_research(
        self, query: str, project_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Conduct deep research on a query.

        Args:
            query: The research question
            project_id: Project identifier for tracking

        Returns:
            Dictionary with final_report and metadata
        """
        logger.info(f"🚀 Starting deep research on: {query}")

        initial_state = {
            "research_question": query,
            "all_research_outcomes": [],
            "iteration": 0,
            "next_action": None,
            "final_report": None,
        }

        config = {"configurable": {"thread_id": project_id}}

        result = await self.graph.ainvoke(initial_state, config)

        logger.info(
            f"✓ Research complete. Iterations: {result.get('iteration', 0)}, Outcomes: {len(result.get('all_research_outcomes', []))}"
        )

        return {
            "final_report": result.get("final_report"),
            "research_question": query,
            "iterations": result.get("iteration", 0),
            "research_outcomes": [
                {
                    "topic": r.research_topic,
                    "outcome": r.research_outcome,
                }
                for r in result.get("all_research_outcomes", [])
            ],
        }
