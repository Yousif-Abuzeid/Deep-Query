"""Graph state definitions and data structures for the Deep Research agent."""

from typing import Any, List

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

###################
# Structured Outputs for Tool Calling
###################


class ConductResearch(BaseModel):
    """Tool schema for delegating research to sub-researchers."""

    research_topics: List[str] = Field(
        description="List of specific topics to research. Each should be detailed and focused on a single aspect."
    )


class ResearchComplete(BaseModel):
    """Tool schema to signal that research is complete."""

    pass  # No fields needed, just signals completion


###################
# State Definitions
###################


class AgentState(TypedDict):
    """Main agent state for the complete research workflow."""

    research_question: str  # The research question/topic
    all_research_outcomes: List[Any]  # List of ResearcherOutputState objects
    iteration: int  # Number of research cycles completed
    next_action: Any  # ConductResearch or ResearchComplete
    final_report: str  # The final comprehensive report


class SupervisorState(TypedDict):
    """State for the supervisor managing research tasks."""

    research_question: str  # The research question/topic
    all_research_outcomes: List[Any]  # List of ResearcherOutputState objects
    iteration: int  # Number of research cycles completed
    next_action: Any  # ConductResearch or ResearchComplete


class ResearcherOutputState(BaseModel):
    """Output from a researcher agent after completing research."""

    research_topic: str = Field(description="The research topic that was investigated")
    research_outcome: str = Field(
        description="Compressed and synthesized research findings"
    )
