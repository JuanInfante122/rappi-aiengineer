"""SQL agent factory for Rappi Operations Intelligence.

Creates a LangChain SQL agent wired to GPT-4o with deterministic settings and
safe execution limits. The agent uses AgentType.OPENAI_FUNCTIONS (structured
JSON tool calls) rather than ReAct to avoid text-parse failures on multi-step
queries that involve schema lookup, SQL generation, and result formatting.
"""

from pathlib import Path

from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from agent.prompts import SYSTEM_PREFIX

DB_PATH = Path("data/rappi_ops.db")


def create_agent(db_path: Path = DB_PATH):
    """Create and return the LangChain SQL agent for Rappi Operations Intelligence.

    Uses AgentType.OPENAI_FUNCTIONS for structured JSON tool calls instead of
    ReAct because the SQL agent's multi-step loop (schema lookup -> query ->
    result -> answer) produces unparseable intermediate text with ReAct.

    handle_parsing_errors is passed via agent_executor_kwargs because
    create_sql_agent does not accept it as a direct parameter — it flows
    through to the underlying AgentExecutor constructor.

    sample_rows_in_table_info=0 prevents the agent from seeing actual data
    values in the schema description, reducing token cost and avoiding
    accidental data leakage in system messages.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Configured AgentExecutor instance ready to call via .invoke().
    """
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        include_tables=["raw_input_metrics", "raw_orders"],
        sample_rows_in_table_info=0,
        max_string_length=2000,
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=SYSTEM_PREFIX,
        max_iterations=10,
        max_execution_time=30,
        verbose=True,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "return_intermediate_steps": True,
        },
    )
