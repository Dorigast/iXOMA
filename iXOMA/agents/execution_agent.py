from crewai import Agent
from utils.logger import log_reasoning

def build_execution_agent(llm, tools, logger):
    return Agent(
        name="Execution Agent",
        role="Trader",
        goal="Place (or simulate) best-effort orders respecting slippage and risk constraints.",
        backstory="Executes quickly and safely with awareness of dry-run mode.",
        tools=[tools["place_order"]],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        callbacks=[lambda *args, **kwargs: log_reasoning(logger, "Execution Agent", str(kwargs.get('output', '')))],
    )
