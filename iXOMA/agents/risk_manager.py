from crewai import Agent
from utils.logger import log_reasoning

def build_risk_manager(llm, tools, logger):
    return Agent(
        name="Risk Manager",
        role="Risk Controller",
        goal="Approve, resize, or veto signals based on risk rules and exposure limits.",
        backstory="Keeps drawdowns contained; enforces sizing and safety.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        callbacks=[lambda *args, **kwargs: log_reasoning(logger, "Risk Manager", str(kwargs.get('output', '')))],
    )
