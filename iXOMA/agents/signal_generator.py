from crewai import Agent
from utils.logger import log_reasoning

def build_signal_generator(llm, tools, logger):
    return Agent(
        name="Signal Generator",
        role="Directional Strategist",
        goal="Propose long/short/flat signals with confidence using TA summary and constraints.",
        backstory="Combines TA with reasoning to craft concise trade calls.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        callbacks=[lambda *args, **kwargs: log_reasoning(logger, "Signal Generator", str(kwargs.get('output', '')))],
    )
