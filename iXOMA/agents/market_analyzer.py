from crewai import Agent
from utils.logger import log_reasoning

def build_market_analyzer(llm, tools, logger):
    return Agent(
        name="Market Analyzer",
        role="Quantitative Market Observer",
        goal="Summarize current WEEX market state with TA context.",
        backstory="Veteran quant focused on fast market reads for perpetual futures.",
        tools=[tools["get_price"], tools["get_order_book"]],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,
        callbacks=[lambda *args, **kwargs: log_reasoning(logger, "Market Analyzer", str(kwargs.get('output', '')))],
    )
