from crewai import Task

def task_generate_signal(agent, symbol: str):
    prompt = f"""
Generate a trading signal for {symbol} based on market summaries from analyzer.
- Consider EMA crossover, momentum, liquidity hints.
- Output one of: LONG, SHORT, FLAT.
- Provide confidence 0-1 and short justification.
Format: action|confidence|reason.
"""
    return Task(
        description=prompt,
        expected_output="Structured action|confidence|reason",
        agent=agent,
        async_execution=False,
        output_key=f"signal_{symbol.replace('/', '_')}",
    )
