from crewai import Task

def task_execute_trade(agent, symbol: str):
    prompt = f"""
Execute the approved plan for {symbol}.
- If decision is VETO or FLAT, do nothing.
- If APPROVE/REDUCE with LONG/SHORT, place a market order (or simulate in dry-run).
- Show final action, size, stop buffer, and order response.
Return concise execution result string.
"""
    return Task(
        description=prompt,
        expected_output="execution_result string",
        agent=agent,
        async_execution=False,
        output_key=f"execution_{symbol.replace('/', '_')}",
    )
