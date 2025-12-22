from crewai import Task

def task_manage_risk(agent, symbol: str, config_snapshot: dict):
    prompt = f"""
You are the risk manager. Assess proposed signal for {symbol}.
Constraints:
- Max position USDT: {config_snapshot['max_position_usdt']}
- Risk per trade: {config_snapshot['risk_per_trade']}
- Dry-run: {config_snapshot['dry_run']}
Inputs:
- Market summary: use prior task output.
- Signal: action|confidence|reason from signal generator.
Decide: APPROVE/REDUCE/VETO and propose size (USDT) and stop buffer pct (e.g., 0.01).
Format: decision|size_usdt|stop_buffer|reason.
"""
    return Task(
        description=prompt,
        expected_output="decision|size_usdt|stop_buffer|reason",
        agent=agent,
        async_execution=False,
        output_key=f"risk_{symbol.replace('/', '_')}",
    )
