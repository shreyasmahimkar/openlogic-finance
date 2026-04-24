import asyncio
from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname("research_papers_to_agents/moe_coordinator/"), ".env"))
from research_papers_to_agents.moe_coordinator.experts import moe_parallel_swarm

async def main():
    try:
        gen = moe_parallel_swarm.run_async(
            "Analyze market.",
            variables={"enriched_market_data": "dummy data", "filtered_news_context": "dummy news"}
        )
        async for ev in gen:
            pass
        print("SUCCESS")
    except Exception as e:
        print("ERROR:", e)

asyncio.run(main())
