import asyncio
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

agent = LlmAgent(name="TestAgent", model="gemini-2.5-flash", instruction="Just reply with 0.75")
svc = InMemorySessionService()
runner = Runner(app_name="test", agent=agent, session_service=svc, auto_create_session=True)

async def main():
    gen = runner.run_async(user_id="1", session_id="1", new_message=Content(role="user", parts=[Part.from_text(text="hi")]))
    outputs = []
    async for ev in gen:
        print("EV TYPE:", type(ev))
        print("EV DICT:", ev)
        outputs.append(ev)

asyncio.run(main())
