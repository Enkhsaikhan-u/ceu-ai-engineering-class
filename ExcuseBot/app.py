import chainlit as cl
import dotenv

from openai.types.responses import ResponseTextDeltaEvent

# Import SQLiteSession along with Runner
from agents import Runner, SQLiteSession
# Import your excuse agent instead of the nutrition agent
from excuse_bot_01 import excuse_agent

dotenv.load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    # Initialize the SQLite session when the chat starts
    session = SQLiteSession("conversation_history")
    # Store it in the Chainlit user session
    cl.user_session.set("agent_session", session)


@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the active session
    session = cl.user_session.get("agent_session")

    # Pass the session into the Runner along with the excuse_agent
    result = Runner.run_streamed(
        excuse_agent, 
        message.content, 
        session=session
    )

    msg = cl.Message(content="")
    
    async for event in result.stream_events():
        # Stream final message text to screen
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)

        # Handle tool call visualizations in the UI
        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments

    await msg.update()