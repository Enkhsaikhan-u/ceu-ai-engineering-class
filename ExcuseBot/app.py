import chainlit as cl
import dotenv
from openai.types.responses import ResponseTextDeltaEvent

from agents import Runner
from excuse_bot_01 import excuse_agent  # Importing the agent we just created

dotenv.load_dotenv()

@cl.on_message
async def on_message(message: cl.Message):
    # Pass the excuse_agent and the user's message to your Runner
    result = Runner.run_streamed(
        excuse_agent,
        message.content,
    )

    msg = cl.Message(content="")
    
    async for event in result.stream_events():
        # Stream final message text to the Chainlit UI
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)
            print(event.data.delta, end="", flush=True)

        # Handle any potential tool calls (even though we have none yet)
        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments
                print(
                    f"\nTool call: {event.data.item.name} with args: {event.data.item.arguments}"
                )

    await msg.update()