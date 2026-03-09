import dotenv
import litellm  # We need this to make the secondary LLM call inside the tool
from agents import (
    Agent,
    FunctionTool,
    function_tool,
)

# Load environment variables (ensure your AWS credentials are set)
dotenv.load_dotenv()

# The main model for generating excuses
MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"

# You can use a different model for the judge, but we'll reuse the same one for simplicity. 
# You could change this to a faster/cheaper model like "litellm/bedrock/eu.amazon.nova-micro-v1:0"
EVAL_MODEL = MODEL 

def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )

@function_tool
def tone_adjustment_tool(message: str, tone: str) -> str:
    """
    Tool that adjusts the tone of an apology or explanation message.

    Args:
        message: The original message.
        tone: Desired tone ("formal", "casual", "professional").

    Returns:
        The message rewritten with the requested tone.
    """
    tone = tone.lower()
    if tone == "formal":
        return "Rewritten in a formal tone:\n\n" + message.replace("Hi", "Dear")
    elif tone == "casual":
        return "Rewritten in a casual tone:\n\n" + message.replace("Dear", "Hi")
    elif tone == "professional":
        return "Rewritten in a professional tone:\n\n" + message
    else:
        return "Unknown tone option. Available tones: formal, casual, professional."

@function_tool
def assess_believability_tool(excuse: str) -> str:
    """
    Tool that uses a secondary LLM to assess how believable and realistic an excuse is.

    Args:
        excuse: The text of the proposed excuse.

    Returns:
        A score out of 10 and a brief critique of the excuse's believability.
    """
    prompt = (
        f"You are a strict HR manager and professor. Rate the following excuse on a "
        f"scale of 1 to 10 for believability. Provide a brief 1-sentence explanation "
        f"for your score, pointing out any obvious flaws.\n\nExcuse: {excuse}"
    )
    
    try:
        # Call the secondary LLM to act as the judge
        response = litellm.completion(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, # Keep temperature low for more consistent grading
        )
        return f"Believability Assessment:\n{response.choices[0].message.content}"
    except Exception as e:
        return f"Assessment failed: {str(e)}"

# Initialize the agent with both tools
excuse_agent = Agent(
    name="Excuse Assistant",
    instructions="""
    You are a highly creative, persuasive, and helpful assistant who generates excuses for students and employees. 
    When a user tells you what they need to get out of, provide strictly 1 option.
    
    Important rules:
    1. If the user asks you to verify or rate an excuse, use the assess_believability_tool.
    2. If the user asks to adjust the tone of a specific message, always use the tone_adjustment_tool.
    """,
    model=MODEL,
    tools=[
        bedrock_tool(tone_adjustment_tool.__dict__),
        bedrock_tool(assess_believability_tool.__dict__)
    ],
)