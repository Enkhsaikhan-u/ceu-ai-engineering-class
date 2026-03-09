import dotenv
import litellm 
from agents import (
    Agent,
    FunctionTool,
    function_tool,
)

# Load environment variables (ensure your AWS credentials are set in .env)
dotenv.load_dotenv()

# 1. The model string formatted for your 'agents' wrapper
MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"

# 2. The model string formatted for standard LiteLLM API calls (no 'litellm/' prefix)
EVAL_MODEL = "bedrock/eu.amazon.nova-lite-v1:0"

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
    """
    # FIX 1: Force the secondary LLM to format its output cleanly
    prompt = (
        f"You are a strict HR manager and professor. Rate the following excuse on a scale of 1 to 10. "
        f"Start your response EXACTLY with 'Believability: X/10' (replace X with the number). "
        f"Then, provide a 1-sentence explanation for your score.\n\nExcuse: {excuse}"
    )
    
    try:
        response = litellm.completion(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n🚨 LITELLM TOOL ERROR: {str(e)} 🚨\n") 
        return f"Assessment failed: {str(e)}"

# Initialize the agent with both tools
excuse_agent = Agent(
    name="Excuse Assistant",
    instructions="""
    You are a creative assistant who generates excuses for students and employees. 
    
    When the user asks for an excuse, you must follow this exact sequence:
    1. Generate exactly 1 excuse based on their prompt.
    2. Call the assess_believability_tool EXACTLY ONCE on that excuse.
    3. IMMEDIATELY output the final excuse and the tool's score to the user.
    
    CRITICAL GUARDRAILS:
    - DO NOT evaluate the score. 
    - DO NOT try to improve, revise, or auto-correct the excuse if the score is low. 
    - Accept the very first score you get, even if it is a 1/10, and show it to the user.
    - DO NOT generate a new excuse unless the user explicitly asks for another one.
    - ONLY use the tone_adjustment_tool if the user explicitly types the word "tone".
    
    Stop generating entirely once you have presented the first excuse and its score.
    """,
    model=MODEL,
    tools=[
        bedrock_tool(tone_adjustment_tool.__dict__),
        bedrock_tool(assess_believability_tool.__dict__)
    ],
)