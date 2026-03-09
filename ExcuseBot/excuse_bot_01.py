import dotenv
import litellm 
from agents import (
    Agent,
    FunctionTool,
    function_tool,
)

# Load environment variables
dotenv.load_dotenv()

MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"
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
def assess_humour_tool(excuse: str) -> str:
    """
    Tool that uses a secondary LLM to assess how funny or witty an excuse is.
    """
    prompt = (
        f"You are a professional stand-up comedian and roast battle judge. "
        f"Rate the following excuse for its 'Humour & Wit' on a scale of 1 to 10. "
        f"Start your response EXACTLY with 'Humour Score: X/10' (replace X with the number). "
        f"Then, provide a 1-sentence witty critique of the joke or excuse.\n\nExcuse: {excuse}"
    )
    
    try:
        response = litellm.completion(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, # Slightly higher temp for "creative" judging
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Humour assessment failed: {str(e)}"

@function_tool
def assess_believability_tool(excuse: str) -> str:
    """
    Tool that uses a secondary LLM to assess how believable and realistic an excuse is.
    """
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
        return f"Assessment failed: {str(e)}"

# Initialize the agent with Believability and Humour tools
excuse_agent = Agent(
    name="Excuse Assistant",
    instructions="""
    You are a creative assistant who generates excuses for students and employees. 
    
    When the user asks for an excuse, you must follow this exact sequence:
    1. Generate exactly 1 excuse based on their prompt.
    2. Call the assess_believability_tool on that excuse.
    3. Call the assess_humour_tool on that same excuse.
    4. IMMEDIATELY output the final excuse followed by both the Believability and Humour scores.
    
    CRITICAL GUARDRAILS:
    - DO NOT attempt to improve the excuse once you have the scores. 
    - Present the scores exactly as they are returned by the tools.
    - DO NOT generate a new excuse unless the user explicitly asks for another one.
    - Stop generating entirely once you have presented the excuse and its two scores.
    """,
    tools=[
        bedrock_tool(assess_believability_tool.__dict__),
        bedrock_tool(assess_humour_tool.__dict__) # Replaced tone tool with this
    ],
    model=MODEL,
)