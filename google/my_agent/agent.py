
from google.adk.agents import Agent
from google.adk.models import LiteLlm
api_base_url = "http://localhost:1234/v1"
model_name_at_endpoint = "openai/google/gemma-4-e2b"
model = LiteLlm(
    model=model_name_at_endpoint,
    api_base=api_base_url,
    api_key="lm-studio",
)
root_agent = Agent(
    model=model,
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge",
)