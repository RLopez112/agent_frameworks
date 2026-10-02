import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()


class AnthropicModel:
    """Configures the Anthropic SDK client for a local or remote model endpoint."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:1234",
        api_key: str = "lm-studio",
        **client_kwargs: Any,
    ):
        self.model_name = model
        self.api_base = api_base
        self.api_key = api_key
        # LM Studio serves the Anthropic Messages API at /v1/messages.
        # The Anthropic SDK automatically appends /v1/messages to base_url,
        # so normalize by stripping any trailing /v1.
        clean_base = api_base.rstrip("/").removesuffix("/v1")
        self.client = Anthropic(
            base_url=clean_base,
            api_key=api_key or "lm-studio",
            **client_kwargs,
        )


class Agent:
    """A lightweight conversational agent powered by the Anthropic SDK."""

    def __init__(
        self,
        model: AnthropicModel | str,
        name: str = "root_agent",
        description: str = "",
        instruction: str = "",
        client: Optional[Anthropic] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.description = description
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self.messages: List[Dict[str, Any]] = []

        if isinstance(model, AnthropicModel):
            self.model_name = model.model_name
            self.client = model.client
        elif isinstance(model, str):
            self.model_name = model
            self.client = client or Anthropic(
                base_url="http://localhost:1234",
                api_key="lm-studio",
            )
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

    def run(self, prompt: str) -> str:
        """Run a single prompt and return the assistant response."""
        messages = [{"role": "user", "content": prompt}]
        params: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if self.instruction:
            params["system"] = self.instruction
        if self.tools:
            params["tools"] = self.tools
        if self.temperature is not None:
            params["extra_body"] = {"temperature": self.temperature}

        response = self.client.messages.create(**params)
        return "".join(
            block.text for block in response.content if hasattr(block, "text") and block.text
        )

    def chat(self, user_input: str) -> str:
        """Send a message in a multi-turn conversation and preserve context."""
        self.messages.append({"role": "user", "content": user_input})
        params: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }
        if self.instruction:
            params["system"] = self.instruction
        if self.tools:
            params["tools"] = self.tools
        if self.temperature is not None:
            params["extra_body"] = {"temperature": self.temperature}

        response = self.client.messages.create(**params)
        reply = "".join(
            block.text for block in response.content if hasattr(block, "text") and block.text
        )
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


# Configuration matching the reference scope (google/my_agent/agent.py)
api_base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234")
model_name_at_endpoint = os.environ.get("LM_STUDIO_MODEL", "google/gemma-4-e2b")

model = AnthropicModel(
    model=model_name_at_endpoint,
    api_base=api_base_url,
    api_key=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
)

root_agent = Agent(
    model=model,
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge",
)


def main():
    print(f"Agent '{root_agent.name}' initialized.")
    print(f"Target model: {root_agent.model_name}")
    print(f"Endpoint: {model.client.base_url}")
    print("\nStarting turn-based chat (type 'goodby' or 'goodbye' to exit)...")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            # Check if user says goodbye
            normalized = user_input.lower().strip(" .,!?:;\"'")
            if (
                normalized in {"goodby", "goodbye", "bye", "exit", "quit"}
                or normalized.startswith(("goodby", "goodbye", "bye"))
            ):
                print("\nAgent: Goodbye! Have a great day.")
                break

            response = root_agent.chat(user_input)
            print(f"\nAgent: {response}")
        except (KeyboardInterrupt, EOFError):
            print("\n\nAgent: Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")
            print("Please ensure LM Studio local server is running on port 1234.")
            break


if __name__ == "__main__":
    main()
