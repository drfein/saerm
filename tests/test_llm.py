import saerm.llm as llm_module


class DummyResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.contents = None

    def generate_content(self, contents):
        self.contents = contents
        return DummyResponse("mock response")


class FakeGenAI:
    def __init__(self) -> None:
        self.configured_api_key = None
        self.last_model = None

    def configure(self, *, api_key: str) -> None:
        self.configured_api_key = api_key

    def GenerativeModel(self, model_name: str):
        self.last_model = FakeModel(model_name)
        return self.last_model


def test_generate_llm_response_with_gemini(monkeypatch):
    fake_genai = FakeGenAI()
    monkeypatch.setattr(llm_module, "genai", fake_genai)

    client = llm_module.GeminiLLMClient(api_key="secret", model_name="gemini-mock")
    output = llm_module.generate_llm_response(
        "user prompt", "assistant prompt", client=client
    )

    assert output == "mock response"
    assert fake_genai.configured_api_key == "secret"
    assert fake_genai.last_model.name == "gemini-mock"
    assert fake_genai.last_model.contents == [
        {"role": "model", "parts": ["assistant prompt"]},
        {"role": "user", "parts": ["user prompt"]},
    ]
