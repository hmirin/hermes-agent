from types import SimpleNamespace

from agent.conversation_compression import compress_context
from agent.codex_responses_adapter import _chat_messages_to_responses_input


class FakeTransport:
    def preflight_kwargs(self, api_kwargs, *, allow_stream=False):
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs

        return _preflight_codex_api_kwargs(api_kwargs, allow_stream=allow_stream)


class FakeResponses:
    def __init__(self):
        self.compact_calls = []

    def compact(self, **kwargs):
        self.compact_calls.append(kwargs)
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="compaction_summary",
                    id="cmp_1",
                    encrypted_content="compact_opaque",
                    created_by="responses.compact",
                )
            ],
            usage=None,
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class DummyCodexResponsesAgent:
    def __init__(self):
        self.api_mode = "codex_responses"
        self.provider = "openai-codex"
        self.model = "gpt-5.5"
        self.base_url = "https://chatgpt.com/backend-api/codex"
        self.api_key = "stub"
        self.session_id = "hermes-session-1"
        self.platform = "cli"
        self._cached_system_prompt = "cached prompt"
        self._client = FakeClient()
        self.context_compressor = SimpleNamespace(
            protect_last_n=1,
            compression_count=0,
            last_compression_rough_tokens=0,
            last_prompt_tokens=123,
            last_completion_tokens=45,
            awaiting_real_usage_after_compression=False,
            update_from_response=lambda usage: None,
        )
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = None
        self.session_cost_source = None
        self.statuses = []
        self.warnings = []
        self.events = []

    def _build_api_kwargs(self, messages):
        return {
            "model": self.model,
            "instructions": self._cached_system_prompt,
            "input": _chat_messages_to_responses_input(messages),
            "store": False,
            "prompt_cache_key": "cache-key",
        }

    def _get_transport(self):
        return FakeTransport()

    def _ensure_primary_openai_client(self, *, reason):
        return self._client

    def _try_refresh_codex_client_credentials(self, *, force=False):
        return False

    def _emit_status(self, message):
        self.statuses.append(message)

    def _emit_warning(self, message):
        self.warnings.append(message)

    def _build_system_prompt(self, system_message):
        return "built prompt"

    def event_callback(self, name, payload):
        self.events.append((name, payload))


def test_codex_responses_compression_calls_responses_compact_and_keeps_tail():
    agent = DummyCodexResponsesAgent()
    messages = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "latest user"},
    ]

    returned, prompt = compress_context(
        agent,
        messages,
        "system",
        approx_tokens=100000,
        task_id="test",
    )

    assert prompt == "cached prompt"
    assert agent._client.responses.compact_calls == [
        {
            "model": "gpt-5.5",
            "input": [
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old assistant"},
            ],
            "instructions": "cached prompt",
            "prompt_cache_key": "cache-key",
        }
    ]
    assert returned == [
        {
            "role": "assistant",
            "content": "",
            "codex_compaction_items": [
                {
                    "type": "compaction_summary",
                    "encrypted_content": "compact_opaque",
                    "id": "cmp_1",
                    "created_by": "responses.compact",
                }
            ],
        },
        {"role": "user", "content": "latest user"},
    ]
    assert agent.context_compressor.compression_count == 1
    assert agent.context_compressor.last_compression_rough_tokens == 100000
    assert agent.context_compressor.last_prompt_tokens == -1
    assert agent.context_compressor.awaiting_real_usage_after_compression is True
    assert agent.session_api_calls == 1
    assert agent.events == [
        (
            "session:compress",
            {
                "platform": "cli",
                "session_id": "hermes-session-1",
                "old_session_id": "",
                "in_place": True,
                "compression_count": 1,
                "runtime": "codex_responses",
                "compaction_items": 1,
            },
        )
    ]
