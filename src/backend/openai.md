# backend/openai.rs: chat/completions with logprobs

For an OpenAI-compatible server (vLLM, SGLang, a hosted API): one
chat/completions request per fingerprint with `max_tokens: 1`,
`logprobs: true`, `top_logprobs: 20`, the prompt flattened into plain text
since the server applies its own chat template. The answer then depends on
the model emitting the label as its first token, so it is weaker than
reading raw logits; kept from upstream for comparing against models that
only exist behind such an API.
