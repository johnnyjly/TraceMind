# Bundled admission vocabulary

`cl100k_base.tiktoken` is the immutable tiktoken vocabulary downloaded from
https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken.
SHA256: `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
The source implementation and regular expression come from
`tiktoken_ext.openai_public.cl100k_base`; its MIT notice is retained in
`LICENSE.tiktoken`.

The runtime loads and verifies this packaged asset locally. It does not fetch
tokenizers, call a model API, or depend on a warmed user cache. This tokenizer
is an admission proxy for DeepSeek and other providers, with 20% headroom and
1024 framing tokens; it is not their exact tokenizer. Provider-reported usage
remains authoritative.
