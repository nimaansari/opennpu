# Security

OpenNPU's server listens on `127.0.0.1:11540` only and has no authentication: anything running as
your user on the same machine can use it, as with Ollama or LM Studio. Do not expose the port to a
network (for example with a reverse proxy or `ssh -R`) unless you add authentication in front of it.

Models are downloaded from their publishers and run locally. The server loads safetensors and GGUF
weights only. The Python API (`opennpu.loaders`) can also load a pickled PyTorch `.pt` file when you
point it at one; unpickling runs code, so only do that with files you trust.

## Reporting a vulnerability

Please report security issues privately through GitHub's "Report a vulnerability" (Security tab of
this repository) rather than in a public issue. Include the OpenNPU version (`opennpu --version` or
`GET /api/version`), what you did, and what happened. You should get a reply within a week.
