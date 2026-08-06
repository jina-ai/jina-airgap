How to get a problem looked at, and how to do it without putting anything sensitive in a public place.

## Pick the right channel

| Your situation | Where to go |
|---|---|
| An error you can describe without your own data | [Open a GitHub issue](https://github.com/jina-ai/jina-on-prem/issues/new) |
| Anything involving your logs, queries, documents, hostnames or topology | Your Elastic support channel, not GitHub |
| A commercial license for a CC-BY-NC-4.0 model, or renewal | [Elastic sales](https://www.elastic.co/contact) |
| A question about how long a model is supported | [Product & Model Lifecycle](Product-And-Model-Lifecycle) |

The GitHub tracker is public and permanent. A container log line can contain a request, a document, an internal hostname or a model id you would rather not publish, and an issue cannot be un-published. If there is any doubt, use your support channel instead — support for a licensed deployment tracks your Elastic support entitlement and is delivered through your approved channel, which never requires the deployment to reach the internet.

Check [Troubleshooting](Troubleshooting) first. It covers the errors that account for most reports, including several whose fix is not obvious from the message.

## What to include

The more of this you can share, the less back-and-forth:

```bash
# Which image, exactly
docker inspect --format '{{index .RepoDigests 0}}' <image>

# What the server thinks it is
curl -s http://localhost:8080/health

# The host
docker version --format '{{.Server.Version}}'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv   # GPU only
```

Plus:

- **What you called and what came back** — the endpoint, the HTTP status, and the shape of the request. Not the contents, unless they are safe to share.
- **Whether it is reproducible**, and whether it happens on a smaller input.
- **What changed** — a new image, a new host, a new model, a different request shape.
- **Container logs** if they are safe to share. `docker logs <container>` is the standard ask, so see below for what is in them.

## What is in the logs

Worth knowing before you paste them anywhere:

- The server's own log line per request is **counts and timings only** — how many inputs or documents, how many tokens, elapsed milliseconds, throughput. No request content.
- The access log records method, path and status, so it shows which endpoint was called but not what was sent.
- Startup logs include the model id, device, dtype, thread count and the license key state.
- A traceback from an unexpected error is the exception: it can include values from the failing call. That is the part to read before sharing.

A safe minimum for a public issue is usually the last few lines around the error plus the `/health` output, with anything identifying replaced.

## Reporting something security-relevant

Do not open a public issue for a suspected vulnerability. Use your Elastic support channel, or [Elastic's security contact](https://www.elastic.co/community/security), and include the image digest and the smallest reproduction you have.

For questions about what the container does on a network — rather than a suspected flaw — [Security & Hardening](Security-And-Hardening) is written to be quoted directly into a questionnaire.

## Next

- [Troubleshooting](Troubleshooting) - the errors that come up most, with fixes
- [Security & Hardening](Security-And-Hardening) - network behaviour, authentication, and the hardening checklist
- [FAQ](FAQ) - licensing, support horizon, and other non-technical questions
