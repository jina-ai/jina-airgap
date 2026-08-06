What the container does and does not do on a network, and what you have to add around it. Written to be quotable in a security questionnaire — every claim here is checkable against the image or the source.

> **The one thing to configure before production:** the server has no authentication. Anyone who can reach port 8080 can use it. Put it behind something that authenticates, and do not expose the port beyond your own network.

## What the container does on a network

| | |
|---|---|
| Listens on | TCP 8080, plain HTTP |
| Authenticates requests | **No** — see [Authentication](#authentication) |
| Encrypts in transit | **No** — terminate TLS in front of it, see [TLS](#tls) |
| Calls out on its own initiative | **No** — no license server, no telemetry, no update check, no model download |
| Calls out when a request asks it to | Only on a multimodal model, and only for an `http(s)` media URL the request itself contains — see [The one outbound path](#the-one-outbound-path) |
| Runs as | **root**, unless you say otherwise — see [Run it as a non-root user](#run-it-as-a-non-root-user) |
| Writes outside the container | Nothing. No volume is required; the model lives in the image |

## Properties for a compliance review

- Model weights and dependencies run on your host. No data or queries leave your environment.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are baked into the image, so the model-loading paths cannot fetch anything at runtime.
- No license server, no phone-home, no telemetry, no usage reporting. The optional [license key](Licensing) is verified locally with an HMAC comparison and never contacts anything.
- The image is built once on a connected machine, transferred through your change-control process, and runs on disconnected machines indefinitely.
- The CLI and the server source are Apache-2.0 and auditable. Model weights carry their own license, per the [Model Catalog](Model-Catalog).
- The image contents are fixed at build time. Pull by digest, or by the immutable `-<sha>` tag rather than `:cpu` / `:gpu`, if your process requires the bytes to be pinned.

For the request lifecycle and a step-by-step of why nothing leaves, see [Why On-Prem → What "air-gap" means in this project](Why-On-Prem#what-air-gap-means-in-this-project). To prove it on your own hardware, see [Verifying the air gap](Troubleshooting#verifying-the-air-gap).

## Authentication

There is none. The server does not read an `Authorization` header, has no API-key concept, and does not distinguish callers. This is deliberate — it is designed to sit inside a perimeter you already control, and adding a second credential system would be one more thing to rotate in an environment where rotation is expensive.

It means the port is the access control, so:

- **Never publish 8080 to an untrusted network.** Bind it to localhost (`-p 127.0.0.1:8080:8080`) when the caller is on the same host, or to an internal interface only.
- **Put a reverse proxy in front** if more than one team or application calls it: nginx, Envoy, HAProxy or your existing API gateway, doing authentication and TLS in one place.
- **Use network policy as the real boundary.** In Kubernetes, a `NetworkPolicy` restricting ingress to the calling namespace is stronger than anything the container could check itself.

`GET /health` is unauthenticated by design and returns no request data — model id, device, readiness, the schema list, and license status. Treat it as safe to expose to your monitoring, and note that it does reveal which model you run.

## TLS

The server speaks plain HTTP. Terminate TLS in front of it — the same reverse proxy that authenticates is the right place — and keep the hop from proxy to container on a network you control, or on localhost.

There is no configuration to make the container itself serve HTTPS, and adding one would mean shipping certificate handling and rotation into an image that is meant to be replaced rather than reconfigured.

## The one outbound path

A **multimodal** model accepts an image, audio or video input, and that input may be given as an `http(s)` URL. When a request contains one, the server reads it. That is the only code path in the container that opens an outbound connection, and a request has to ask for it.

What constrains it:

- **It does not exist on a text-only model.** Every media-accepting endpoint calls `require_multimodal()` first, so on `jina-embeddings-v5-text-*`, `v3`, the `v2` family, the rerankers and ColBERT there is no reachable fetch at all.
- **Only `http` and `https`.** A `file:`, `gs:`, `s3:` or any other scheme is rejected with a 400, and a redirect to one of them is rejected too — the redirect handler re-checks the scheme rather than trusting the first URL.
- **10 MB per input, with a fetch timeout.** A large or slow URL fails the request instead of holding the worker.
- **Your network decides whether it resolves.** With `--network none`, or with an egress policy, the fetch simply fails and the request returns an error.

If you want the capability without the fetch, send media as an inline `data:` URL or an upload. Those never touch the network.

## Custom model code, and why `trust_remote_code` is forced on

Jina's models ship their own modeling code alongside the weights, which `transformers` loads only when `trust_remote_code` is enabled. The server enables it unconditionally — it replaces `transformers.dynamic_module_utils.resolve_trust_remote_code` with a function that always returns `True`.

Stated that way it sounds like the container will execute whatever a model repository serves it. It will not, and the reason is the same offline guarantee as everything else:

- **The code is baked in at bundle time**, from the model revision pinned in `models/catalog.json`, on the connected machine that built the image. It is a normal part of the image layers.
- **At runtime the container cannot fetch code.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set, so there is no path by which a new or changed module could arrive. The prompt that `trust_remote_code` normally guards has nothing left to guard.
- **The bundle deletes each model repository's own `requirements.txt`** after download, because some `trust_remote_code` paths will otherwise `pip install` from it at load time. Nothing installs at runtime.

So the audit question is not "will it run remote code" but "what code is in this image", and that is answerable: the module files are in the image, and the revision they came from is in the catalog. Pin the image by digest if you need the answer to stay fixed.

## Run it as a non-root user

The image does not set a `USER`, so the process runs as root inside the container. Nothing in the server needs root — it reads the model from the image and listens on 8080.

```bash
docker run --user 65534:65534 --read-only \
  --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  -p 127.0.0.1:8080:8080 jina/MODEL:cpu
```

`--read-only` works because the container writes nothing persistent; `/tmp` is given as a tmpfs for Python's own scratch use. In Kubernetes the equivalent is a `securityContext` with `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false` and dropped capabilities.

GPU deployments need the NVIDIA runtime's device access, which is granted by `--gpus`, not by root.

## A hardening checklist

| | Why |
|---|---|
| Bind 8080 to localhost or an internal interface | There is no authentication in the server |
| Terminate TLS at a proxy in front | The server speaks plain HTTP |
| Authenticate at that proxy or gateway | Same reason |
| Restrict egress from the container | Makes the media-URL path unusable, and proves the air gap |
| Run with `--user`, `--read-only`, `--cap-drop ALL` | Nothing needs root or a writable filesystem |
| Pin the image by digest or `-<sha>` tag | `:cpu` and `:gpu` move when a new build is published |
| Choose a text-only model where multimodal is not needed | Removes the only outbound code path |
| Scan the image on arrival | It is a normal OCI image; your existing scanner applies |

## Next

- [Why On-Prem](Why-On-Prem) - the request lifecycle and what "air-gap" means precisely
- [Licensing](Licensing) - the offline key, and what it does not do
- [Sizing & Hardware](Sizing-And-Hardware) - deployment topologies and redundancy
- [Support](Support) - reporting a problem without putting data in a public tracker
