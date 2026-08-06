![perimeter](images/pixel-airgap-perimeter.png)

jina-on-prem runs Jina's embedding, reranking and reader models on your own infrastructure, as a Docker image with the weights baked in. It never calls out, so it works whether your host has restricted egress, no egress, or a full air gap.

Those are three different situations, and only the last one is what the industry calls an air gap. The product is built for the strictest of them, which is why it also suits the other two: an image that needs no network cannot be broken by a network policy you have not told us about.

## Where this fits

| Sector | What drives the requirement | Typical placement |
|---|---|---|
| **Banking / Finance** | Client data, trading signals and internal research cannot leave the network; the regulatory perimeter is hard. | Private VPC, on-prem ES cluster, locked-down inference |
| **Government / Defense** | Classified or controlled-unclassified data, approved-vendor lists, no SaaS. | SCIF / IL5 environments, no inbound or outbound internet |
| **Healthcare** | HIPAA, GDPR, patient records that legally cannot transit a third-party API. | Hospital data center, on-prem ES, sometimes isolated clinical research VLANs |
| **Industrial / OT** | Plant networks isolated from corporate IT, mission-critical with no tolerance for an upstream outage. | Operational tech network, edge inference next to sensors |
| **Sovereign / localized AI** | Data residency law (EU AI Act, China CSL, India DPDPA) requires the model to run in-country, often in your own data center. | National cloud, your data center, an approved region |

jina-on-prem is a good fit when:

- Procurement requires on-premises or self-managed infrastructure.
- Data cannot be sent to a third-party API.
- The deployment host has no outbound internet.
- The system must keep working if an upstream service goes down.
- Audit requires the model artifacts to sit on hardware you control.

## How it compares

```
  HOSTED API  (api.openai.com / api.cohere.com / api.jina.ai)
  ─────────────────────────────────────────────────────────────
    your app      ──►  vendor API   every request crosses your
                  ◄──                perimeter; the vendor handles
                                     logs and retention

  jina-on-prem  (inside your environment)
  ─────────────────────────────────────────────────────────────
    your app      ──►  jina-on-prem   localhost / internal DNS
                                     ╳ no outbound calls
                                     ╳ nothing visible to us
```

| | Hosted API (SaaS) | Self-managed VPC endpoint | jina-on-prem |
|---|---|---|---|
| Data leaves your network | yes | no, but vendor still operates the endpoint | no |
| Works without internet | no | partial (depends on auth/control plane) | yes |
| You hold the model weights | no | no | yes |
| Per-request cost | yes | reserved capacity | your hardware only |
| Latency | network RTT to vendor | network RTT inside VPC | localhost / LAN |
| Audit story | vendor compliance docs | vendor compliance docs | you hold the artifacts |
| Time to first request | minutes | hours | minutes once the image is on disk |

The trade is that you run the host - a GPU box, or a CPU machine for the smaller models. For most enterprises that is already true of the rest of the stack.

## What "air-gap" means in this project

A jina-on-prem container does **not** call out to:

- HuggingFace Hub (`HF_HUB_OFFLINE=1` baked in)
- Any model registry (`TRANSFORMERS_OFFLINE=1` baked in)
- A license server (there isn't one)
- Telemetry or logging endpoints (none exist)

All weights, tokenizers, processors, and Python dependencies are baked into the Docker image at bundle time. After the image is loaded onto the offline machine, the only "network" it speaks is the HTTP API on port 8080.

### Where data goes - request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant App as Your app
    participant LB as Optional LB
    participant Cont as jina-on-prem container
    participant GPU as GPU/CPU
    participant Disk as Local disk
    participant Net as Internet

    App->>LB: POST /v1/embeddings {input}
    LB->>Cont: forward
    Cont->>Disk: read weights (already cached)
    Cont->>GPU: encode tokens -> tensors
    GPU-->>Cont: embeddings
    Cont-->>LB: 200 OK {embeddings}
    LB-->>App: 200 OK
    Note over Cont,Net: No call to Internet at any step.
HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1
make download attempts fail-fast.
```

The request never crosses your perimeter. The container initiates nothing outbound on its own: weights are baked in, and the offline env vars make `huggingface_hub` and `transformers` refuse downloads at the function-call layer rather than at the network layer.

There is exactly one way to make it fetch something, and a request has to ask for it: a multimodal request may carry an `http(s)` URL for an image or a video, and the server will read it. Whether that URL resolves is your network's decision. Inline `data:` URLs and file uploads never touch the network at all.

> **How to verify it**: run the container with `docker run --network none`, which removes the network stack, and reach the server over the container's own loopback with `docker exec` - not through `-p`, which cannot forward without a container IP. That is what [`verify-offline.sh`](https://github.com/jina-ai/jina-on-prem/blob/main/verify-offline.sh) does, and it is a stronger check than the env vars alone: egress becomes impossible rather than merely refused. Steps in [Troubleshooting -> Verifying the air gap](Troubleshooting#verifying-the-air-gap).

## Next

- [Quick Start](Quick-Start) - get the first response back in 5 minutes
- [Deployment Patterns](Deployment-Patterns) - worked examples per industry
- [Sizing & Hardware](Sizing-And-Hardware) - GPU/CPU sizing
