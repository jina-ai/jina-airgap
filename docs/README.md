# Documentation

Mirror of the [GitHub Wiki](https://github.com/jina-ai/jina-on-prem/wiki). Edit here via pull request, then run `./scripts/sync-wiki.sh` to push to the wiki.

## Pages

- [Home](Home.md) - overview + navigation
- [Why Air-Gap?](Why-Airgap.md) - explains the concept, compares with SaaS/VPC endpoints
- [Quick Start](Quick-Start.md) - 5-minute walkthrough using a prebuilt image
- [Customer Scenarios](Customer-Scenarios.md) - per-industry playbooks
- [Picking a Model](Picking-A-Model.md) - decision tree for the 29 models
- [Sizing & Hardware](Sizing-And-Hardware.md) - capacity planning, k8s, throughput
- [Bundling Guide](Bundling-Guide.md) - build your own .tar.gz
- [API Reference](API-Reference.md) - four schemas + reranker + ES integration
- [Architecture](Architecture.md) - how the pieces fit
- [Model Catalog](Model-Catalog.md) - auto-generated, regenerate via `python3 scripts/gen_catalog_md.py`
- [FAQ](FAQ.md) - common questions
- [Troubleshooting](Troubleshooting.md) - errors and fixes

## Wiki-only files

The wiki also has `_Sidebar.md` and `_Footer.md` which provide persistent navigation. They live only in the wiki repo (not synced from docs/) because they reference wiki-relative paths.

## Regenerating the Model Catalog

```bash
python3 scripts/gen_catalog_md.py > docs/Model-Catalog.md
./scripts/sync-wiki.sh
```
