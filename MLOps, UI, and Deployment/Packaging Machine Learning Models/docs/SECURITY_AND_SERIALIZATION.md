# Security and Serialization Notes

## Unsafe Deserialization Risk

Pickle and Joblib are Python object deserializers and can execute arbitrary code if artifacts are untrusted.

## Enforced Controls

- API/CLI load with `require_manifest=True`
- Integrity verification enabled by default (`ML_VERIFY_ARTIFACTS=true`)
- Unsafe deserialization blocked by default (`ML_ALLOW_UNSAFE_DESERIALIZATION=false`)
- Unsafe load allowed only when artifact digest is trusted

## Trust Sources

Trusted digests are resolved from:

- `ML_TRUSTED_DIGESTS` environment variable
- active/matching registry metadata (`models/registry.json`)

## Format Tradeoffs

- Pickle/Joblib: highest Python compatibility, highest trust requirements
- ONNX: portable runtime, safer for non-Python serving stacks
- TorchScript: PyTorch deployment path, less relevant for sklearn-first systems

## Operational Best Practices

- never load artifacts from untrusted sources
- keep registry and manifests under change control
- rotate and re-verify artifacts on deployment
- add signed artifacts for high-trust environments
