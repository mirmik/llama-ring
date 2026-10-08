# Third-party notices

The project code is MIT licensed. Third-party material retains its own notices.

- **llama.cpp**, The ggml authors, MIT. The revision and upstream repository are
  recorded in [deps.json](deps.json). Patches contain upstream code context;
  its [MIT notice](licenses/llama.cpp-MIT.txt) is included here. Bootstrap retains
  upstream license files, including notices for vendored components. Source
  exports do not bundle the downloaded upstream tree. Redistributors of binaries
  should preserve the notices of the components they include.
- **MemoryAgentBench**, Yuanzhe Hu, MIT. `scripts/memory_bench.py` adapts output
  parsing/scoring from upstream revision
  `538026089d1a8a8eff05121d0db89b388f360eba`; the
  [MIT notice](licenses/MemoryAgentBench-MIT.txt) is included. Benchmark preparation
  copies source provenance and the upstream license into the generated fixture.
  Dataset and model files are not included in this source release and retain
  their own licenses.

Python packages and optional GPU toolchains are installed separately and retain
their respective licenses.
