# Security

KPViz is a single-user tool that runs on your own machine. It binds to
`127.0.0.1` by default and has no authentication: anyone who can reach its
port can browse your documents. Passing another `--host` prints a warning;
do so only on a network you trust.

KPViz reads your files in place and writes only to its state directory
(`.kpviz/` next to the data, or `--state`). With `--offline` it never uses
the network; otherwise it downloads tokenizer assets from Hugging Face or
tiktoken's servers, once, into that state directory.

## Reporting a vulnerability

Please do not open a public issue. Write to the maintainer listed in
[CITATION.cff](CITATION.cff) with a description and, if possible, steps to
reproduce. You will get an answer within a week.
