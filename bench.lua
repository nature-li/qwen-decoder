-- bench.lua
-- wrk -t4 -c64 -d30s -s bench.lua http://localhost:8080/v1/chat/completions
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"messages":[{"role":"user","content":"请详细介绍 Transformer 架构的工作原理，包括 self-attention 机制、multi-head attention、position encoding、feed-forward network 等核心组件，并举例说明每个部分的作用"}],"max_tokens":256,"stream":false}'