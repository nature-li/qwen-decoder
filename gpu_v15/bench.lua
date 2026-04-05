-- bench.lua
-- wrk -t4 -c64 -d30s -s bench.lua http://localhost:8080/v1/chat/completions
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"messages":[{"role":"user","content":"讲一下Transformer"}],"max_tokens":256,"stream":false}'