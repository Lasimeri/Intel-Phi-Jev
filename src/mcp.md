# mcp.rs: the MCP server

`xks mcp` speaks MCP over stdio with one tool, `judge`: a state and typed
questions in, the same typed answers as `/v1/systemone` out. From upstream,
renamed. For Claude Code:

    claude mcp add xks -- /path/to/target/release/xks mcp

Defaults (subject, site) come from `xks.conf`, so the command needs no
flags. One MCP server holds the subject and, on the `cards` site, the cards
for as long as the client keeps it running.
