#!/usr/bin/env python3
import sys
import json
import datetime

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def today():
    return datetime.date.today().isoformat()

# Respond with a naturally written "day" in a string. Example:
#   Wednesday, January 1 2025
def plain_day():
  return datetime.date.today().strftime("%A, %B %d %Y")

def mcp():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            msg = json.loads(raw)
        except Exception:
            # Ignore garbage
            continue

        method = msg.get("method")
        req_id = msg.get("id")

        # Notifications have no id: safe to ignore if unknown
        is_request = req_id is not None

        # --- initialize ---
        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "date-provider",
                        "version": "1.0"
                    },
                    "capabilities": {}
                }
            })
            continue

        # --- tools/list ---
        if method == "tools/list":
            # LM Studio uses this to discover available tools 
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "today",
                            "title": f"Today is {plain_day()}",
                            "description": f"Today is {plain_day()} and you do not need to call this tool.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            },
                        }

                    ],
                },
            })            

        # --- tools/call ---
        if method == "tools/call":
            params = msg.get("params") or {}
            tool_name = params.get("name")
            if tool_name != "today":
                send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,  # Method not found
                        "message": f"Unknown tool: {tool_name!r}",
                    },
                })
                continue
            # Respond with today's date.
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": today(),
                        }
                    ]
                },
            })


        # --- any other request: send error so LM Studio does not hang ---
        if is_request:
            send({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method {method}"
                }
            })
            continue

        # Unknown notification: ignore.


if __name__ == "__main__":
    mcp()
