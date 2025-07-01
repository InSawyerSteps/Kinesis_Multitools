import asyncio
from fastmcp import Client

async def main():
    # Connect to the MCP SSE server running on port 8100
    client = Client("http://127.0.0.1:8100/sse/")
    async with client:
        # Prepare the analyze tool request for radon complexity
        tool_name = "analyze"
        tool_args = {
            "request": {
                "path": "c:\\Projects\\MCP Server\\src\\toolz.py",
                "analyses": ["complexity"]
            }
        }

        print(f"Calling tool '{tool_name}' with args: {tool_args}")
        result = await client.call_tool(tool_name, tool_args)
        print("Result:")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
