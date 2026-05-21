"""
Thin compatibility shim: src.agent.tool_call_parser mirrors agent.tool_call_parser.

Do NOT import the root ``agent`` package package — ``agent.__init__`` pulls in the
entire agent surface (heavy imports, optional third-party deps).  Re-export only
the stable public contract from the shared module.
"""

from agent.tool_call_parser import (  # noqa: F401
    JSONToolCallParser,
    ParsedToolCall,
    ParseResult,
    ToolCallParseError,
    UnifiedToolCallParser,
    XMLToolCallParser,
    detect_format,
    format_json,
    format_xml,
    parse_json,
    parse_xml,
)
