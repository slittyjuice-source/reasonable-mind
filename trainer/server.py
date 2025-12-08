import http.server
import socketserver
import json
import sys
import os

# Add project root to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.logic.reasoning_agent import ReasoningAgent, LogicType

PORT = 8000

# Initialize Agent
agent = ReasoningAgent(
    name="TrainerAI",
    system_prompt="You are a logic tutor helping a student learn critical thinking.",
    logic_framework=LogicType.PROPOSITIONAL,
    verbose=True,
)


class ReasoningRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/reason":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                query = data.get("query")
                context = data.get("context", "")

                if not query:
                    self.send_error(400, "Missing 'query' field")
                    return

                print(f"Received reasoning request: {query[:50]}...")

                # Execute reasoning
                # Combining context and query for the agent
                full_prompt = f"{context}\n\nQuestion: {query}" if context else query
                result = agent.reason(full_prompt)

                # Construct response
                response = {
                    "conclusion": result["conclusion"],
                    "confidence": result["confidence"],
                    "reasoning_chain": [
                        {
                            "premise": step["premise"],
                            "rule": step["rule"],
                            "conclusion": step["conclusion"],
                        }
                        for step in result["reasoning_chain"]
                    ],
                    "formal_notation": result["formal_conclusion"],
                    "warnings": result.get("warnings", []),
                }

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header(
                    "Access-Control-Allow-Origin", "*"
                )  # Allow all for local dev
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


print(f"Starting Reasoning Server on port {PORT}...")
with socketserver.TCPServer(("", PORT), ReasoningRequestHandler) as httpd:
    httpd.serve_forever()
