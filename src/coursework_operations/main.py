from fastapi import FastAPI

from src.coursework_operations.handlers.experiment1_handler import experiment1_handler
from src.coursework_operations.handlers.experiment2_handler import experiment2_handler
from src.coursework_operations.handlers.experiment3_handler import experiment3_handler
from src.coursework_operations.handlers.experiment4_handler import experiment4_handler
from src.coursework_operations.handlers.solve_handler import solve_handler

app = FastAPI()

app.add_api_websocket_route("/ws/solve", solve_handler)
app.add_api_websocket_route("/ws/experiment1", experiment1_handler)
app.add_api_websocket_route("/ws/experiment2", experiment2_handler)
app.add_api_websocket_route("/ws/experiment3", experiment3_handler)
app.add_api_websocket_route("/ws/experiment4", experiment4_handler)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
