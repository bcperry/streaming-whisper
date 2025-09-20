from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import io
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from TTS import MicWhisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Store transcription callbacks per client
client_callbacks = {}

html = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <title>WebSocket Audio + Text Demo</title>
    <style>
        body { font-family: system-ui, Arial, sans-serif; margin: 1.5rem; }
        #status { font-size: .9rem; color: #555; }
        button { margin-right: .5rem; }
        #messages { list-style: none; padding-left: 0; }
        #messages li { padding: 2px 4px; border-bottom: 1px solid #eee; font-family: monospace; }
        .audio { color: #0a6; }
        .error { color: #c00; }
    </style>
</head>
<body>
    <h1>WebSocket Audio + Text Demo</h1>
    <h2>Client ID: <span id=\"ws-id\"></span></h2>

    <section>
        <h3>Text Chat</h3>
        <form onsubmit=\"sendMessage(event)\">
            <input type=\"text\" id=\"messageText\" autocomplete=\"off\" placeholder=\"Type a message...\" />
            <button type=\"submit\">Send</button>
        </form>
    </section>

    <section>
        <h3>Microphone Streaming</h3>
        <div id=\"audio-controls\">
            <button id=\"startBtn\" type=\"button\">Start Mic</button>
            <button id=\"stopBtn\" type=\"button\" disabled>Stop Mic</button>
        </div>
        <div id=\"status\">Idle</div>
        <p style=\"max-width:680px;font-size:.85rem;\">This page requests microphone access, records short Opus chunks (MediaRecorder) and sends them as binary WebSocket messages. The server currently just acknowledges receipt. Open DevTools Network panel to inspect frame traffic.</p>
    </section>

    <h3>Messages</h3>
    <ul id=\"messages\"></ul>

    <script>
        const clientId = Date.now();
        document.getElementById('ws-id').textContent = clientId;

        // Use the current host dynamically (supports non-localhost / https)
        const wsProto = (location.protocol === 'https:') ? 'wss' : 'ws';
        const ws = new WebSocket(`${wsProto}://${location.host}/ws/${clientId}`);

        const messagesEl = document.getElementById('messages');
        const statusEl = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        function addMessage(text, cls) {
            const li = document.createElement('li');
            if (cls) li.className = cls;
            li.textContent = text;
            messagesEl.appendChild(li);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        ws.addEventListener('open', () => addMessage('[ws] connected'));
        ws.addEventListener('close', () => addMessage('[ws] closed', 'error'));
        ws.addEventListener('error', (e) => addMessage('[ws] error', 'error'));
        ws.addEventListener('message', ev => {
            addMessage(ev.data);
        });

        function sendMessage(ev) {
            ev.preventDefault();
            const input = document.getElementById('messageText');
            if (input.value.trim()) ws.send(input.value.trim());
            input.value = '';
        }

        // --- Microphone streaming ---
        let audioContext = null;
        let sourceNode = null;
        let processorNode = null;
        let audioStream = null;
        let chunkCount = 0;

        async function startMic() {
            if (audioContext) return; // already running
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
            } catch (err) {
                addMessage('Microphone permission denied: ' + err, 'error');
                return;
            }

            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            sourceNode = audioContext.createMediaStreamSource(audioStream);
            
            // Create a ScriptProcessorNode to capture raw PCM data
            const bufferSize = 4096; // ~256ms at 16kHz
            processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            processorNode.onaudioprocess = function(event) {
                if (ws.readyState === WebSocket.OPEN) {
                    const inputBuffer = event.inputBuffer;
                    const channelData = inputBuffer.getChannelData(0); // mono
                    
                    // Convert Float32Array to ArrayBuffer
                    const buffer = new ArrayBuffer(channelData.length * 4); // 4 bytes per float32
                    const view = new Float32Array(buffer);
                    view.set(channelData);
                    
                    chunkCount++;
                    ws.send(buffer); // send raw PCM as binary
                }
            };
            
            sourceNode.connect(processorNode);
            processorNode.connect(audioContext.destination);

            addMessage('[audio] Raw PCM streaming started (16kHz mono)');
            statusEl.textContent = 'Streaming...';
            startBtn.disabled = true;
            stopBtn.disabled = false;
        }

        function stopMic() {
            if (!audioContext) return;
            
            if (processorNode) {
                processorNode.disconnect();
                processorNode = null;
            }
            if (sourceNode) {
                sourceNode.disconnect();
                sourceNode = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            if (audioStream) {
                audioStream.getTracks().forEach(t => t.stop());
                audioStream = null;
            }
            
            statusEl.textContent = 'Idle';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        startBtn.addEventListener('click', startMic);
        stopBtn.addEventListener('click', stopMic);

        // Expose for console debugging (optional)
        window.__ws = ws;
    </script>
</body>
</html>
"""


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Configuration for transcription storage
TRANSCRIPTION_STORAGE_DIR = "transcriptions"
# Ensure transcription storage directory exists
Path(TRANSCRIPTION_STORAGE_DIR).mkdir(exist_ok=True)

def asses_and_act(client_id: int, text: str):
    """
    Function to assess the transcription text and perform actions.
    This function will call the action agent to determine if the primary agent needs to be invoked.
    Args:
        client_id: The ID of the client that generated the transcription
        text: The final transcription text
    """
    logger.info(f"Assessing transcription for client {client_id}: {text}")
    # Example action: Log the text length
    

async def handle_final_transcription(client_id: int, text: str):
    """
    Handle final transcription text for a client.
    Calls the Agent to perform the appropriate action based on the transcription.
    Then calls the helper method to save the transcription to a file.

    Args:
        client_id: The ID of the client that generated the transcription
        text: The final transcription text
    """
    logger.info(f"Handling final transcription for client {client_id}: {text}")
    # Call the helper method to save the transcription
    await save_to_file(client_id, text)
    asses_and_act(client_id, text)
    



async def save_to_file(client_id: int, text: str):
    """
    Handle final transcription text for a client.
    Saves the transcription to a client-specific JSON file with timestamp.
    
    Args:
        client_id: The ID of the client that generated the transcription
        text: The final transcription text
    """
    logger.info(f"Processing final transcription for client {client_id}: {text}")
    
    try:
        # Create filename based on client ID
        filename = f"{client_id}.json"
        filepath = Path(TRANSCRIPTION_STORAGE_DIR) / filename
        
        # Prepare the transcription entry with timestamp
        timestamp = datetime.now().isoformat()
        
        # Load existing data or create new structure
        data = {
            "all_text": "",
            "utterances": []
        }
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Ensure structure exists for backward compatibility
                if "all_text" not in data:
                    data["all_text"] = ""
                if "utterances" not in data:
                    data["utterances"] = []
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not read existing transcriptions for client {client_id}: {e}")
                data = {
                    "all_text": "",
                    "utterances": []
                }
        
        # Create new utterance entry
        new_utterance = {
            "timestamp": timestamp,
            "text": text
        }
        
        # Append to utterances
        data["utterances"].append(new_utterance)
        
        # Append to all_text with space separator (if not first utterance)
        if data["all_text"]:
            data["all_text"] += " " + text
        else:
            data["all_text"] = text
        
        # Save back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcription for client {client_id} to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving transcription for client {client_id}: {e}")
    
    # Future enhancement: Add cloud storage support here
    # This could be extended to also save to Azure Blob Storage, AWS S3, etc.
    # Example structure:
    # if CLOUD_STORAGE_ENABLED:
    #     await save_to_cloud_storage(client_id, text, timestamp)

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    """Accepts both text (chat) and binary (audio chunks) frames.

    Binary frames are currently acknowledged with a short text summary.
    This is a placeholder where you can integrate real-time transcription
    (e.g., streaming Whisper) later.
    """
    await manager.connect(websocket)
    
    # Set up transcription callback for this client
    async def send_transcription(text: str, is_final: bool):
        prefix = "[final]" if is_final else "[interim]"
        logger.info(f"Sending transcription to client {client_id}: {prefix} {text}")
        try:
            # Call helper method for final transcriptions
            if is_final:
                await handle_final_transcription(client_id, text)
            await manager.send_personal_message(f"{prefix} {text}", websocket)
        except Exception as e:
            logger.error(f"Error sending transcription to client {client_id}: {e}")
    
    # Store callback for this client and create a new transcriber instance
    client_callbacks[client_id] = send_transcription
    client_transcriber = MicWhisper()
    client_transcriber.transcription_callback = send_transcription
    logger.info(f"Set transcription callback for client {client_id}")
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            
            # message structure: {'type': 'websocket.receive', 'text': '...', 'bytes': b'...'}
            if message.get("text") is not None:
                data = message["text"].strip()
                if not data:
                    continue
                await manager.send_personal_message(f"You wrote: {data}", websocket)
                await manager.broadcast(f"Client #{client_id} says: {data}")
            elif message.get("bytes") is not None:
                audio_bytes = message["bytes"]
                size_kb = len(audio_bytes) / 1024

                try:
                    # Convert raw Float32 PCM bytes to numpy array
                    # The client sends Float32Array as ArrayBuffer
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Ensure shape (frames, channels) - add channel dimension for mono
                    audio_data = audio_data.reshape(-1, 1)
                    
                    # Data is already in correct format:
                    # - 16kHz sample rate (from client)
                    # - Float32 [-1, 1] range
                    # - Shape (frames, channels)

                    client_transcriber.on_input(audio_data)
                    # Transcription results will be sent via the callback

                except Exception as e:
                    try:
                        await manager.send_personal_message(
                            f"[server] error processing audio: {str(e)}", websocket
                        )
                    except:
                        # WebSocket might be closed, ignore error
                        pass
            # Ignore other message types (pings, etc.)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        # Clean up transcription callback when client disconnects
        if client_id in client_callbacks:
            del client_callbacks[client_id]
        client_transcriber.transcription_callback = None
        try:
            manager.disconnect(websocket)
        except ValueError:
            # Already removed from connections
            pass
        await manager.broadcast(f"Client #{client_id} left the chat")