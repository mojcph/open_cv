# Development Environment & Workflow Constraints

## System Architecture
- **Host Machine:** Windows PC (Handles all AI logic, heavy processing, and Git controls).
- **Target Device:** Raspberry Pi (Dedicated to hardware execution and camera interfacing).
- **AI Backend:** Local Ollama instance running on Windows PC host (`localhost:11434`) or cloud fallback.
- **Workflow:** Code is written on Windows via VS Code, pushed to GitHub, and pulled/synchronized to the Raspberry Pi for execution via a VS Code Remote-SSH configuration.

## Coding Constraints
1. **Camera Constraints:** The project strictly utilizes physical Raspberry Pi cameras. All code involving image acquisition, hardware pins, or video streams MUST be compatible with the Raspberry Pi environment (e.g., utilize `cv2.VideoCapture` indexes or specific Pi camera libraries like `picamera2`).
2. **Environment Separation:** 
   - Never suggest running installation commands (`pip`, `apt`, etc.) or execution commands (`python main.py`) assuming they run on Windows. Always specify that execution happens in the Pi's terminal.
   - Do not suggest complex network tunneling frameworks to route Ollama or Copilot back to the Pi, as the coding workspace is strictly localized to the Windows host.
3. **Paths:** Pay attention to path formats. Code execution paths should conform to Linux conventions (`/home/pi/...`) rather than Windows paths when manipulating files or camera outputs.

## Prompt Guidelines
- When generating OpenCV code, optimize for Raspberry Pi hardware limitations (e.g., suggest efficient frame resolutions, avoid blocking loops, and use hardware-accelerated color conversions where applicable).
