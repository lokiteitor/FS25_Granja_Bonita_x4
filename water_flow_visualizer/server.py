#!/usr/bin/env python3
import http.server
import socketserver
import sys

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

class CORSHTTPRequestHandler(Handler):
    def end_headers(self):
        # Allow cross-origin requests (useful for local development)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
            
    # Change directory to the script's directory to serve its files
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            print(f"==================================================")
            print(f" Visualizador de Caída de Agua - Granja Bonita x4")
            print(f" Servidor activo en: http://localhost:{port}")
            print(f"==================================================")
            print("Presiona Ctrl+C para detener el servidor.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")

if __name__ == "__main__":
    main()
