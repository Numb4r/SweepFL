# import http.server
# import socketserver
# import os
# import zipfile
# import io
# from urllib.parse import unquote

# PORT = 8080
# LOG_DIR = "./logs"

# if not os.path.exists(LOG_DIR):
#     os.makedirs(LOG_DIR)

# class ExperimentManager(http.server.SimpleHTTPRequestHandler):
#     def do_GET(self):
#         path_parts = self.path.split('?')
#         clean_path = unquote(path_parts[0])
        
#         if clean_path == "/download_all":
#             self.send_response(200)
#             self.send_header('Content-Type', 'application/zip')
#             self.send_header('Content-Disposition', 'attachment; filename=logs_completos.zip')
#             self.end_headers()
#             buffer = io.BytesIO()
#             with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
#                 for root, _, files in os.walk(LOG_DIR):
#                     for file in files:
#                         filepath = os.path.join(root, file)
#                         z.write(filepath, os.path.relpath(filepath, LOG_DIR))
#             self.wfile.write(buffer.getvalue())
#             return

#         if clean_path.startswith("/delete/"):
#             file_to_del = clean_path.replace("/delete/", "")
#             if os.path.exists(file_to_del):
#                 os.remove(file_to_del)
#                 self.send_response(200)
#                 self.end_headers()
#                 return

#         if clean_path.startswith("/get_content/"):
#             rel_path = clean_path.replace("/get_content/", "")
#             if os.path.exists(rel_path):
#                 self.send_response(200)
#                 mime = "text/csv" if rel_path.lower().endswith(".csv") else "text/plain"
#                 self.send_header("Content-type", f"{mime}; charset=utf-8")
#                 self.send_header("Cache-Control", "no-cache, must-revalidate")
#                 self.end_headers()
#                 with open(rel_path, "rb") as f:
#                     self.wfile.write(f.read())
#             return
        
#         if clean_path == "/":
#             self.send_response(200)
#             self.send_header("Content-type", "text/html; charset=utf-8")
#             self.end_headers()
#             self.wfile.write(self.get_interface().encode())
#             return
            
#         return super().do_GET()

#     def get_interface(self):
#         files_structure = {}
#         for root, _, files in os.walk(LOG_DIR):
#             folder = os.path.relpath(root, '.')
#             valid = [f for f in files if f.endswith(('.log', '.txt', '.csv', '.out', '.py'))]
#             if valid: files_structure[folder] = sorted(valid)

#         return f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Recod Log Manager PRO</title>
#             <style>
#                 body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; overflow: hidden; }}
#                 #sidebar {{ width: 300px; min-width: 200px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; padding: 15px; flex-shrink: 0; }}
#                 #resizer {{ width: 6px; cursor: col-resize; background: #30363d; flex-shrink: 0; }}
#                 #resizer:hover {{ background: #1f6feb; }}
#                 #viewer {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }}
#                 #toolbar {{ background: #161b22; padding: 10px 20px; display: flex; justify-content: space-between; border-bottom: 1px solid #30363d; align-items: center; flex-wrap: wrap; gap: 10px; }}
#                 #content-area {{ flex-grow: 1; overflow: auto; background: #0d1117; position: relative; }}
#                 .folder-header {{ color: #58a6ff; font-weight: bold; margin: 15px 0 5px 0; font-size: 13px; }}
#                 .file-item {{ display: flex; align-items: center; margin-bottom: 2px; }}
#                 .file-name {{ flex-grow: 1; padding: 6px 10px; cursor: pointer; color: #8b949e; font-size: 13px; border-radius: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
#                 .file-name:hover {{ background: #21262d; color: #f0f6fc; }}
#                 .file-name.active {{ background: #1f6feb; color: white; }}
#                 .btn-delete {{ background: #da3633; color: white; border: none; padding: 2px 8px; border-radius: 4px; cursor: pointer; margin-left: 5px; opacity: 0.7; }}
#                 .btn-delete:hover {{ opacity: 1; }}
#                 .btn-all {{ background: #238636; color: white; width: 100%; display: block; text-align: center; margin-bottom: 20px; padding: 12px 0; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 13px; }}
#                 pre {{ font-family: 'Consolas', monospace; font-size: 13px; margin: 0; padding: 20px; white-space: pre; display: inline-block; min-width: 100%; box-sizing: border-box; }}
#                 pre.wrap {{ white-space: pre-wrap; }}
#                 table {{ border-collapse: collapse; min-width: 100%; font-size: 12px; }}
#                 th {{ background: #1f6feb; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }}
#                 td {{ padding: 8px; border: 1px solid #30363d; white-space: nowrap; }}
#                 tr:nth-child(even) {{ background: #161b22; }}
#                 .opt-group {{ display: flex; align-items: center; gap: 15px; font-size: 13px; }}
#             </style>
#         </head>
#         <body>
#             <div id="sidebar">
#                 <a href="/download_all" class="btn-all">📦 BAIXAR TUDO (.ZIP)</a>
#                 {self.generate_menu(files_structure)}
#             </div>
#             <div id="resizer"></div>
#             <div id="viewer">
#                 <div id="toolbar">
#                     <strong id="file-title">Monitor Recod</strong>
#                     <div class="opt-group">
#                         <label><input type="checkbox" id="reverse-opt" onchange="refresh()"> Inverter Ordem</label>
#                         <label><input type="checkbox" id="wrap-opt" onchange="toggleWrap()"> Quebrar Linhas</label>
#                         <select id="refresh-rate" onchange="resetTimer()">
#                             <option value="5000">5s</option>
#                             <option value="15000">15s</option>
#                             <option value="0">Pausar</option>
#                         </select>
#                     </div>
#                 </div>
#                 <div id="content-area"><div id="output"></div></div>
#             </div>
#             <script>
#                 let currentFile = "";
#                 let timer = null;

#                 const resizer = document.getElementById('resizer');
#                 const sidebar = document.getElementById('sidebar');
#                 resizer.addEventListener('mousedown', (e) => {{
#                     document.addEventListener('mousemove', resize);
#                     document.addEventListener('mouseup', stopResize);
#                 }});
#                 function resize(e) {{ sidebar.style.width = e.clientX + 'px'; }}
#                 function stopResize() {{ document.removeEventListener('mousemove', resize); }}

#                 function toggleWrap() {{
#                     const pre = document.querySelector('pre');
#                     if(pre) pre.classList.toggle('wrap', document.getElementById('wrap-opt').checked);
#                 }}

#                 function loadFile(path, element) {{
#                     currentFile = path;
#                     document.getElementById('file-title').innerText = path;
#                     document.querySelectorAll('.file-name').forEach(f => f.classList.remove('active'));
#                     element.classList.add('active');
#                     refresh();
#                 }}

#                 function deleteFile(path) {{
#                     if(confirm("Deletar arquivo?")) fetch("/delete/" + path).then(() => location.reload());
#                 }}

#                 function refresh() {{
#                     if(!currentFile) return;
#                     fetch("/get_content/" + currentFile + "?t=" + Date.now())
#                         .then(r => r.text())
#                         .then(data => {{
#                             const out = document.getElementById('output');
#                             const shouldReverse = document.getElementById('reverse-opt').checked;
                            
#                             if(currentFile.toLowerCase().endsWith('.csv')) {{
#                                 renderCsv(data, out, shouldReverse);
#                             }} else {{
#                                 let processedData = data;
#                                 if(shouldReverse) {{
#                                     processedData = data.trim().split('\\n').reverse().join('\\n');
#                                 }}
#                                 out.innerHTML = '<pre id="raw-text"></pre>';
#                                 const pre = document.getElementById('raw-text');
#                                 pre.textContent = processedData;
#                                 if(document.getElementById('wrap-opt').checked) pre.classList.add('wrap');
#                             }}
#                         }});
#                     resetTimer();
#                 }}

#                 function renderCsv(data, container, reverse) {{
#                     const lines = data.trim().split('\\n');
#                     if(lines.length === 0) return;
#                     const headers = lines[0];
#                     let rows = lines.slice(1);
#                     if(reverse) rows.reverse();
                    
#                     let html = "<table><thead><tr>" + headers.split(',').map(h => "<th>"+h+"</th>").join('') + "</tr></thead><tbody>";
#                     rows.forEach(row => {{
#                         html += "<tr>" + row.split(',').map(td => "<td>"+td+"</td>").join('') + "</tr>";
#                     }});
#                     container.innerHTML = html + "</tbody></table>";
#                 }}

#                 function resetTimer() {{
#                     if(timer) clearInterval(timer);
#                     const rate = parseInt(document.getElementById('refresh-rate').value);
#                     if(rate > 0) timer = setInterval(refresh, rate);
#                 }}
#             </script>
#         </body>
#         </html>
#         """

#     def generate_menu(self, structure):
#         out = ""
#         for folder, files in structure.items():
#             out += f'<div class="folder-header">📂 {folder}</div>'
#             for f in files:
#                 path = os.path.join(folder, f).replace("./", "")
#                 out += f'<div class="file-item"><div class="file-name" onclick="loadFile(\'{path}\', this)">{f}</div>'
#                 out += f'<button class="btn-delete" onclick="deleteFile(\'{path}\')">✕</button></div>'
#         return out

# socketserver.TCPServer.allow_reuse_address = True
# with socketserver.TCPServer(("", PORT), ExperimentManager) as httpd:
#     print(f"🚀 Dashboard Pro Ativo em http://localhost:{PORT}")
#     httpd.serve_forever()


# graficos

# import http.server
# import socketserver
# import os
# import zipfile
# import io
# from urllib.parse import unquote

# PORT = 8080
# LOG_DIR = "./logs"

# if not os.path.exists(LOG_DIR):
#     os.makedirs(LOG_DIR)

# class ExperimentManager(http.server.SimpleHTTPRequestHandler):
#     def do_GET(self):
#         path_parts = self.path.split('?')
#         clean_path = unquote(path_parts[0])
        
#         if clean_path == "/download_all":
#             self.send_response(200)
#             self.send_header('Content-Type', 'application/zip')
#             self.send_header('Content-Disposition', 'attachment; filename=logs.zip')
#             self.end_headers()
#             buffer = io.BytesIO()
#             with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
#                 for root, _, files in os.walk(LOG_DIR):
#                     for file in files:
#                         filepath = os.path.join(root, file)
#                         z.write(filepath, os.path.relpath(filepath, LOG_DIR))
#             self.wfile.write(buffer.getvalue())
#             return

#         if clean_path.startswith("/delete/"):
#             file_to_del = clean_path.replace("/delete/", "")
#             if os.path.exists(file_to_del):
#                 os.remove(file_to_del)
#                 self.send_response(200)
#                 self.end_headers()
#             return

#         if clean_path.startswith("/get_content/"):
#             rel_path = clean_path.replace("/get_content/", "")
#             if os.path.exists(rel_path):
#                 self.send_response(200)
#                 mime = "text/csv" if rel_path.lower().endswith(".csv") else "text/plain"
#                 self.send_header("Content-type", f"{mime}; charset=utf-8")
#                 self.send_header("Cache-Control", "no-cache")
#                 self.end_headers()
#                 with open(rel_path, "rb") as f:
#                     self.wfile.write(f.read())
#             return
        
#         if clean_path == "/":
#             self.send_response(200)
#             self.send_header("Content-type", "text/html; charset=utf-8")
#             self.end_headers()
#             self.wfile.write(self.get_interface().encode())
#             return
            
#         return super().do_GET()

#     def get_interface(self):
#         files_structure = {}
#         for root, _, files in os.walk(LOG_DIR):
#             folder = os.path.relpath(root, '.')
#             valid = [f for f in files if f.endswith(('.log', '.txt', '.csv', '.out', '.py'))]
#             if valid: files_structure[folder] = sorted(valid)

#         return f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Recod ML Monitor</title>
#             <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
#             <style>
#                 body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; overflow: hidden; }}
#                 #sidebar {{ width: 300px; min-width: 200px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; padding: 15px; flex-shrink: 0; }}
#                 #resizer {{ width: 6px; cursor: col-resize; background: #30363d; flex-shrink: 0; }}
#                 #viewer {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }}
#                 #toolbar {{ background: #161b22; padding: 10px 20px; display: flex; justify-content: space-between; border-bottom: 1px solid #30363d; align-items: center; }}
#                 #content-area {{ flex-grow: 1; overflow: auto; background: #0d1117; padding: 20px; }}
#                 .folder-header {{ color: #58a6ff; font-weight: bold; margin-top: 15px; font-size: 13px; }}
#                 .file-item {{ display: flex; align-items: center; margin-bottom: 2px; }}
#                 .file-name {{ flex-grow: 1; padding: 6px 10px; cursor: pointer; color: #8b949e; font-size: 13px; border-radius: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
#                 .file-name:hover {{ background: #21262d; }}
#                 .file-name.active {{ background: #1f6feb; color: white; }}
#                 .btn-all {{ background: #238636; color: white; width: 100%; display: block; text-align: center; margin-bottom: 20px; padding: 12px 0; text-decoration: none; border-radius: 6px; font-weight: bold; }}
#                 #chart-container {{ background: #161b22; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: none; border: 1px solid #30363d; }}
#                 table {{ border-collapse: collapse; min-width: 100%; font-size: 12px; background: #161b22; }}
#                 th {{ background: #1f6feb; color: white; padding: 10px; text-align: left; position: sticky; top: 0; }}
#                 td {{ padding: 8px; border: 1px solid #30363d; white-space: nowrap; }}
#                 pre {{ font-family: 'Consolas', monospace; font-size: 13px; white-space: pre; }}
#             </style>
#         </head>
#         <body>
#             <div id="sidebar">
#                 <a href="/download_all" class="btn-all">📦 BAIXAR TUDO (.ZIP)</a>
#                 {self.generate_menu(files_structure)}
#             </div>
#             <div id="resizer"></div>
#             <div id="viewer">
#                 <div id="toolbar">
#                     <strong id="file-title">Monitor Recod</strong>
#                     <div>
#                         <select id="refresh-rate" onchange="resetTimer()">
#                             <option value="5000">5s</option>
#                             <option value="15000">15s</option>
#                             <option value="0">Pausar</option>
#                         </select>
#                     </div>
#                 </div>
#                 <div id="content-area">
#                     <div id="chart-container"><canvas id="myChart"></canvas></div>
#                     <div id="output"></div>
#                 </div>
#             </div>
#             <script>
#                 let currentFile = "";
#                 let timer = null;
#                 let myChart = null;

#                 // Redimensionador lateral
#                 const resizer = document.getElementById('resizer');
#                 resizer.addEventListener('mousedown', (e) => {{
#                     document.addEventListener('mousemove', resize);
#                     document.addEventListener('mouseup', stopResize);
#                 }});
#                 function resize(e) {{ document.getElementById('sidebar').style.width = e.clientX + 'px'; }}
#                 function stopResize() {{ document.removeEventListener('mousemove', resize); }}

#                 function loadFile(path, element) {{
#                     currentFile = path;
#                     document.getElementById('file-title').innerText = path;
#                     document.querySelectorAll('.file-name').forEach(f => f.classList.remove('active'));
#                     element.classList.add('active');
#                     refresh();
#                 }}

#                 function refresh() {{
#                     if(!currentFile) return;
#                     fetch("/get_content/" + currentFile + "?t=" + Date.now())
#                         .then(r => r.text())
#                         .then(data => {{
#                             const out = document.getElementById('output');
#                             if(currentFile.toLowerCase().endsWith('.csv')) {{
#                                 renderCsvAndChart(data, out);
#                             }} else {{
#                                 document.getElementById('chart-container').style.display = 'none';
#                                 out.innerHTML = '<pre>' + data + '</pre>';
#                             }}
#                         }});
#                     resetTimer();
#                 }}

#                 function renderCsvAndChart(data, container) {{
#                     const lines = data.trim().split('\\n').map(l => l.split(','));
#                     if(lines.length < 2) return;

#                     const headers = lines[0].map(h => h.trim());
#                     const roundIdx = headers.indexOf('round');
#                     const lossIdx = headers.indexOf('loss');
#                     const accIdx = headers.indexOf('accuracy');

#                     if (roundIdx !== -1 && lossIdx !== -1) {{
#                         processChartData(lines.slice(1), roundIdx, lossIdx, accIdx);
#                     }}

#                     let html = "<table><thead><tr>" + headers.map(h => "<th>"+h+"</th>").join('') + "</tr></thead><tbody>";
#                     for(let i=1; i<lines.length; i++) {{
#                         html += "<tr>" + lines[i].map(td => "<td>"+td+"</td>").join('') + "</tr>";
#                     }}
#                     container.innerHTML = html + "</tbody></table>";
#                 }}

#                 function processChartData(rows, rIdx, lIdx, aIdx) {{
#                     const roundsMap = {{}};
#                     rows.forEach(row => {{
#                         const r = row[rIdx];
#                         const l = parseFloat(row[lIdx]);
#                         const a = parseFloat(row[aIdx]);
#                         if(!roundsMap[r]) roundsMap[r] = {{ loss: [], acc: [] }};
#                         if(!isNaN(l)) roundsMap[r].loss.push(l);
#                         if(!isNaN(a)) roundsMap[r].acc.push(a);
#                     }});

#                     const labels = Object.keys(roundsMap).sort((a,b) => a-b);
#                     const avgLoss = labels.map(r => roundsMap[r].loss.reduce((a,b) => a+b, 0) / roundsMap[r].loss.length);
#                     const avgAcc = labels.map(r => roundsMap[r].acc.reduce((a,b) => a+b, 0) / roundsMap[r].acc.length);

#                     updateChart(labels, avgLoss, avgAcc);
#                 }}

#                 function updateChart(labels, lossData, accData) {{
#                     document.getElementById('chart-container').style.display = 'block';
#                     const ctx = document.getElementById('myChart').getContext('2d');
                    
#                     if(myChart) myChart.destroy();
                    
#                     myChart = new Chart(ctx, {{
#                         type: 'line',
#                         data: {{
#                             labels: labels,
#                             datasets: [
#                                 {{ label: 'Avg Loss', data: lossData, borderColor: '#f87171', backgroundColor: 'rgba(248, 113, 113, 0.1)', yAxisID: 'y' }},
#                                 {{ label: 'Avg Accuracy', data: accData, borderColor: '#34d399', backgroundColor: 'rgba(52, 211, 153, 0.1)', yAxisID: 'y1' }}
#                             ]
#                         }},
#                         options: {{
#                             responsive: true,
#                             scales: {{
#                                 x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
#                                 y: {{ title: {{ display: true, text: 'Loss', color: '#f87171' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }},
#                                 y1: {{ position: 'right', title: {{ display: true, text: 'Accuracy', color: '#34d399' }}, ticks: {{ color: '#8b949e' }}, grid: {{ drawOnChartArea: false }} }}
#                             }},
#                             plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
#                         }}
#                     }});
#                 }}

#                 function resetTimer() {{
#                     if(timer) clearInterval(timer);
#                     const rate = parseInt(document.getElementById('refresh-rate').value);
#                     if(rate > 0) timer = setInterval(refresh, rate);
#                 }}
#             </script>
#         </body>
#         </html>
#         """

#     def generate_menu(self, structure):
#         out = ""
#         for folder, files in structure.items():
#             out += f'<div class="folder-header">📂 {folder}</div>'
#             for f in files:
#                 path = os.path.join(folder, f).replace("./", "")
#                 out += f'<div class="file-name" onclick="loadFile(\'{path}\', this)">{f}</div>'
#         return out

# socketserver.TCPServer.allow_reuse_address = True
# with socketserver.TCPServer(("", PORT), ExperimentManager) as httpd:
#     print(f"🚀 Dashboard Yuri com Gráficos: http://localhost:{PORT}")
#     httpd.serve_forever()


# Funcionando


# import http.server
# import socketserver
# import os
# import zipfile
# import io
# from urllib.parse import unquote

# PORT = 8080
# LOG_DIR = "./logs"

# if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# class ExperimentManager(http.server.SimpleHTTPRequestHandler):
#     def do_GET(self):
#         path_parts = self.path.split('?')
#         clean_path = unquote(path_parts[0])
        
#         if clean_path == "/download_all":
#             self.send_response(200)
#             self.send_header('Content-Type', 'application/zip')
#             self.send_header('Content-Disposition', 'attachment; filename=logs.zip')
#             self.end_headers()
#             buffer = io.BytesIO()
#             with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
#                 for root, _, files in os.walk(LOG_DIR):
#                     for file in files:
#                         filepath = os.path.join(root, file)
#                         z.write(filepath, os.path.relpath(filepath, LOG_DIR))
#             self.wfile.write(buffer.getvalue())
#             return

#         if clean_path.startswith("/get_content/"):
#             rel_path = clean_path.replace("/get_content/", "")
#             if os.path.exists(rel_path):
#                 self.send_response(200)
#                 mime = "text/csv" if rel_path.lower().endswith(".csv") else "text/plain"
#                 self.send_header("Content-type", f"{mime}; charset=utf-8")
#                 self.end_headers()
#                 with open(rel_path, "rb") as f: self.wfile.write(f.read())
#             return
        
#         if clean_path == "/":
#             self.send_response(200)
#             self.send_header("Content-type", "text/html; charset=utf-8")
#             self.end_headers()
#             self.wfile.write(self.get_interface().encode())
#             return
            
#         return super().do_GET()

#     def get_interface(self):
#         files_structure = {}
#         for root, _, files in os.walk(LOG_DIR):
#             folder = os.path.relpath(root, '.')
#             valid = [f for f in files if f.endswith(('.log', '.txt', '.csv', '.out'))]
#             if valid: files_structure[folder] = sorted(valid)

#         return f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Recod Analytics PRO</title>
#             <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
#             <style>
#                 body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; overflow: hidden; }}
#                 #sidebar {{ width: 280px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; padding: 15px; flex-shrink: 0; }}
#                 #viewer {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }}
#                 #toolbar {{ background: #161b22; padding: 10px 20px; display: flex; justify-content: space-between; border-bottom: 1px solid #30363d; align-items: center; }}
#                 #content-area {{ flex-grow: 1; overflow: auto; padding: 20px; }}
                
#                 /* Painel de Gráfico */
#                 #chart-panel {{ background: #1c2128; border: 1px solid #444c56; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: none; }}
#                 .config-row {{ display: flex; gap: 20px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; font-size: 13px; }}
#                 select, input {{ background: #0d1117; color: white; border: 1px solid #30363d; padding: 5px; border-radius: 4px; }}
                
#                 .file-name {{ padding: 6px 10px; cursor: pointer; color: #8b949e; font-size: 13px; border-radius: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
#                 .file-name:hover {{ background: #21262d; }}
#                 .file-name.active {{ background: #1f6feb; color: white; }}
#                 .btn-all {{ background: #238636; color: white; width: 100%; display: block; text-align: center; margin-bottom: 20px; padding: 10px 0; text-decoration: none; border-radius: 6px; font-weight: bold; }}
#                 table {{ border-collapse: collapse; min-width: 100%; font-size: 12px; margin-top: 20px; }}
#                 th {{ background: #1f6feb; color: white; padding: 8px; position: sticky; top: 0; }}
#                 td {{ padding: 6px; border: 1px solid #30363d; }}
#             </style>
#         </head>
#         <body>
#             <div id="sidebar">
#                 <a href="/download_all" class="btn-all">📦 Baixar Todos</a>
#                 {self.generate_menu(files_structure)}
#             </div>
#             <div id="viewer">
#                 <div id="toolbar">
#                     <strong id="file-title">Selecione um experimento</strong>
#                     <select id="refresh-rate" onchange="resetTimer()">
#                         <option value="5000">Auto 5s</option>
#                         <option value="0">Pausar</option>
#                     </select>
#                 </div>
#                 <div id="content-area">
#                     <div id="chart-panel">
#                         <div class="config-row">
#                             <label>Métrica (Y): <select id="y-axis" onchange="updateChart()"></select></label>
#                             <label>Eixo (X): <select id="x-axis" onchange="updateChart()"></select></label>
#                             <label>Tipo: <select id="chart-type" onchange="updateChart()">
#                                 <option value="line">Linha</option>
#                                 <option value="bar">Barras</option>
#                             </select></label>
#                             <label><input type="checkbox" id="log-scale" onchange="updateChart()"> Escala Log</label>
#                         </div>
#                         <canvas id="mainChart" style="max-height: 400px;"></canvas>
#                     </div>
#                     <div id="output"></div>
#                 </div>
#             </div>
#             <script>
#                 let currentFile = "", timer = null, myChart = null, csvData = [];

#                 function loadFile(path, el) {{
#                     currentFile = path;
#                     document.getElementById('file-title').innerText = path;
#                     document.querySelectorAll('.file-name').forEach(f => f.classList.remove('active'));
#                     el.classList.add('active');
#                     refresh(true); // true para resetar seletores de colunas
#                 }}

#                 function refresh(resetSelectors = false) {{
#                     if(!currentFile) return;
#                     fetch("/get_content/" + currentFile + "?t=" + Date.now()).then(r => r.text()).then(text => {{
#                         if(currentFile.endsWith('.csv')) {{
#                             parseCSV(text, resetSelectors);
#                             renderTable(text);
#                         }} else {{
#                             document.getElementById('chart-panel').style.display = 'none';
#                             document.getElementById('output').innerHTML = "<pre style='white-space:pre-wrap'>"+text+"</pre>";
#                         }}
#                     }});
#                     resetTimer();
#                 }}

#                 function parseCSV(text, resetSelectors) {{
#                     const lines = text.trim().split('\\n').map(l => l.split(',').map(c => c.trim()));
#                     if(lines.length < 2) return;
#                     const headers = lines[0];
#                     csvData = lines.slice(1);

#                     if(resetSelectors) {{
#                         const ySel = document.getElementById('y-axis');
#                         const xSel = document.getElementById('x-axis');
#                         ySel.innerHTML = ""; xSel.innerHTML = "";
#                         headers.forEach((h, i) => {{
#                             let isNumeric = !isNaN(parseFloat(csvData[0][i]));
#                             if(isNumeric) {{
#                                 ySel.add(new Option(h, i));
#                                 xSel.add(new Option(h, i));
#                             }}
#                         }});
#                         // Defaults: X=round (0), Y=loss (2)
#                         if(ySel.options.length > 2) ySel.selectedIndex = 2;
#                     }}
#                     document.getElementById('chart-panel').style.display = 'block';
#                     updateChart();
#                 }}

#                 function updateChart() {{
#                     const xIdx = document.getElementById('x-axis').value;
#                     const yIdx = document.getElementById('y-axis').value;
#                     const isLog = document.getElementById('log-scale').checked;
#                     const type = document.getElementById('chart-type').value;

#                     // Agrupar e tirar média
#                     const grouped = {{}};
#                     csvData.forEach(row => {{
#                         const x = row[xIdx], y = parseFloat(row[yIdx]);
#                         if(!grouped[x]) grouped[x] = [];
#                         if(!isNaN(y)) grouped[x].push(y);
#                     }});

#                     const labels = Object.keys(grouped).sort((a,b) => parseFloat(a) - parseFloat(b));
#                     const data = labels.map(l => grouped[l].reduce((a,b) => a+b, 0) / grouped[l].length);

#                     if(myChart) myChart.destroy();
#                     myChart = new Chart(document.getElementById('mainChart'), {{
#                         type: type,
#                         data: {{
#                             labels: labels,
#                             datasets: [{{ 
#                                 label: document.getElementById('y-axis').selectedOptions[0].text,
#                                 data: data, 
#                                 borderColor: '#1f6feb', 
#                                 backgroundColor: 'rgba(31, 111, 235, 0.2)',
#                                 tension: 0.1 
#                             }}]
#                         }},
#                         options: {{
#                             scales: {{ 
#                                 y: {{ type: isLog ? 'logarithmic' : 'linear', grid: {{ color: '#30363d' }} }},
#                                 x: {{ grid: {{ color: '#30363d' }} }}
#                             }}
#                         }}
#                     }});
#                 }}

#                 function renderTable(text) {{
#                     const lines = text.trim().split('\\n');
#                     let html = "<table><thead><tr>" + lines[0].split(',').map(h => "<th>"+h+"</th>").join('') + "</tr></thead><tbody>";
#                     for(let i=1; i<Math.min(lines.length, 50); i++) {{
#                         html += "<tr>" + lines[i].split(',').map(td => "<td>"+td+"</td>").join('') + "</tr>";
#                     }}
#                     document.getElementById('output').innerHTML = html + "</tbody></table>" + (lines.length > 50 ? "<p>... exibindo apenas primeiras 50 linhas</p>" : "");
#                 }}

#                 function resetTimer() {{
#                     if(timer) clearInterval(timer);
#                     const rate = parseInt(document.getElementById('refresh-rate').value);
#                     if(rate > 0) timer = setInterval(() => refresh(false), rate);
#                 }}
#             </script>
#         </body>
#         </html>
#         """

#     def generate_menu(self, structure):
#         out = ""
#         for folder, files in structure.items():
#             out += f'<div class="folder-header">📂 {folder}</div>'
#             for f in files:
#                 path = os.path.join(folder, f).replace("./", "")
#                 out += f'<div class="file-name" onclick="loadFile(\'{path}\', this)">{f}</div>'
#         return out

# socketserver.TCPServer.allow_reuse_address = True
# with socketserver.TCPServer(("", PORT), ExperimentManager) as httpd:
#     print(f"🚀 Analisador de Experimentos ativo na porta {PORT}")
#     httpd.serve_forever()




import http.server
import socketserver
import os
import zipfile
import io
from urllib.parse import unquote

PORT = 8080
LOG_DIR = "./logs"

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

class ExperimentManager(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path_parts = self.path.split('?')
        clean_path = unquote(path_parts[0])
        
        if clean_path == "/download_all":
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Disposition', 'attachment; filename=logs.zip')
            self.end_headers()
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(LOG_DIR):
                    for file in files:
                        filepath = os.path.join(root, file)
                        z.write(filepath, os.path.relpath(filepath, LOG_DIR))
            self.wfile.write(buffer.getvalue())
            return

        if clean_path.startswith("/get_content/"):
            rel_path = clean_path.replace("/get_content/", "")
            if os.path.exists(rel_path):
                self.send_response(200)
                mime = "text/csv" if rel_path.lower().endswith(".csv") else "text/plain"
                self.send_header("Content-type", f"{mime}; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(rel_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                # Adiciona uma resposta amigável caso o arquivo suma ou o caminho esteja errado
                self.send_response(404)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Erro 404: Arquivo nao encontrado no servidor.")
            return
        
        if clean_path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.get_interface().encode())
            return
            
        return super().do_GET()

    def get_interface(self):
        files_structure = {}
        for root, _, files in os.walk(LOG_DIR):
            folder = os.path.relpath(root, '.')
            valid = [f for f in files if f.endswith(('.log', '.txt', '.csv', '.out', '.py', '.err'))]
            if valid: files_structure[folder] = sorted(valid)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Recod Monitor PRO</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ margin: 0; font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; display: flex; height: 100vh; overflow: hidden; }}
                #sidebar {{ width: 300px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; padding: 15px; flex-shrink: 0; }}
                #resizer {{ width: 6px; cursor: col-resize; background: #30363d; flex-shrink: 0; }}
                #viewer {{ flex-grow: 1; display: flex; flex-direction: column; overflow: hidden; }}
                #toolbar {{ background: #161b22; padding: 10px 20px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }}
                #content-area {{ flex-grow: 1; overflow: auto; padding: 20px; }}
                
                .file-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
                .file-name {{ flex-grow: 1; padding: 6px; cursor: pointer; color: #8b949e; font-size: 13px; border-radius: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                .file-name:hover {{ background: #21262d; color: #f0f6fc; }}
                .file-name.active {{ background: #1f6feb; color: white; }}
                .add-btn {{ background: #238636; color: white; border: none; border-radius: 4px; cursor: pointer; padding: 2px 8px; margin-left: 5px; font-weight: bold; }}
                
                #comp-shelf {{ background: #1c2128; border: 1px solid #444c56; border-radius: 8px; padding: 15px; margin-bottom: 20px; display: none; }}
                .tag {{ display: inline-block; background: #30363d; padding: 2px 10px; border-radius: 15px; font-size: 11px; margin: 3px; border: 1px solid #1f6feb; }}
                
                pre {{ background: #161b22; padding: 15px; border-radius: 5px; font-family: 'Consolas', monospace; font-size: 13px; white-space: pre; overflow: auto; min-width: 100%; box-sizing: border-box; }}
                table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
                th {{ background: #1f6feb; color: white; padding: 8px; text-align: left; position: sticky; top: 0; }}
                td {{ padding: 6px; border: 1px solid #30363d; white-space: nowrap; }}
                tr:nth-child(even) {{ background: #161b22; }}
                .folder-header {{ color: #58a6ff; font-weight: bold; margin-top: 15px; font-size: 13px; }}
            </style>
        </head>
        <body>
            <div id="sidebar">
                <a href="/download_all" style="background:#238636; color:white; display:block; text-align:center; padding:10px; text-decoration:none; border-radius:5px; margin-bottom:15px; font-weight:bold;">📦 Download ZIP</a>
                {self.generate_menu(files_structure)}
            </div>
            <div id="resizer"></div>
            <div id="viewer">
                <div id="toolbar">
                    <strong id="view-title">Recod Monitor</strong>
                    <select id="refresh-rate" onchange="resetTimer()">
                        <option value="5000">Auto 5s</option>
                        <option value="0">Pausar</option>
                    </select>
                </div>
                <div id="content-area">
                    <div id="comp-shelf">
                        <div id="tags-container"></div>
                        <div style="margin: 10px 0; font-size: 12px;">
                            Y: <select id="y-sel" onchange="updateChart()"></select>
                            X: <select id="x-sel" onchange="updateChart()"></select>
                            <label><input type="checkbox" id="log-scale" onchange="updateChart()"> Log</label>
                            <button onclick="clearComparison()" style="background:#da3633; color:white; border:none; padding:2px 10px; border-radius:3px; cursor:pointer; margin-left:10px;">Limpar Gráfico</button>
                        </div>
                        <canvas id="compChart" style="max-height: 350px;"></canvas>
                    </div>
                    <div id="file-output"></div>
                </div>
            </div>
            <script>
                let currentFile = "";
                let timer = null;
                let myChart = null;
                let compareList = JSON.parse(localStorage.getItem('compareList') || '[]');

                // Resizer logic
                const resizer = document.getElementById('resizer');
                resizer.addEventListener('mousedown', () => {{
                    document.onmousemove = e => {{
                        if(e.clientX > 150 && e.clientX < 600) document.getElementById('sidebar').style.width = e.clientX + 'px';
                    }};
                    document.onmouseup = () => document.onmousemove = null;
                }});

                function loadFile(path, el) {{
                    currentFile = path;
                    document.getElementById('view-title').innerText = path;
                    document.querySelectorAll('.file-name').forEach(f => f.classList.remove('active'));
                    if(el) el.classList.add('active');
                    refresh();
                }}

                function addToCompare(path) {{
                    if(!compareList.includes(path)) {{
                        compareList.push(path);
                        localStorage.setItem('compareList', JSON.stringify(compareList));
                        renderComparison();
                    }}
                }}

                function clearComparison() {{
                    compareList = [];
                    localStorage.removeItem('compareList');
                    renderComparison();
                }}

                function renderComparison() {{
                    const shelf = document.getElementById('comp-shelf');
                    if(compareList.length === 0) {{
                        shelf.style.display = 'none';
                        return;
                    }}
                    shelf.style.display = 'block';
                    document.getElementById('tags-container').innerHTML = compareList.map(p => `<span class="tag">${{p.split('/').pop()}}</span>`).join('');
                    updateChart(true);
                }}

                async function updateChart(resetSelectors = false) {{
                    if(compareList.length === 0) return;
                    const datasets = [];
                    let commonHeaders = [];
                    const colors = ['#58a6ff', '#3fb950', '#f87171', '#d299ff', '#e3b341'];

                    for(let i=0; i < compareList.length; i++) {{
                        try {{
                            const res = await fetch("/get_content/" + compareList[i] + "?t=" + Date.now());
                            const text = await res.text();
                            const lines = text.trim().split('\\n').map(l => l.split(','));
                            if(i === 0) commonHeaders = lines[0].map(h => h.trim());
                            
                            const xIdx = document.getElementById('x-sel').value || commonHeaders.indexOf('round');
                            const yIdx = document.getElementById('y-sel').value || commonHeaders.indexOf('loss');

                            const grouped = {{}};
                            lines.slice(1).forEach(row => {{
                                const x = row[xIdx], y = parseFloat(row[yIdx]);
                                if(!grouped[x]) grouped[x] = [];
                                if(!isNaN(y)) grouped[x].push(y);
                            }});

                            const labels = Object.keys(grouped).sort((a,b) => parseFloat(a) - parseFloat(b));
                            datasets.push({{
                                label: compareList[i].split('/').pop(),
                                data: labels.map(l => ({{ x: l, y: grouped[l].reduce((a,b) => a+b, 0)/grouped[l].length }})),
                                borderColor: colors[i % colors.length],
                                tension: 0.1
                            }});
                        }} catch(e) {{ console.error(e); }}
                    }}

                    if(resetSelectors && commonHeaders.length > 0) {{
                        const yS = document.getElementById('y-sel'), xS = document.getElementById('x-sel');
                        yS.innerHTML = ""; xS.innerHTML = "";
                        commonHeaders.forEach((h, i) => {{
                            yS.add(new Option(h, i));
                            xS.add(new Option(h, i));
                        }});
                        yS.value = commonHeaders.indexOf('loss') !== -1 ? commonHeaders.indexOf('loss') : 0;
                        xS.value = commonHeaders.indexOf('round') !== -1 ? commonHeaders.indexOf('round') : 0;
                    }}

                    if(myChart) myChart.destroy();
                    myChart = new Chart(document.getElementById('compChart'), {{
                        type: 'line',
                        data: {{ datasets }},
                        options: {{ scales: {{ 
                            y: {{ type: document.getElementById('log-scale').checked ? 'logarithmic' : 'linear', grid:{{color:'#30363d'}} }},
                            x: {{ type: 'linear', grid:{{color:'#30363d'}} }}
                        }} }}
                    }});
                }}

                function refresh() {{
                    if(compareList.length > 0) updateChart(false);
                    if(!currentFile) return;
                    
                    fetch("/get_content/" + currentFile + "?t=" + Date.now())
                        .then(r => r.text()).then(data => {{
                            const out = document.getElementById('file-output');
                            // CORREÇÃO: Usar endsWith (maiúsculo)
                            if(currentFile.toLowerCase().endsWith('.csv')) {{
                                renderTable(data, out);
                            }} else {{
                                out.innerHTML = '<pre id="raw-text"></pre>';
                                document.getElementById('raw-text').textContent = data;
                            }}
                        }});
                    resetTimer();
                }}

                function renderTable(data, container) {{
                    const lines = data.trim().split('\\n');
                    if(lines.length === 0) return;
                    let html = "<table><thead><tr>" + lines[0].split(',').map(h => "<th>"+h+"</th>").join('') + "</tr></thead><tbody>";
                    lines.slice(1, 100).forEach(row => {{
                        html += "<tr>" + row.split(',').map(td => "<td>"+td+"</td>").join('') + "</tr>";
                    }});
                    container.innerHTML = html + "</tbody></table>" + (lines.length > 100 ? "<small>... exibindo as primeiras 100 linhas</small>" : "");
                }}

                function resetTimer() {{
                    if(timer) clearInterval(timer);
                    const rate = parseInt(document.getElementById('refresh-rate').value);
                    if(rate > 0) timer = setInterval(refresh, rate);
                }}

                // Init
                renderComparison();
            </script>
        </body>
        </html>
        """

    def generate_menu(self, structure):
        out = ""
        for folder, files in structure.items():
            out += f'<div class="folder-header">📂 {folder}</div>'
            for f in files:
                # 1. Cria o caminho e força o uso de barras normais (/)
                path = os.path.join(folder, f).replace("\\", "/").replace("./", "")
                
                # 2. Escapa aspas simples caso o arquivo tenha nomes como "log's.csv"
                safe_path = path.replace("'", "\\'")
                
                is_csv = f.lower().endswith('.csv')
                out += f'<div class="file-item">'
                out += f'<div class="file-name" onclick="loadFile(\'{safe_path}\', this)">{f}</div>'
                if is_csv:
                    out += f'<button class="add-btn" onclick="addToCompare(\'{safe_path}\')">+</button>'
                out += '</div>'
        return out

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), ExperimentManager) as httpd:
    print(f"🚀 Monitor Final Ativo: http://localhost:{PORT}")
    httpd.serve_forever()
