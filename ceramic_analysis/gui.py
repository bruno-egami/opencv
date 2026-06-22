import os
import sys
import threading
import subprocess
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Configurações iniciais do CustomTkinter
ctk.set_appearance_mode("System")  # Segue o tema do sistema (Dark/Light)
ctk.set_default_color_theme("blue")  # Tema de cores dos botões

class ConsoleText(ctk.CTkTextbox):
    """Componente de texto para simular um console."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(state="disabled", font=("Consolas", 12))
        
    def write(self, text):
        self.configure(state="normal")
        self.insert("end", text)
        self.see("end")  # Rolagem automática para o final
        self.configure(state="disabled")

class CeramicAnalysisGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Ceramic Analysis Pipeline - GUI")
        self.geometry("900x650")
        self.minsize(800, 600)

        # Configurar grid principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------
        # PAINEL LATERAL (Esquerda) - Menu de Ações
        # ---------------------------------------------------------
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Ceramic Analysis\nPipeline", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 20))

        # Botões de Ação
        self.btn_create_session = ctk.CTkButton(self.sidebar_frame, text="📁 Criar Sessão", command=self.cmd_create_session)
        self.btn_create_session.grid(row=1, column=0, padx=20, pady=10)

        self.btn_calibrate = ctk.CTkButton(self.sidebar_frame, text="🔧 Calibrar Câmera", command=self.cmd_calibrate)
        self.btn_calibrate.grid(row=2, column=0, padx=20, pady=10)

        self.btn_convert = ctk.CTkButton(self.sidebar_frame, text="🔄 Converter RAW", command=self.cmd_convert)
        self.btn_convert.grid(row=3, column=0, padx=20, pady=10)

        self.btn_process = ctk.CTkButton(self.sidebar_frame, text="🔍 Processar Imagens", command=self.cmd_process)
        self.btn_process.grid(row=4, column=0, padx=20, pady=10)

        self.btn_analyze = ctk.CTkButton(self.sidebar_frame, text="📊 Analisar Retração", command=self.cmd_analyze)
        self.btn_analyze.grid(row=5, column=0, padx=20, pady=10)

        self.btn_cad = ctk.CTkButton(self.sidebar_frame, text="🖥️ Comparar CAD", command=self.cmd_cad_compare)
        self.btn_cad.grid(row=6, column=0, padx=20, pady=10)

        self.btn_report = ctk.CTkButton(self.sidebar_frame, text="📄 Gerar Relatório", command=self.cmd_report)
        self.btn_report.grid(row=7, column=0, padx=20, pady=10)

        self.btn_full = ctk.CTkButton(self.sidebar_frame, text="🚀 Pipeline Completo", fg_color="green", hover_color="darkgreen", command=self.cmd_full)
        self.btn_full.grid(row=8, column=0, padx=20, pady=20)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Tema:", anchor="w")
        self.appearance_mode_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("System")

        # ---------------------------------------------------------
        # ÁREA PRINCIPAL (Direita)
        # ---------------------------------------------------------
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # --- Frame de Configurações ---
        self.config_frame = ctk.CTkFrame(self.main_frame)
        self.config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.config_frame.grid_columnconfigure(1, weight=1)

        # Linha 1: Sessão
        self.lbl_session = ctk.CTkLabel(self.config_frame, text="ID da Sessão:")
        self.lbl_session.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.entry_session = ctk.CTkEntry(self.config_frame, placeholder_text="Ex: 20250612")
        self.entry_session.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.btn_browse_session = ctk.CTkButton(self.config_frame, text="Selecionar Pasta", width=120, command=self.browse_session)
        self.btn_browse_session.grid(row=0, column=2, padx=10, pady=10)

        # Linha 2: CAD
        self.lbl_cad = ctk.CTkLabel(self.config_frame, text="Modelo CAD:")
        self.lbl_cad.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.entry_cad = ctk.CTkEntry(self.config_frame, placeholder_text="Caminho para .stl ou .step (opcional)")
        self.entry_cad.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.btn_browse_cad = ctk.CTkButton(self.config_frame, text="Procurar CAD", width=120, command=self.browse_cad)
        self.btn_browse_cad.grid(row=1, column=2, padx=10, pady=10)

        # Linha 3: Vista e Estado
        self.options_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        self.options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        
        self.lbl_view = ctk.CTkLabel(self.options_frame, text="Vista:")
        self.lbl_view.pack(side="left", padx=(0, 5))
        self.combo_view = ctk.CTkComboBox(self.options_frame, values=["both", "top", "side", "all"])
        self.combo_view.pack(side="left", padx=(0, 20))

        self.lbl_state = ctk.CTkLabel(self.options_frame, text="Estado:")
        self.lbl_state.pack(side="left", padx=(0, 5))
        self.combo_state = ctk.CTkComboBox(self.options_frame, values=["both", "wet", "dry"])
        self.combo_state.pack(side="left", padx=(0, 20))

        # Linha 4: Burr Shaver
        self.burr_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        self.burr_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        
        self.var_burr_enabled = ctk.BooleanVar(value=False)
        self.chk_burr = ctk.CTkCheckBox(self.burr_frame, text="Ativar Burr Shaver (Arredondar Quinas)", variable=self.var_burr_enabled, command=self.toggle_burr_size)
        self.chk_burr.pack(side="left", padx=(0, 20))
        
        self.lbl_burr_size = ctk.CTkLabel(self.burr_frame, text="Tamanho Kernel:")
        self.lbl_burr_size.pack(side="left", padx=(0, 5))
        
        self.entry_burr_size = ctk.CTkEntry(self.burr_frame, width=60)
        self.entry_burr_size.insert(0, "201")
        self.entry_burr_size.configure(state="disabled")
        self.entry_burr_size.pack(side="left")

        # --- Frame do Console ---
        self.console_frame = ctk.CTkFrame(self.main_frame)
        self.console_frame.grid(row=1, column=0, sticky="nsew")
        self.console_frame.grid_rowconfigure(1, weight=1)
        self.console_frame.grid_columnconfigure(0, weight=1)

        self.lbl_console = ctk.CTkLabel(self.console_frame, text="Console Output:", font=ctk.CTkFont(weight="bold"))
        self.lbl_console.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))

        self.console = ConsoleText(self.console_frame)
        self.console.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Objeto para armazenar o processo atual (para evitar concorrência)
        self.current_process = None

    # ---------------------------------------------------------
    # FUNÇÕES AUXILIARES E DE UI
    # ---------------------------------------------------------
    def toggle_burr_size(self):
        if self.var_burr_enabled.get():
            self.entry_burr_size.configure(state="normal")
        else:
            self.entry_burr_size.configure(state="disabled")

    def get_burr_args(self):
        args = []
        if self.var_burr_enabled.get():
            args.append("--burr-shaver")
            size = self.entry_burr_size.get().strip()
            if size.isdigit():
                args.extend(["--burr-size", size])
        return args

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def browse_session(self):
        folder_path = filedialog.askdirectory(title="Selecione a pasta da sessão")
        if folder_path:
            # Tenta extrair o ID da pasta (ex: "session_20250612" -> "20250612")
            folder_name = os.path.basename(folder_path)
            if folder_name.startswith("session_"):
                session_id = folder_name.replace("session_", "")
                self.entry_session.delete(0, "end")
                self.entry_session.insert(0, session_id)
            else:
                self.entry_session.delete(0, "end")
                self.entry_session.insert(0, folder_name)

    def browse_cad(self):
        file_path = filedialog.askopenfilename(
            title="Selecione o modelo CAD",
            filetypes=[("Modelos CAD", "*.stl *.step *.stp"), ("Todos os Arquivos", "*.*")]
        )
        if file_path:
            self.entry_cad.delete(0, "end")
            self.entry_cad.insert(0, file_path)

    def write_console(self, text):
        """Método thread-safe para escrever no console."""
        self.console.after(0, self.console.write, text)

    def clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def get_session_id(self):
        session_id = self.entry_session.get().strip()
        if not session_id:
            messagebox.showwarning("Aviso", "Por favor, insira ou selecione um ID de Sessão.")
            return None
        return session_id

    def get_cad_path(self):
        return self.entry_cad.get().strip()

    def set_buttons_state(self, state):
        """Desabilita ou habilita botões durante o processamento."""
        buttons = [
            self.btn_create_session, self.btn_calibrate, self.btn_convert, self.btn_process, 
            self.btn_analyze, self.btn_cad, self.btn_report, self.btn_full
        ]
        for btn in buttons:
            btn.configure(state=state)

    # ---------------------------------------------------------
    # EXECUÇÃO DO PIPELINE (SUBPROCESS)
    # ---------------------------------------------------------
    def run_command_in_thread(self, command_args):
        """Executa um comando python pipeline.py em uma thread separada."""
        if self.current_process is not None and self.current_process.poll() is None:
            messagebox.showwarning("Aviso", "Um processo já está em andamento. Aguarde o término.")
            return

        self.clear_console()
        self.set_buttons_state("disabled")
        
        # Comando base
        base_cmd = [sys.executable, "pipeline.py"]
        full_cmd = base_cmd + command_args

        self.write_console(f"Executando: {' '.join(full_cmd)}\n")
        self.write_console("-" * 60 + "\n")

        # Iniciar thread
        thread = threading.Thread(target=self._execute_subprocess, args=(full_cmd,))
        thread.daemon = True  # Termina a thread se a interface for fechada
        thread.start()

    def _execute_subprocess(self, cmd):
        # Configurar ambiente para forçar saída UTF-8 e flush
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            # Ocultar janela do console no Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True,
                env=env,
                startupinfo=startupinfo,
                cwd=os.path.dirname(os.path.abspath(__file__)) # Executar no diretório do script
            )

            # Ler a saída linha por linha em tempo real
            for line in iter(self.current_process.stdout.readline, ''):
                self.write_console(line)

            self.current_process.stdout.close()
            return_code = self.current_process.wait()

            if return_code == 0:
                self.write_console("\n" + "-" * 60 + "\n[SUCESSO] Comando concluído com êxito.\n")
            else:
                self.write_console(f"\n" + "-" * 60 + f"\n[ERRO] Comando falhou com código de saída {return_code}.\n")

        except Exception as e:
            self.write_console(f"\n[ERRO FATAL] Falha ao executar subprocesso:\n{str(e)}\n")
        finally:
            # Reabilitar botões na thread principal
            self.after(0, self.set_buttons_state, "normal")
            self.current_process = None

    # ---------------------------------------------------------
    # EVENTOS DOS BOTÕES (COMANDOS)
    # ---------------------------------------------------------
    def cmd_create_session(self):
        session = self.get_session_id()
        if session:
            self.run_command_in_thread(["create-session", "--session", session])

    def cmd_calibrate(self):
        self.run_command_in_thread(["calibrate"])

    def cmd_convert(self):
        session = self.get_session_id()
        if session:
            self.run_command_in_thread(["convert", "--session", session])

    def cmd_process(self):
        session = self.get_session_id()
        if session:
            view = self.combo_view.get()
            state = self.combo_state.get()
            args = ["process", "--session", session, "--view", view, "--state", state]
            args.extend(self.get_burr_args())
            self.run_command_in_thread(args)

    def cmd_analyze(self):
        session = self.get_session_id()
        if session:
            args = ["analyze", "--session", session]
            args.extend(self.get_burr_args())
            self.run_command_in_thread(args)

    def cmd_cad_compare(self):
        session = self.get_session_id()
        if not session:
            return
            
        cad_path = self.get_cad_path()
        if not cad_path:
            messagebox.showwarning("Aviso", "Por favor, selecione um Modelo CAD para a comparação.")
            return

        view = self.combo_view.get()
        state = self.combo_state.get()
        
        args = [
            "cad-compare", 
            "--session", session, 
            "--cad", cad_path, 
            "--view", view, 
            "--state", state
        ]
        args.extend(self.get_burr_args())
        self.run_command_in_thread(args)

    def cmd_report(self):
        session = self.get_session_id()
        if session:
            self.run_command_in_thread(["report", "--session", session])

    def cmd_full(self):
        session = self.get_session_id()
        if not session:
            return
            
        cmd_args = ["full", "--session", session]
        
        cad_path = self.get_cad_path()
        if cad_path:
            cmd_args.extend(["--cad", cad_path])
            
        cmd_args.extend(self.get_burr_args())
        self.run_command_in_thread(cmd_args)

if __name__ == "__main__":
    app = CeramicAnalysisGUI()
    app.mainloop()
