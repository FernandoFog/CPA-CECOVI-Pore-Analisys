import customtkinter as ctk

class HelpWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Documentación / Ayuda")
        self.geometry("600x700")
        
        # Make sure it floats on top
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False)) # normalize after open

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self._add_content()

    def _add_content(self):
        # Title
        ctk.CTkLabel(self.scroll, text="Guía de Uso", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 20))

        # 1. Cargar
        self._add_section("1. Cargar Imágenes", 
            "Seleccione todas las imágenes (cortes) que componen su tomografía.\n"
            "El sistema verificará que todas tengan la misma resolución.")

        # 2. Escala
        self._add_section("2. Escala (Geometría)", 
            "Es CRÍTICO ingresar los valores correctos para obtener volúmenes reales:\n"
            "• Dimensión del pixel: Cuántos mm mide el lado de un píxel.\n"
            "• Distancia entre imágenes: El paso en Z entre cada corte.")

        # 3. ROI
        self._add_section("3. Región de Interés", 
            "Recorte la imagen para analizar solo la muestra.\n"
            "Use la 'Máscara Circular' si su muestra es cilíndrica para ignorar el aire exterior.")

        # 4. Umbral
        self._add_section("4. Umbralización", 
            "Ajuste el valor de corte (0-255).\n"
            "• Píxeles < Umbral se consideran POROS (negro).\n"
            "• Píxeles > Umbral se consideran MATERIAL (blanco/gris).\n"
            "Use la pestaña 'Binarizada' para verificar qué se está detectando.")

        # 5. Ejecución
        self._add_section("5. Análisis y Exportación", 
            "Presione 'EJECUTAR ANÁLISIS'.\n"
            "Al finalizar, podrá exportar:\n"
            "• STL: Modelo 3D de los poros internos.\n"
            "• CSV (Volúmenes 3D): Lista de poros conectados y sus volúmenes.\n"
            "• CSV (Análisis de poros): Métricas detalladas corte a corte.")

    def _add_section(self, title, text):
        ctk.CTkLabel(self.scroll, text=title, font=ctk.CTkFont(size=16, weight="bold"), anchor="w").pack(fill="x", pady=(10, 5))
        ctk.CTkLabel(self.scroll, text=text, anchor="w", justify="left").pack(fill="x", pady=(0, 10))

        # Separator (visual)
        ctk.CTkFrame(self.scroll, height=2, fg_color="gray").pack(fill="x", pady=5)
