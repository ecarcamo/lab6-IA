class Menu:

    HEADER = "======== MENÚ ========"
    OPTIONS = [
        ("1", "5 mensajes aleatorios"),
        ("2", "Total de mensajes"),
        ("3", "Distribución"),
        ("4", "Densidad spam"),
        ("5", "Densidad ham"),
        ("6", "Top palabras spam"),
        ("7", "Top palabras ham"),
        ("8", "WordCloud spam"),
        ("9", "WordCloud ham"),
        ("10", "Ejecutar preprocesamiento"),
        ("11", "Alternar dataset (original / preprocesado)"),
        ("0", "Salir"),
    ]
    PROMPT = "Opción: "

    @classmethod
    def show(cls) -> None:
        print(f"\n{cls.HEADER}")
        for option_key, option_label in cls.OPTIONS:
            print(f"{option_key}) {option_label}")

    @classmethod
    def get_option(cls) -> str:
        return input(cls.PROMPT).strip()
