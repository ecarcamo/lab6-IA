from data.loader import DataLoader
from eda.analyzer import EDAAnalyzer
from preprocessing.text_processor import TextProcessor
from app.menu import Menu
from config.settings import FILE_PATH


class MainApp:

    def __init__(self):
        self.df_original = None
        self.df_processed = None
        self.eda = None
        self.processor = None
        self.using_processed = False

    def run(self):
        self._load_data()
        self._init_modules()
        self._menu_loop()

    def _load_data(self):
        self.df_original = DataLoader.load(FILE_PATH)
        print("Dataset cargado.")

    def _init_modules(self):
        self.eda = EDAAnalyzer(self.df_original)
        self.processor = TextProcessor(self.df_original)

    def _menu_loop(self):
        while True:
            Menu.show()
            self._print_active_dataset()
            option = Menu.get_option()

            if option == "0":
                print("Saliendo...")
                break

            self._handle_option(option)

    def _print_active_dataset(self):
        label = "preprocesado" if self.using_processed else "original"
        print(f"[Dataset activo: {label}]")

    def _handle_option(self, option):
        actions = {
            "1": lambda: self.eda.show_random_messages(),
            "2": lambda: self.eda.total_messages(),
            "3": lambda: self.eda.plot_distribution(),
            "4": lambda: self.eda.plot_length_density("spam"),
            "5": lambda: self.eda.plot_length_density("ham"),
            "6": lambda: self.eda.plot_top_words("spam"),
            "7": lambda: self.eda.plot_top_words("ham"),
            "8": lambda: self.eda.plot_wordcloud("spam"),
            "9": lambda: self.eda.plot_wordcloud("ham"),
            "10": self._run_preprocessing,
            "11": self._toggle_dataset,
        }

        action = actions.get(option)

        if action:
            try:
                action()
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Opción inválida")

    def _run_preprocessing(self):
        if self.df_processed is not None:
            print("El dataset ya estaba preprocesado. Se mantiene en caché.")
        else:
            print("Ejecutando preprocesamiento...")
            self.df_processed = self.processor.process()
            print("Preprocesamiento aplicado.")

        self.using_processed = True
        self.eda = EDAAnalyzer(self.df_processed)
        print("Las próximas opciones usarán el dataset transformado.")

    def _toggle_dataset(self):
        if self.using_processed:
            self.using_processed = False
            self.eda = EDAAnalyzer(self.df_original)
            print("Cambiado a dataset ORIGINAL.")
            return

        if self.df_processed is None:
            print("Aún no se ha ejecutado el preprocesamiento (opción 10).")
            return

        self.using_processed = True
        self.eda = EDAAnalyzer(self.df_processed)
        print("Cambiado a dataset PREPROCESADO.")