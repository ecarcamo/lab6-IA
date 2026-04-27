from app.menu import Menu
from config.settings import FILE_PATH, HAM_LABEL, SPAM_LABEL
from data.loader import DataLoader
from eda.analyzer import EDAAnalyzer
from preprocessing.text_processor import TextProcessor


class MainApp:

    EXIT_OPTION = "0"
    DATASET_ORIGINAL = "original"
    DATASET_PROCESSED = "preprocesado"

    def __init__(self):
        self.original_dataset = None
        self.processed_dataset = None
        self.analyzer = None
        self.processor = None
        self.is_using_processed = False

    def run(self) -> None:
        self._load_data()
        self._init_modules()
        self._menu_loop()

    def _load_data(self) -> None:
        self.original_dataset = DataLoader.load(FILE_PATH)
        print("Dataset cargado.")

    def _init_modules(self) -> None:
        self.analyzer = EDAAnalyzer(self.original_dataset)
        self.processor = TextProcessor(self.original_dataset)

    def _menu_loop(self) -> None:
        while True:
            Menu.show()
            self._print_active_dataset()
            selected_option = Menu.get_option()

            if selected_option == self.EXIT_OPTION:
                print("Saliendo...")
                break

            self._handle_option(selected_option)

    def _print_active_dataset(self) -> None:
        active_dataset_label = self._active_dataset_label()
        print(f"[Dataset activo: {active_dataset_label}]")

    def _active_dataset_label(self) -> str:
        if self.is_using_processed:
            return self.DATASET_PROCESSED
        return self.DATASET_ORIGINAL

    def _handle_option(self, option: str) -> None:
        action = self._get_action_for(option)
        if action is None:
            print("Opción inválida")
            return
        self._execute_action_safely(action)

    def _get_action_for(self, option: str):
        return self._build_action_map().get(option)

    def _build_action_map(self) -> dict:
        return {
            "1": self.analyzer.show_random_messages,
            "2": self.analyzer.total_messages,
            "3": self.analyzer.plot_distribution,
            "4": lambda: self.analyzer.plot_length_density(SPAM_LABEL),
            "5": lambda: self.analyzer.plot_length_density(HAM_LABEL),
            "6": lambda: self.analyzer.plot_top_words(SPAM_LABEL),
            "7": lambda: self.analyzer.plot_top_words(HAM_LABEL),
            "8": lambda: self.analyzer.plot_wordcloud(SPAM_LABEL),
            "9": lambda: self.analyzer.plot_wordcloud(HAM_LABEL),
            "10": self._run_preprocessing,
            "11": self._toggle_dataset,
        }

    @staticmethod
    def _execute_action_safely(action) -> None:
        try:
            action()
        except Exception as error:
            print(f"Error: {error}")

    def _run_preprocessing(self) -> None:
        if self._is_already_preprocessed():
            print("El dataset ya estaba preprocesado. Se mantiene en caché.")
        else:
            self._build_processed_dataset()

        self._switch_to_processed()
        print("Las próximas opciones usarán el dataset transformado.")

    def _is_already_preprocessed(self) -> bool:
        return self.processed_dataset is not None

    def _build_processed_dataset(self) -> None:
        print("Ejecutando preprocesamiento...")
        self.processed_dataset = self.processor.process()
        print("Preprocesamiento aplicado.")

    def _toggle_dataset(self) -> None:
        if self.is_using_processed:
            self._switch_to_original()
            print("Cambiado a dataset ORIGINAL.")
            return

        if not self._is_already_preprocessed():
            print("Aún no se ha ejecutado el preprocesamiento (opción 10).")
            return

        self._switch_to_processed()
        print("Cambiado a dataset PREPROCESADO.")

    def _switch_to_original(self) -> None:
        self.is_using_processed = False
        self.analyzer = EDAAnalyzer(self.original_dataset)

    def _switch_to_processed(self) -> None:
        self.is_using_processed = True
        self.analyzer = EDAAnalyzer(self.processed_dataset)
