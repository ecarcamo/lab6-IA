class Menu:

    @staticmethod
    def show():
        print("""
======== MENÚ ========
1) 5 mensajes aleatorios
2) Total de mensajes
3) Distribución
4) Densidad spam
5) Densidad ham
6) Top palabras spam
7) Top palabras ham
8) WordCloud spam
9) WordCloud ham
10) Ejecutar preprocesamiento
11) Alternar dataset (original / preprocesado)
0) Salir
""")

    @staticmethod
    def get_option():
        return input("Opción: ").strip()