# hpc

genetic algorithm

На вход подаем:

const int NumberOfPoint = 500; //количество точек

const int NumberOfIndividov = 1000; //кол-во индивидов в выборке

const int MathMutation = 5; //мутации 

const double dispersionMutation = 5.0f; //максимальная мутация

const int powCount = 3; - степень полинома

const double randMaxCount = 20.0f; //максимальный разброс рандома

const int KolOfPokoleni = 30; //максимальное кол-во поколений

Задача алгоритма: 
1) Создание случайного набора точек 
2) Генерация поколения
3) Вычисление ошибки
4) Сменить поколение. Те, что больше половины - зануляются и в результате мутации заполняются случайным образом данными лучшего индивида

Время расчета на  

 CPU 2074
 
 GPU 344
 
 разница 6 раз.
 
