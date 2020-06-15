# LSTM

![alt](https://media1.giphy.com/media/3oEjHYibHwRL7mrNyo/giphy.gif)

## Spis treści

1. Wybrany model
2. Architektura
3. Wykorzystane artykuły
4. Wykorzystane mechanizmy
5. Wyniki
6. Najlepszy wynik

### Wybrany model

Na etapie przydzielania zadań wybrałem **LSTM** jako *swój* model. Wiedziałem, że w związku z naturą danych jakie będziemy dostawać od zespołu zajmującego się samym urządzeniem(dane z sygnaturą czasową) wybrany model powinien sprawować się dobrze.

### Architektura

Model który zbudowałem jako pierwszy na podstawie danych uzyskanych z kaggla był bardzo prosty:

 * lstm(128)
 * Dense(5)

Zaproponowana architektura bardziej wynikała z faktu że była to moja pierwsza styczność z LSTM jako klasyfikatorem i podążając za dokumentacją pakietu Keras taka architektura wydała się odpowiednia.

### Wykorzystane artykuły

W pracach nad modelem korzystałem z dwóch artykułów:

 * *Human activity recognition from inertial sensor time-series using batch normalized deep LSTM recurrent networks*, Zebin, Tahmina & Peek, Niels & Casson, Alex & Sperrin, Matthew, (2018).  wraz z Rafał Kocoń
 * *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*, Francisco Javier Ordóñez * and Daniel Roggen, (2016).

Pierwsza z prac była wręcz identyczna do naszej sytuacji. Bardzo podobne dane - obrobione jako zbiór danych statystycznych o poszczególnych miernikach(dodanie żyroskopu, zamiast tylko akcelerometru). Oprócz tego wykorzystano także dwie warstwy LSTM zamiast jednej oraz dropout tuż przed ostatnią warstwą FC. Oba warianty testowane były na danych z urządzenia(w wersji znormalizowanej i nie) i odnosiły wyniki podobne lub gorsze do zastosowanej wcześniej architektury. Z tego też powodu pozostaliśmy przy wyjściowej architekturze.

Druga praca opierała się na połączeniu CNN i dwuwarstwowego LSTM. Dużą jednak różnicą były wykorzystywane przez autorów dane. Otrzymywali oni surowy odczyt akcelerometru i żyroskopu w trzech płaszczyznach, co nijak się ma do naszych danych statystycznych. Dodatkowo wykorzystany przez nich model był już opracowywany przez **Łukasza Brolla**(*chyba, tak wynika z naszego arkusza*). Pozwoliłem sobie pożyczyć wykorzystaną architekturę i puścić na niej nasze dane - niestety uzyskane wyniki były znacznie gorsze, stąd wróciliśmy do startowej architektury.

### Wykorzystane mechanizmy

Jako że pracowałem z LSTMem musiałem zmienić wymiarowość danych do trójwymiarową. Z racji tego jak wyglądały dane jakie otrzymywaliśmy, oraz ze względu na fakt że otrzymywaliśmy je jako *podsumowanie* pewnego stałego odcinka czasowego(5 sekund) zdecydowałem się na następujący kształt: (*ilość próbek*, *1 obserwacja na próbkę*, *ilość zmiennych opisujących próbkę*)

Oprócz tego wykorzystałem możliwości jakie daje pakiet *sklearn* i dzięki jego *KerasClassifier*, który potrafi przetwarzać modele kerasowe zastosowałem GridSearch do optymalizacji parametrów sieci. Ostateczne wyniki prezentują się następująco:

 * funkcja aktywacji: **relu**
 * wielkość batch: **10**
 * dropout: **0.0**
 * ilość epok: **100**
 * funkcja optymalizująca: **Adadelta**

Przestrzeń którą przeszukałem dla znalezienia tych parametrów wygląda następująco:

```py
batch_size = [None, 10, 20, 40]
epochs = [10, 50, 100, 200]
optimizer = ['Adadelta', 'Adam','SGD', 'RMSprop']
activation = ['relu', 'tanh', 'sigmoid']
dropout_rate = [0.0, 0.1, 0.2, 0.3]
```

### Wyniki

> Nie pamiętam które dane są z czego. Napiszę więc po prostu nazwę danych dla których to policzyłem, możesz to potem zedytować żeby było dobrze!

**dataset_5secondWindow%5B1%5D_normalized.csv**

 * dane z sygnaturą czasową
 * znormalizowane dane z kaggla
 * **Pomiary**: akcelerometr, żyroskop, dźwięk
 * **Klasy**: Still, Car, Train, Bus, Walking

```
Accuracy: 0.6140797285835454
Precision: 0.6198964274807803
Recall: 0.6140797285835454
F-score: 0.6133934749880708
```

![xd](https://i.imgur.com/wGLaLom.png)
![xd](https://i.imgur.com/CMhRqP4.png)

Jak widać wyniki są całkiem dobre ale występuje dość dużo błędów. Wyniki są podobne do uzyskanych przez uczestników kagglowego konkursu.


**data_real_5s.csv**

 * dane z sygnaturą czasową
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.7156398104265402
Precision: 0.7001898703060021
Recall: 0.7156398104265402
F-score: 0.6924655561698452
```

![xd]([3](https://i.imgur.com/zFQDOZx.png))
![xd](https://i.imgur.com/Ya6OXfl.png)

Bardzo słabe wyniki - mała ilość próbek(~1000) plus fakt że większość z nich opisuje tylko siedzenie.

**data_real_5s_without_step_25_05.csv**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.8256679464075845
Precision: 0.8019837507306017
Recall: 0.8256679464075845
F-score: 0.8077755176627621
```

![xd](https://i.imgur.com/fJs3d2f.png)
![xd](https://i.imgur.com/ZsMdtvz.png)

Lepsze wyniki - większa ilość próbek, niestety problemy z rozróżnianiem podobnych czynności np. siedzenie i jazda autem(podobne odczyty).

**data_real_5s_without_step_25_05_normalized.csv**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Accuracy: 0.9233722479040978
Precision: 0.9222713485808242
Recall: 0.9233722479040978
F-score: 0.9214489448164878
```

![xd](https://i.imgur.com/Hpf2MpD.png)
![xd](https://i.imgur.com/l2wagRG.png)

Bardzo dobre wyniki - zwiększona została ilość próbek i ich różnorodność!

**data_real_5s_without_step_12_06.csv z magnetometrem**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.9944468175993165
Precision: 0.994453729472423
Recall: 0.9944468175993165
F-score: 0.9944474034543244
```

![xd](https://i.imgur.com/QcXvYJP.png)
![xd](https://i.imgur.com/hGUPI1v.png)

**data_real_5s_without_step_12_06.csv bez magnetometru**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.9667283686933409
Precision: 0.9669833994537878
Recall: 0.9667283686933409
F-score: 0.9667460550751522
```

![xd](https://i.imgur.com/7tiuCqd.png)
![xd](https://i.imgur.com/RwJlmtQ.png)

**data_real_5s_without_step_12_06.csv znormalizowane z magnetometrem**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Accuracy: 0.9923109782144383
Precision: 0.9923284039832785
Recall: 0.9923109782144383
F-score: 0.9923121746829552
```

![xd](https://i.imgur.com/9ggwUe5.png)
![xd](https://i.imgur.com/WMkBNWi.png)

**data_real_5s_without_step_12_06.csv znormalizowane bez magnetometru**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Accuracy: 0.9667283686933409
Precision: 0.9668055698711079
Recall: 0.9667283686933409
F-score: 0.9667257747206576
```

![xd](https://i.imgur.com/jdFX3kU.png)
![xd](https://i.imgur.com/JcOwcY0.png)

### Najlepszy wynik

**data_real_5s_without_step_12_06.csv z magnetometrem**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.9944468175993165
Precision: 0.994453729472423
Recall: 0.9944468175993165
F-score: 0.9944474034543244
```

![xd](https://i.imgur.com/QcXvYJP.png)
![xd](https://i.imgur.com/hGUPI1v.png)
![xd](https://i.imgur.com/w4zVvaI.png)