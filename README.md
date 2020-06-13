# LSTM

```
Przygotował: Jarosław Ciołek-Żelechowski(218386)
```

Postęp prac: [GitHub](https://github.com/Torak28/Auto_Tag)

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
Accuracy: 0.6174724342663274
Precision: 0.6246078280260899
Recall: 0.6174724342663274
F-score: 0.6179654363431908
```

![xd](https://i.imgur.com/hSYVY9g.png)

Jak widać wyniki są całkiem dobre ale występuje dość dużo błędów. Wyniki są podobne do uzyskanych przez uczestników kagglowego konkursu.


**data_real_5s.csv**

 * dane z sygnaturą czasową
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.6445497630331753
Precision: 0.49757678260609955
Recall: 0.6445497630331753
F-score: 0.5616047976653925
```

![xd](https://i.imgur.com/D4bqxFc.png)

Bardzo słabe wyniki - mała ilość próbek(~1000) plus fakt że większość z nich opisuje tylko siedzenie.

**data_real_5s_without_step_25_05.csv**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.8311525503408289
Precision: 0.8090625205503664
Recall: 0.8311525503408289
F-score: 0.8139217211742709
```

![xd](https://i.imgur.com/sba4QfD.png)

Lepsze wyniki - większa ilość próbek, niestety problemy z rozróżnianiem podobnych czynności np. siedzenie i jazda autem(podobne odczyty).

**data_real_5s_without_step_25_05_normalized.csv**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Accuracy: 0.9215701637546031
Precision: 0.9203272421196427
Recall: 0.9215701637546031
F-score: 0.9196046991366243
```

![xd](https://i.imgur.com/I2g8mCi.png)

Bardzo dobre wyniki - zwiększona została ilość próbek i ich różnorodność!

**data_real_5s_without_step_12_06.csv z magnetometrem**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.9946841330865253
Precision: 0.9946944728252107
Recall: 0.9946841330865253
F-score: 0.9946858959668322
```

![xd](https://i.imgur.com/W3Na8Hn.png)

**data_real_5s_without_step_12_06.csv bez magnetometru**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * nieznormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop
 * **Klasy**: cooking, driving, sitting, sport, walking

```
Accuracy: 0.9656367174521809
Precision: 0.9657552597675314
Recall: 0.9656367174521809
F-score: 0.9655925614857553
```

![xd](https://i.imgur.com/wOs3db2.png)

**data_real_5s_without_step_12_06.csv znormalizowane z magnetometrem**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Accuracy: 0.9906497698039775
Precision: 0.990666286511539
Recall: 0.9906497698039775
F-score: 0.9906509231757189
```

![xd](https://i.imgur.com/kdANrvP.png)

**data_real_5s_without_step_12_06.csv znormalizowane bez magnetometru**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Accuracy: 0.9669656841805496
Precision: 0.9670468280958601
Recall: 0.9669656841805496
F-score: 0.9669701535258015
```

![xd](https://i.imgur.com/5yabllq.png)

### Najlepszy wynik

**data_real_5s_without_step_12_06.csv znormalizowane z magnetometrem**

 * dane bez sygnatury czasowej(z racji jednak że przychodzą co 5s nie jest to potrzebne do LSTMa)
 * znormalizowane dane z opaski
 * **Pomiary**: akcelerometr, żyroskop, magnetometr
 * **Klasy**: cooking, driving, sitting, sport, walking


```
Test loss: 0.0921
Test accuracy: 0.967

Accuracy: 0.9906497698039775
Precision: 0.990666286511539
Recall: 0.9906497698039775
F-score: 0.9906509231757189
```

![xd](https://i.imgur.com/kdANrvP.png)

![xd](https://i.imgur.com/02ucohy.png)