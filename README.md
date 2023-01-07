# UMA-ID3

## Requirements
* sudo python3 -m pip install wheel
* sudo python3 -m pip install pandas
* sudo python3 -m pip install sklearn
* sudo python3 -m pip install sklearn-learn

## Example
```
python <filename> <predicted_atribute> <output_file> <times> <test>
```
Available test values:
* 1 (Ordinary test)
* 2 (DecisionTreeClassifier test)
* 3 (Over-sampling test) TODO
* 4 (Under-sampling test) TODO
* 5 (Added weights test)

```
python3 main.py input/play.csv Play output 1 1
```
