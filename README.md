# **Grab Challenge - Traffic Management**

## **PROBLEM STATEMENT**

Economies in Southeast Asia are turning to AI to solve traffic congestion, which hinders mobility and economic growth. The first step in the push towards alleviating traffic congestion is to understand travel demand and travel patterns within the city.


Can we accurately forecast travel demand based on historical Grab bookings to predict areas and times with high travel demand?

In this challenge, I aim to build a model trained on a historical demand dataset, that can forecast demand on a Hold-out test dataset. The model should be able to accurately forecast ahead by T+1 to T+5 time intervals (where each interval is 15-min) given all data up to time T.

## **Generating predictions**

1) Open terminal and run pip3 install -r requirements.txt 

2) Run python3 generate_predictions.py --file < test set csv file path > 

The script will output a csv file of each geohash and its respective T+1 to T+5 predictions, the format is as shown:

| Geohash6 | day | Hour | Minute | Predicted Demand |
| -------- | --- | ---- | ------ | ---------------- |
|    ..    | ..  | ..   |  ..    |      ..          |


## **Training**

1) Open terminal and run pip3 install -r requirements.txt 

2) Put the dataset csv file into the Data/ folder.

3) Check the parameters in Config.py to match your preferences/settings.

4) Run preprocess_data.py in the Utils folder.

5) Run train.py