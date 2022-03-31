import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab import drive
import mpld3
from mpld3 import plugins
import seaborn as sns
#k-nn
from sklearn.neighbors import NearestNeighbors
#LOF
from sklearn.neighbors import LocalOutlierFactor
#HBOS
from pyod.models.hbos import HBOS


#verwandelt eine JSON datei in ein DataFrame und speichert sie zusätzlich als CSV ab
def data2csv(path, exportname):
    files = [path + "\\" + file for file in os.listdir(path)]
    data = []
    for file in files:
        with open(file, mode = "r") as new_file:
            new_file = eval(new_file.read())
        for entry in new_file:
            x = float(entry["x"])
            y = float(entry["y"])
            z = float(entry["z"])
            t = int(entry["t"])
            entry = [x, y, z, t]
            data.append(entry)
    df = pd.DataFrame(data)
    df.to_csv(exportname, sep = ",", index = False)
    return df

path = r"C:\Users\greta\OneDrive\Desktop\kyan\AceImportant"
data2csv(path, "ImportantCSVtest2.csv")


#Daten einlesen
acData = pd.read_csv("/drive/MyDrive/Uni/ImportantCSV.csv")
acData = acData.rename(columns={"0": "x", "1": "y", "2": "z", "3": "t"})
acData["sum"]=abs(acData["x"])+abs(acData["y"])+abs(acData["z"])

#Zeitstempel wird von Nanosekunden in Sekunden umgewandelt und dann gerundet
acData["t"] = ((acData["t"]-1619789053415523)/1000000000)
acData["t"] = [int(x) for x in acData["t"]]

#Label für die Momente in denen ich Anomalien überquert habe sind in Sekunden angegeben
anomalies = [56,60,61,62,66,68,74,76,80,81,82,83,84,85,91,92,156,158,187,188,190,194,196,210,223,224,225,226,232,234,236,257,258,274,280,292,298,337,351,355,356,357,358,359,360,380,426,523,537,538,558,560,581,583,969,972,976,1149,1153,1168,1171,1180,1194,1233,1266,1307,1317]

#Sliding Window Methode
windowSize = 750
slide = 250
window_start = 55000
window_end = windowSize + window_start
aMean = []
aDev = []
aRange = []
tMean = []
ano = []

while window_end <= len(acData):
  window = acData[window_start:window_end]

  aMinT = window["sum"].min()
  aMaxT = window["sum"].max()
  aMeanT = window["sum"].mean()
  aDevT  = np.std(window["sum"])
  aRangeT = aMaxT - aMinT
  tMeanT = int(window["t"].mean())
  anoT = 0
  if tMeanT in anomalies or tMeanT+1 in anomalies or tMeanT-1 in anomalies:
    anoT = 30

  aMean.append(aMeanT)
  aDev.append(aDevT)
  aRange.append(aRangeT)
  tMean.append(tMeanT)
  ano.append(anoT)

  window_start += slide
  window_end += slide

#DataFrames aus den Ergebnissen des Sliding Windows erstellen
df = pd.DataFrame(list(zip(aMean, aDev, aRange)))
dfa = pd.DataFrame(list(zip(aMean, aDev, aRange, ano)))

#Heatmaps kreieren für die Ergebnisse des K-NN in verschiedenen Metriken, aktuell wird F-Score ausgegeben

trh = [0.2, 0.4,0.6,0.8,1,1.5,2,3,4,8]
nn = [2,4,8,16,32,64,128,256,512,1024]
results_f1=pd.DataFrame()
results_pre=pd.DataFrame()
results_jac=pd.DataFrame()
results_fmi=pd.DataFrame()

x = df.values
acc = []
rec = []
pre = []
f1 = []
for i in range(10):

  f1 = []
  pre = []
  jac =[]
  fmi = []
  nbrs = NearestNeighbors(n_neighbors = 2**(i+1))
  nbrs.fit(x)
  distances, indexes = nbrs.kneighbors(x)
  for j in range(10):

    outlier_index = np.where(distances.mean(axis = 1) > trh[j])

    outlier_values = dfa.iloc[outlier_index]

    indexes=outlier_values.index.tolist()
    dfaMod = dfa.drop(indexes)

    fp = (outlier_values[3]==0).sum()
    tp = (outlier_values[3]!=0).sum()
    tn = (dfaMod[3]==0).sum()
    fn = (dfaMod[3]!=0).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f_score = (2 * precision * recall) / (precision + recall)
    jaccard = tp/(tp+fp+fn)
    fmiT = tp / np.sqrt((tp + fp) * (tp + fn))

    acc.append(accuracy)
    rec.append(recall)
    pre.append(precision)
    f1.append(f_score)
    jac.append(jaccard)
    fmi.append(fmiT)
  results_f1[i] = f1
  results_pre[i] = pre
  results_jac[i] = jac
  results_fmi[i] = fmi

sns.set(rc={'figure.figsize':(20,15)})
sns.heatmap(results_fmi, annot=True)


#Heatmaps kreieren für die Ergebnisse des LOF in verschiedenen Metriken, aktuell wird F-Score ausgegeben

acc = []
rec = []
pre = []
f1 = []
nn = [2,4,8,16,32,64,128,256,512,1024]
LL = [200,300,350,400,450,500,550,600,650,700]
results_f1 = pd.DataFrame()
for i in range(10):
  f1 = []
  for j in range(10):
      model1 = LocalOutlierFactor(n_neighbors = nn[i], metric = "manhattan", contamination = j/100 +0.01)
      y_pred = model1.fit_predict(df)

      outlier_index = np.where(y_pred == -1) # negative values are outliers and positives inliers

      outlier_values = dfa.iloc[outlier_index]
   #   print("n = ", i, ", c = ", j)
    #  print(len(outlier_index[0]))

      indexes=outlier_values.index.tolist()
      dfaMod = dfa.drop(indexes)

      fp = (outlier_values[3]==0).sum()
      tp = (outlier_values[3]!=0).sum()
      tn = (dfaMod[3]==0).sum()
      fn = (dfaMod[3]!=0).sum()

      accuracy = (tp + tn) / (tp + tn + fp + fn)
      recall = tp / (tp + fn)
      precision = tp / (tp + fp)
      f_score = (2 * precision * recall) / (precision + recall)

      acc.append(accuracy)
      rec.append(recall)
      pre.append(precision)
      f1.append(f_score)
  results_f1[i] = f1

sns.heatmap(results_f1, annot=True)
sns.set(rc={'figure.figsize':(20,15)})


#Heatmaps kreieren für die Ergebnisse des HBOS mit statische Bin-Größe in verschiedenen Metriken, aktuell wird F-Score ausgegeben


acc = []
rec = []
pre = []
f1 = []
results_f1 = pd.DataFrame()
for j in range(10):
  f1 = []
  for i in range(12):

    h = HBOS(n_bins = 2**j+2,contamination=((i/100)+0.01))
    h.fit(df)
    y_pred = h.predict(df)

    outlier_index = np.where(y_pred == 1)
    outlier_values = dfa.iloc[outlier_index]
    indexes=outlier_values.index.tolist()
    dfaMod = dfa.drop(indexes)
    fp = (outlier_values[3]==0).sum()
    tp = (outlier_values[3]!=0).sum()
    tn = (dfaMod[3]==0).sum()
    fn = (dfaMod[3]!=0).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f_score = (2 * precision * recall) / (precision + recall)

    acc.append(accuracy)
    rec.append(recall)
    pre.append(precision)
    f1.append(f_score)
  results_f1 [j] = f1

sns.set(rc={'figure.figsize':(20,15)})
sns.heatmap(results_f1, annot=True)


#Heatmaps kreieren für die Ergebnisse des HBOS mit dynamischer Bin-Größe in verschiedenen Metriken, aktuell wird F-Score ausgegeben

acc = []
rec = []
pre = []
f1 = []
results_f1 = pd.DataFrame()

for i in range(40):

  h = HBOS(n_bins = "auto",contamination=((i/100)+0.01))
  h.fit(df)
  y_pred = h.predict(df)

  outlier_index = np.where(y_pred == 1)
  outlier_values = dfa.iloc[outlier_index]

  indexes=outlier_values.index.tolist()
  dfaMod = dfa.drop(indexes)
  fp = (outlier_values[3]==0).sum()
  tp = (outlier_values[3]!=0).sum()
  tn = (dfaMod[3]==0).sum()
  fn = (dfaMod[3]!=0).sum()

  accuracy = (tp + tn) / (tp + tn + fp + fn)
  recall = tp / (tp + fn)
  precision = tp / (tp + fp)
  f_score = (2 * precision * recall) / (precision + recall)

  acc.append(accuracy)
  rec.append(recall)
  pre.append(precision)
  f1.append(f_score)

results_f1[0] = f1
plt.plot(f1)
